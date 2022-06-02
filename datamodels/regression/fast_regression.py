import numpy as np
from tqdm import tqdm
import gc
import torch as ch
from threading import Thread
from .kernels import fast_threshold, avg_grad_update, normalize

def get_num_examples(loader, train_mode):
    largest_ind, n_ex = 0, 0.
    for bool_X, y, idx in loader:
        bool_X.logical_not_()
        n_ex += float(bool_X.shape[0]) if not train_mode else bool_X.sum(0).float()
        largest_ind = max(largest_ind, idx.max().cpu().item())
    
    return largest_ind, n_ex

def eval_saga(weight, bias, loader, *, pct, 
                batch_size, num_inputs, num_outputs, train_mode=False):
    _, n_ex = get_num_examples(loader, train_mode)

    residual = ch.zeros((batch_size, num_outputs), dtype=ch.float32, device=weight.device)
    total_loss = ch.zeros(num_outputs, dtype=ch.float32, device=weight.device)
    X = ch.empty(batch_size, num_inputs, dtype=ch.float32, device=weight.device)
    mm_mu, mm_sig = pct, np.sqrt(pct * (1-pct)).item()

    total_loss[:] = 0.
    for bool_X, y, idx in tqdm(loader):
        if bool_X.shape[0] != X.shape[0]:
            continue
        # Previous residuals
        X.copy_(bool_X)
        normalize(X, mm_mu, mm_sig, X)
        bool_X.logical_not_()

        # Compute residuals
        y -= bias
        ch.addmm(input=y, mat1=X, mat2=weight, out=residual, beta=-1)
        if train_mode: 
            residual *= bool_X
            residual += ch.diag(weight)

        residual.pow_(2)
        losses = residual.sum(0)
        total_loss.add_(losses, alpha=0.5)

    return total_loss / n_ex

def tensor_factory(dtype, device):
    def make_tensor(*shape):
        print(shape)
        return ch.zeros(shape, dtype=dtype, device=device)
    return make_tensor

def train_saga(weight, bias, loader, val_loader, *, lr, pct, 
               start_lams, lam_decay, end_lams, 
               early_stop_freq=2, early_stop_eps=1e-5, 
               train_mode=False, verbose=True):
    largest_ind, n_ex = get_num_examples(loader, train_mode)
    zeros = tensor_factory(ch.float32, weight.device)
    lam = start_lams.clone()
    batch_size, num_inputs, num_outputs = 0, 0, 0 
    for X, y, _ in loader:
        batch_size, num_inputs, num_outputs = y.shape[0], X.shape[1], y.shape[1]

    a_table = zeros(largest_ind + batch_size, num_outputs).cpu().pin_memory()
    shuttle = zeros(batch_size, num_outputs).cpu().pin_memory()

    w_grad_avg = zeros(*weight.shape)
    w_saga = zeros(*weight.shape)
    b_grad_avg = zeros(*bias.shape)

    residual = zeros(batch_size, num_outputs)
    total_loss, total_loss_prev = zeros(num_outputs), zeros(num_outputs)
    total_loss += 10
    
    best_losses, best_lambdas = zeros(num_outputs), zeros(num_outputs)
    best_lambdas -= 1
    best_losses += float('inf')

    X = zeros(batch_size, num_inputs)
    mm_mu, mm_sig = pct, np.sqrt(pct * (1-pct)).item()
    t = 0
    INF = ch.tensor(ch.inf).double().cuda()
    while True:
        iterator = tqdm(loader)
        if early_stop_freq and t % early_stop_freq == 0: 
            total_loss_prev.copy_(total_loss)
        total_loss[:] = 0.
        thr = None
        for bool_X, y, idx in iterator:
            # Previous residuals
            a_prev = a_table[idx].cuda(non_blocking=True)
            X.copy_(bool_X)
            normalize(X, mm_mu, mm_sig, X)
            bool_X.logical_not_()

            # Effective batch size
            B = bool_X.sum(0).float() if train_mode else float(batch_size)

            # Compute residuals
            y -= bias
            ch.addmm(input=y, mat1=X, mat2=weight, out=residual, beta=-1)
            if train_mode: 
                residual *= bool_X
                residual += ch.diag(weight)

            residual -= a_prev
            ch.mm(X.T, residual, out=w_saga)
            if train_mode:
                w_saga[ch.arange(w_saga.shape[0]), ch.arange(w_saga.shape[0])] += residual.sum(0)

            w_saga /= B
            w_saga += w_grad_avg
            b_saga = residual.sum(0) / B
            b_saga += b_grad_avg

            # Gradient steps for weight
            weight.add_(w_saga, alpha=-lr)
            bias.add_(b_saga, alpha=-lr)

            # update table and averages
            residual += a_prev

            # Move data to the residual while other stuff happens, don't
            # really need it until the next iteration
            if thr is not None: thr.join()
            def do_work(_idx): a_table.index_copy_(0, _idx, shuttle)
            shuttle.copy_(residual, non_blocking=True)
            thr = Thread(target=do_work, args=(idx.cpu(),))
            thr.start()
            
            # Update average gradients
            avg_grad_update(w_grad_avg, w_saga, B, n_ex)
            avg_grad_update(b_grad_avg, b_saga, B, n_ex)

            # Thresholding operation
            fast_threshold(weight, lr * lam)

            residual.pow_(2)
            losses = residual.sum(0)
            total_loss.add_(losses, alpha=0.5)

        w_cpu = weight.cpu()
        total_loss /= n_ex 
        total_loss += lam * ch.norm(weight, p=1, dim=0)

        # Measure progress
        if early_stop_freq and t % early_stop_freq == early_stop_freq - 1:
            done_optimizing = (total_loss >= total_loss_prev - early_stop_eps)
            if val_loader is not None:
                val_losses = eval_saga(weight, bias, val_loader, pct=pct, 
                                       batch_size=batch_size, num_inputs=num_inputs,
                                       num_outputs=num_outputs, train_mode=train_mode)
                # Find indices that (a) we're done with and (b) val loss is better
                val_loss_improved = (val_losses < best_losses)
                val_loss_improved |= (val_losses < 0)
                val_loss_improved &= done_optimizing

                best_losses = ch.where(val_loss_improved, val_losses, best_losses)
                best_lambdas = ch.where(val_loss_improved, lam, best_lambdas)

            lam_decays = ch.where(done_optimizing, lam_decay, 1.)
            total_loss += (done_optimizing * 100)
            lam *= lam_decays
            if verbose:
                curr_eps = (lam / start_lams)
                print(f'Avg eps: {curr_eps.mean():.5f} | '
                      f'% Done: {(lam < end_lams).float().mean():.2f}')
            if ch.all(lam < end_lams): 
                break

        nnz = (ch.abs(w_cpu) > 1e-5).sum(0).float().mean().item()
        total = weight.shape[0]
        if verbose and (t % verbose) == 0:
            print(f"obj (index 0): {total_loss.cpu()[4].item():.5f} | "
                  f"obj {total_loss.cpu().mean().item():.5f} | "
                  f"weight nnz {nnz}/{total} ({nnz/total:.4f})")
        t += 1

    return best_lambdas 