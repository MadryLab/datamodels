import os
from argparse import ArgumentParser
from typing import Optional

import numpy as np
import torch as ch

from fastargs import Param, Section
from fastargs import get_current_config
from fastargs.decorators import param, section
from fastargs.validation import And, OneOf
from ffcv.fields.basics import IntDecoder
from ffcv.fields.ndarray import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Squeeze, ToDevice, ToTensor

from .fast_regression import eval_saga, train_saga

Section('data', 'source data info').params(
    pct=Param(float, 'Percenter to analyze', required=True),
    data_path=Param(str, 'Path to beton file', required=True),
    num_train=Param(int, 'Number of models for training', required=True),
    num_val=Param(int, 'Number of models for validation', required=True),
    split=Param(And(str, OneOf(['train', 'test'])), 'Which data we are computing on', required=True),
)

Section('cfg', 'arguments to give the writer').params(
    k=Param(int, 'Number of lambdas on the regularization path', required=True),
    eps=Param(float, 'Multiplicative gap between top and bottom lambda', default=0.0001),
    epochs=Param(int, 'Number of epochs to run', default=100),
    batch_size=Param(int, 'Batch size for regression', required=True),
    out_dir=Param(str, 'Where to write', required=True),
    num_workers=Param(int, 'Number of workers to use for dataloading', default=8),
    true_random=Param(bool, 'Whether to use RANDOM instead of QUASI_RANDOM', is_flag=True)
)

Section('early_stopping', 'arguments specific to early stopping').params(
    check_every=Param(int, 'How often to check for improvement', default=2),
    eps=Param(float, '(Additive) improvement required at every check', default=1e-5)
)

# Calculate maximum regularization
def calc_max_lambda(loader):
    n, y_sum = 0., 0.
    # calculate mean
    for X, y, _ in loader:
        y_sum += y.sum(dim=0).float()
        n += y.shape[0]
    y_bar = y_sum / n

    # calculate maximum regularization
    inner_products = 0
    for X, y, _ in loader:
        y_map = (y - y_bar)
        inner_products += X.T.float().mm(y_map)
    return inner_products.abs().max(dim=0).values / n

@param('data.data_path')
@param('cfg.num_workers')
@param('cfg.true_random')
@param('cfg.batch_size')
def make_loader(subset, data_path=None, num_workers=None,
                true_random=False, drop_last=True, batch_size=None):
    return  Loader(data_path,
                batch_size=batch_size,
                num_workers=num_workers,
                order=(OrderOption.RANDOM if true_random else OrderOption.QUASI_RANDOM),
                indices=subset,
                drop_last=drop_last,
                os_cache=True,
                pipelines={
                    'mask': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'))],
                    'targets': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'))],
                    'idx': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'))]
                })

@param('data.num_train')
@param('data.num_val')
def make_loaders(num_train, num_val):
    return make_loader(subset=np.arange(num_train), drop_last=False), \
           make_loader(subset=np.arange(num_train)), \
           make_loader(subset=np.arange(num_train, num_train + num_val), drop_last=False), \
           make_loader(subset=np.arange(num_train + num_val))

def get_dims(loader):
    masks, targets, _ = next(iter(loader))
    return masks.shape[1], targets.shape[1]

@section('data')
@param('pct')
@param('split')
@section('cfg')
@param('k')
@param('eps')
@param('epochs')
@param('out_dir')
@section('early_stopping')
@param('check_every', alias='early_stop_freq')
@param('eps', alias='early_stop_eps')
def main(pct: float, split: str, k: int, eps: float,
         epochs: int, out_dir: str, resume_pt: Optional[str] = None,
         early_stop_freq: int = None, early_stop_eps: float = None):
    # TODO: read in_dim and out_dim from the mmap directly
    reg_loader, train_loader, val_loader, full_loader = make_loaders()
    max_lam = calc_max_lambda(reg_loader)

    n_train, n_targets = get_dims(val_loader)
    weight = ch.zeros(n_train, n_targets).cuda()
    bias = ch.zeros(n_targets).cuda()

    train_mode = (split == 'train')
    saga_args = {
        'lr': 0.01,
        'verbose': True,
        'start_lams': max_lam,
        'end_lams': max_lam * eps,
        'lam_decay': eps ** (1/k),
        'pct': pct,
        'train_mode': train_mode,
        'early_stop_freq': early_stop_freq,
        'early_stop_eps': early_stop_eps
    }
    best_lams = train_saga(weight, bias, train_loader, val_loader, **saga_args)
    saga_args['start_lams'] = best_lams
    saga_args['end_lams'] = best_lams
    saga_args['early_stop_eps'] = 0.
    print('Rerunning with best values of lambda')
    train_saga(weight, bias, full_loader, None, **saga_args)
    ch.cuda.empty_cache()

    ch.save({
        'lam': best_lams.cpu(),
        'k': k,
        'eps': eps,
        'epochs': epochs,
        'max_lam': max_lam,
        'resume_pt': resume_pt
    }, os.path.join(out_dir, f'datamodels-meta.pt'))
    ch.save({
        'lam': best_lams.cpu(),
        'weight': weight.cpu(),
        'bias': bias.cpu(),
    }, os.path.join(out_dir, f'datamodels.pt'))

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()