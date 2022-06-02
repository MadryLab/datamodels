from cupy import ElementwiseKernel

kernel = ElementwiseKernel(
    'float32 data, float32 lamb',
    'float32 out',
    'out = (data - lamb) * (data > lamb) + (data + lamb) * (data < -lamb)',
    'soft_thresholding'
)

def fast_threshold(data, lamb):
    kernel(data, lamb, data)

mix_grad_kernel = ElementwiseKernel(
    'float32 grad_avg, float32 grad_saga, float32 B, float32 n_ex',
    'float32 out',
    'out = (1 - B / n_ex) * grad_avg + (B / n_ex) * grad_saga',
    'grad_update'
)

def avg_grad_update(grad_avg, grad_saga, B, n_ex):
    mix_grad_kernel(grad_avg, grad_saga, B, n_ex, grad_avg)

normalize_kernel = ElementwiseKernel(
    'float32 X_bool, float32 mean, float32 std',
    'float32 out',
    'out = (X_bool - mean) / std',
    'ez_normalize'
)

def normalize(X_bool, mean, std, X):
    normalize_kernel(X_bool, mean, std, X)