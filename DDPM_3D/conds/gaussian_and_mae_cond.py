from .gaussian_mse_cond import *
from .low_res_mae_cond import *

def gaussian_and_mae_cond_fn(x, t, low_res, kernels, sigmas, weights=[0.2, 0.2, 0.6], guided_scale1=0.5, guided_scale2=0.5):
    grad_gaussian = gaussian_mse_cond_fn(x, t, low_res, kernels, sigmas, weights, guided_scale1)
    grad_mae = low_res_mae_cond_fn(x, t, low_res, guided_scale2)

    return grad_gaussian + grad_mae