from .low_res_mse_cond import low_res_mse_cond_fn
from .low_res_mae_cond import low_res_mae_cond_fn
from .gaussian_mse_cond import gaussian_mse_cond_fn
from .gaussian_and_mae_cond import gaussian_and_mae_cond_fn


from .single_range_cond import single_range
from .full_range_cond import full_range
from .random_sample_cond import random_sample
from .equal_intervals_cond import equal_intervals
from .range_intervals_cond import range_interval

def cond_fn_factary(name):
    cond_fn_map = {
        "low_res_mse": low_res_mse_cond_fn,
        "low_res_mae": low_res_mae_cond_fn,
        "gaussian_mse": gaussian_mse_cond_fn,
        "gaussian_and_mae": gaussian_and_mae_cond_fn,
    }

    try:
        return cond_fn_map[name]
    except KeyError:
        raise ValueError(f"The cond_fn: {name} is not exist!")
    
def cond_indeces_factary(name, config):
    cond_indeces_map = {
        "single_range": single_range,
        "random_sample": random_sample,
        "equal_intervals": equal_intervals,
        "default": full_range,
        "range_intervals": range_interval,
    }

    try:
        return cond_indeces_map[name](**config)
    except KeyError:
        return []