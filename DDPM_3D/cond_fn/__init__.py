from .mse_cond import MSE_cond_fn

def cond_fn_factory(name):
    cond_fn_map = {
        "mse": MSE_cond_fn,
    }
    try:
        return cond_fn_map[name]
    except KeyError:
        raise ValueError("PLEASE CHECK COND FN CONFIG")
    