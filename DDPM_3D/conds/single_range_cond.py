
def single_range(lower_bound, higher_bound):
    assert lower_bound >= 0
    assert higher_bound <= 999
    assert lower_bound < higher_bound

    return list(range(lower_bound, higher_bound))
