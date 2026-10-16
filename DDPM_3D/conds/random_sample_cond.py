from random import randint

def random_sample(lower_bound, higher_bound, num_samples):
    assert lower_bound >= 0
    assert higher_bound <= 999
    assert lower_bound < higher_bound

    sample_list = []
    for _ in range(num_samples):
        random_num = randint(lower_bound, higher_bound)
        sample_list.append(random_num)
    sample_list.sort()
    
    return sample_list