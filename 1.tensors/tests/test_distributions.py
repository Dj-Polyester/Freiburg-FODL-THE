from lib.distributions import std_normal, normal
import numpy as np


def test_standard_normal_dist():

    # The approximation method is inaccurate so we need lots of samples and a high error tolerance
    n_samples = 10000
    epsilon = 3e-3

    # Test creation of standard normal distribution by sampling a uniform distribution.
    target_mean, target_stddev = 0., 1.
    np.random.seed(1024)
    samples = std_normal(n_samples)
    mean = samples.mean()
    var = samples.var()
    assert (mean - target_mean) ** 2 < epsilon and (var - target_stddev ** 2) ** 2 < epsilon * target_stddev, (
        f"Mean and variance should be {target_mean} and {target_stddev ** 2} but are {mean} and {var}")


def test_normal_dist():

    # The approximation method is inaccurate so we need lots of samples and a high error tolerance
    n_samples = 10000
    epsilon = 1e-2

    # Test normal distribution
    np.random.seed(1024)
    target_mean, target_stddev = 1., 3.
    samples = normal(target_mean, target_stddev, n_samples)
    mean = samples.mean()
    var = samples.var()
    assert (mean - target_mean) ** 2 < epsilon and (var - target_stddev ** 2) ** 2 < epsilon * target_stddev**2, (
        f"Mean and variance should be {target_mean} and {target_stddev ** 2} but are {mean} and {var}")


if __name__ == "__main__":
    test_standard_normal_dist()
    test_normal_dist()
    print('Test complete.')
