import os


import matplotlib
import numpy as np
import torch

from lib.utils import add_noise


matplotlib.use('Agg')


def test_add_noise():
    # Seed the environment
    np.random.seed(0)
    torch.manual_seed(0)
    # sample random time step
    t = np.random.randint(0, 1000)
    # initialize random baralphas of dim 1000
    baralphas = torch.rand(1000)
    # initialize random batch of data points
    batch = torch.rand(32, 2)

    noise, eps = add_noise(batch, t, baralphas)
    # assertTrue(np.isclose(torch.sum(noise).item(), 34.20818, rtol=1.e-2))
    # assertTrue(np.isclose(torch.sum(eps).item(), 13.0110, rtol=1.e-2))
    assert (np.isclose(torch.sum(noise).item(), 34.20818, rtol=1.e-2)
            and np.isclose(torch.sum(eps).item(), 13.0110, rtol=1.e-2)
            ), f"Your current values do not match the hardcoded values,please recheck the implementation"


if __name__ == '__main__':
    test_add_noise()
    print("Test complete.")
