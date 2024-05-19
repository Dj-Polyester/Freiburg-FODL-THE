import numpy as np

from lib.gradient_utilities import zero_grad
from lib.network_base import Parameter


def test_zero_grad():
    """Test zero_grad function."""
    W = Parameter(np.random.rand(20, 10), np.random.rand(20, 10), "W")
    b = Parameter(np.random.rand(20, 1), np.random.rand(20, 1), "b")
    params = [W, b]

    zero_grad(params)

    err_msg = "Zero-grad function is not implemented correctly."
    for param_num in range(2):
        np.testing.assert_array_equal(
            params[param_num].grad.shape, params[param_num].data.shape, err_msg=err_msg
        )
        np.testing.assert_allclose(
            np.sum(np.abs(params[param_num].grad)), 0, err_msg=err_msg
        )


def test_zero_grad_module_behavior():
    """Test zero_grad function behavior simulating an initialized Module"""
    W = Parameter(np.random.rand(20, 10), name="W")
    b = Parameter(np.random.rand(20, 1), name="b")
    params = [W, b]

    zero_grad(params)

    err_msg = (
        "Zero-grad function is not implemented correctly. Remember that models initially do not have any gradient"
        "information saved and thus also no arrays for the gradients initialized."
    )
    for param_num in range(2):
        np.testing.assert_array_equal(
            params[param_num].grad.shape, params[param_num].data.shape, err_msg=err_msg
        )
        np.testing.assert_allclose(
            np.sum(np.abs(params[param_num].grad)), 0, err_msg=err_msg
        )


if __name__ == "__main__":
    test_zero_grad()
    test_zero_grad_module_behavior()
    print("Test complete.")
