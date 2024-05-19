import numpy as np

from lib.activations import ReLU


def test_relu1d():
    """Test the ReLU function on an example."""
    x = np.linspace(-4, +4, 9)
    relu = ReLU()
    y = relu(x)
    np.testing.assert_allclose(y, [0., 0., 0., 0., 0., 1., 2., 3., 4.], err_msg="ReLU is not implemented correctly.")


def test_relu2d():
    """Test the ReLU function on an example."""
    x = np.linspace(-4, +4, 9).reshape(3, -1)
    relu = ReLU()
    y = relu(x)
    np.testing.assert_allclose(y, [[0., 0., 0.], [0., 0., 1.], [2., 3., 4.]],
                               err_msg="ReLU is not implemented correctly for multidimensional arrays")


if __name__ == '__main__':
    test_relu1d()
    test_relu2d()
    print("Test complete.")
