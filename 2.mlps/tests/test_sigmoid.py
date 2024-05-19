import numpy as np

from lib.activations import Sigmoid


def test_sigmoid1d():
    """Test the Sigmoid function on a 1D-Example."""
    x = np.linspace(-4, +4, 10).reshape(1, -1)
    sigmoid = Sigmoid()
    y = sigmoid(x)
    np.testing.assert_allclose(y, [
        [0.01798621, 0.04265125, 0.0977726, 0.20860853, 0.39068246, 0.60931754, 0.79139147, 0.9022274, 0.95734875,
         0.98201379]], err_msg="Sigmoid is not implemented correctly.")


def test_sigmoid2d():
    """Test the Sigmoid function on a 2D-Example."""
    x = np.linspace(-4, +4, 10).reshape(2, -1)
    sigmoid = Sigmoid()
    y = sigmoid(x)
    np.testing.assert_allclose(y, [
        [0.01798621, 0.04265125, 0.0977726, 0.20860853, 0.39068246],
        [0.60931754, 0.79139147, 0.9022274, 0.95734875, 0.98201379]],
                               err_msg="Sigmoid is not implemented correctly for multidimensional input.")


if __name__ == '__main__':
    test_sigmoid1d()
    test_sigmoid2d()
    print("Test complete.")
