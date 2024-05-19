import numpy as np

from lib.activations import ReLU
from lib.network import Sequential
from lib.utilities import Dummy


def test_gradient_relu():
    """Test ReLU gradient."""
    input_vector = np.random.uniform(-1., 1., size=(2, 10))
    ReLU().check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(4, 20))
    ReLU().check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(6, 40))
    ReLU().check_gradients((input_vector,))


def test_gradient_relu_bp():
    """Test ReLU gradient.
    This test will fail if you do not actually backpropagate the gradient of the later module, assuming you passed
    test_backward_pass"""
    input_vector = np.random.uniform(-1., 1., size=(2, 10))
    model = Sequential(Dummy(), ReLU(), Dummy(), ReLU())
    model.check_gradients((input_vector, ))

    input_vector = np.random.uniform(-1., 1., size=(4, 20))
    model = Sequential(Dummy(), ReLU(), Dummy(), ReLU())
    model.check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(6, 40))
    model = Sequential(Dummy(), ReLU(), Dummy(), ReLU())
    model.check_gradients((input_vector,))


if __name__ == '__main__':
    test_gradient_relu()
    test_gradient_relu_bp()
    print("Test complete.")
