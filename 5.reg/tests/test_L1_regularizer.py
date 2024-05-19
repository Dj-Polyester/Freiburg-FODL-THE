import numpy as np

from lib.network_base import Parameter
from lib.regularizers import L1Regularization


def test_forward_pass():
    """Test the forward pass of L1 regularization."""
    np.random.seed(0)
    param_data = np.random.randn(50, 1) * 0.1
    param_data[6] = 0
    params = [Parameter(param_data)]
    l1 = L1Regularization(0.1, params)
    loss = l1.forward()

    err_msg = 'L1Regularization forward pass not implemented correctly'
    np.testing.assert_almost_equal(loss, 0.44749749800368427, decimal=5, err_msg=err_msg)


def test_forward_pass_multiple_parameters():
    """Test the forward pass of L1 regularization, using a model with multiple parameters."""
    np.random.seed(0)
    params = [Parameter(np.random.randn(50, 1)) for _ in range(20)]
    l1 = L1Regularization(0.1, params)
    loss = l1.forward()

    err_msg = 'L1Regularization forward pass not implemented correctly for a model with multiple parameters'
    np.testing.assert_almost_equal(loss, 78.62409916434608, decimal=5, err_msg=err_msg)


def test_forward_pass_multiple_dimensions():
    """Test the forward pass of L1 regularization, with a multidimensional parameter."""
    np.random.seed(0)
    param_data = np.random.randn(5, 5, 5, 5) * 0.1
    params = [Parameter(param_data)]
    l1 = L1Regularization(0.1, params)
    loss = l1.forward()
    err_msg = 'L1Regularization forward pass not implemented correctly'
    np.testing.assert_almost_equal(loss, 4.992964358443598, decimal=5, err_msg=err_msg)


def test_gradient():
    """Test the gradient of L1 Regularization."""
    np.random.seed(0)
    param_data = np.random.randn(50, 1) * 0.1
    params = [Parameter(param_data)]
    L1Regularization(0.1, params).check_gradients_wrt_params((), 1e-6)


def test_backward_pass():
    """Test the backward pass of L1 regularization."""
    np.random.seed(0)
    param_data = np.random.randn(5, 1) * 0.1
    params = [Parameter(param_data)]
    for p in params:
        p.grad = np.abs(np.random.randn(*p.data.shape))
    l1 = L1Regularization(0.1, params)
    l1.backward()

    expected_grads = np.array([[[1.07727788], [1.05008842], [0.25135721], [0.20321885], [0.5105985]]])

    err_msg = 'L1Regularization backward pass not implemented correctly'
    np.testing.assert_allclose([p.grad for p in l1.params], expected_grads, rtol=1e-5, err_msg=err_msg)


def test_backward_pass_multiple_parameters():
    """Test the backward pass of L1 regularization, using a model with multiple parameters."""
    np.random.seed(0)
    params = [Parameter(np.random.randn(2, 1)) for _ in range(2)]
    for p in params:
        p.grad += np.abs(np.random.randn(*p.data.shape))
    l1 = L1Regularization(0.1, params)
    l1.backward()

    expected_grads = np.array([[[1.96755799], [1.07727788]], [[1.05008842], [0.25135721]]])
    err_msg = 'L1Regularization backward pass not implemented correctly for a model with multiple parameters'
    np.testing.assert_allclose([p.grad for p in l1.params], expected_grads, rtol=1e-5, err_msg=err_msg)


if __name__ == '__main__':
    test_forward_pass()
    test_forward_pass_multiple_parameters()
    test_forward_pass_multiple_dimensions()
    test_gradient()
    test_backward_pass()
    test_backward_pass_multiple_parameters()
    print('Test complete.')
