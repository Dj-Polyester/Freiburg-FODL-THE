import numpy as np

from lib.network_base import Parameter
from lib.regularizers import L2Regularization


def test_forward_pass():
    """Test the forward pass of L2 regularization."""
    np.random.seed(0)
    param_data = np.random.randn(50, 1) * 0.1
    param_data[6] = 0
    params = [Parameter(param_data)]
    l2 = L2Regularization(0.1, params)
    loss = l2.forward()
    err_msg = 'L2Regularization forward pass not implemented correctly'
    np.testing.assert_almost_equal(loss, 0.03171263484262931, decimal=5, err_msg=err_msg)


def test_forward_pass_multiple_parameters():
    """Test the forward pass of L2 regularization, using a model with multiple parameters."""
    np.random.seed(0)
    params = [Parameter(np.random.randn(50, 1)) for _ in range(20)]
    l2 = L2Regularization(0.1, params)
    loss = l2.forward()

    err_msg = 'L2Regularization forward pass not implemented correctly for a model with multiple parameters'
    np.testing.assert_almost_equal(loss, 48.81413129425037, decimal=5, err_msg=err_msg)


def test_forward_pass_multiple_dimensions():
    """Test the forward pass of L2 regularization, with a multidimensional parameter."""
    np.random.seed(0)
    param_data = np.random.randn(5, 5, 5, 5) * 0.1
    params = [Parameter(param_data)]
    l2 = L2Regularization(0.1, params)
    loss = l2.forward()
    err_msg = 'L2Regularization forward pass not implemented correctly'
    np.testing.assert_almost_equal(loss, 0.3129269745539161, decimal=5, err_msg=err_msg)


def test_gradient():
    """Test the gradient of L2 regularization."""
    np.random.seed(0)
    param_data = np.random.randn(50, 1) * 0.1
    param_data[6] = 0
    params = [Parameter(param_data)]
    L2Regularization(0.1, params).check_gradients_wrt_params((), 1e-6)


def test_backward_pass():
    """Test the backward pass of L2 regularization."""
    np.random.seed(0)
    param_data = np.random.randn(5, 1) * 0.1
    params = [Parameter(param_data)]
    for p in params:
        p.grad = np.random.randn(*p.data.shape)
    l2 = L2Regularization(0.1, params)
    l2.backward()

    expected_grads = np.array([[[-0.95963736], [0.95408999], [-0.14156983], [-0.08080992], [0.42927408]]])

    err_msg = 'L2Regularization backward pass not implemented correctly'
    np.testing.assert_allclose([p.grad for p in l2.params], expected_grads, rtol=1e-5, err_msg=err_msg)


def test_backward_pass_multiple_parameters():
    """Test the backward pass of L2 regularization, using a model with multiple parameters."""
    np.random.seed(0)
    params = [Parameter(np.random.randn(2, 1)) for _ in range(2)]
    for p in params:
        p.grad += np.abs(np.random.randn(*p.data.shape))
    l2 = L2Regularization(0.1, params)
    l2.backward()

    expected_grads = np.array([[[2.04396322], [1.0172936]], [[1.04796222], [0.37544653]]])
    err_msg = 'L2Regularization backward pass not implemented correctly for a model with multiple parameters'
    np.testing.assert_allclose([p.grad for p in l2.params], expected_grads, rtol=1e-5, err_msg=err_msg)


if __name__ == '__main__':
    test_forward_pass()
    test_forward_pass_multiple_parameters()
    test_forward_pass_multiple_dimensions()
    test_gradient()
    test_backward_pass()
    test_backward_pass_multiple_parameters()
    print('Test complete.')
