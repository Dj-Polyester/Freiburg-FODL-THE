import numpy as np
import torch

from lib.utilities import create_single_model, rescaled_sinc
from lib.model import DNGO
from tests.results import dngo_fit_blr_mu, dngo_fit_blr_sigma, dng_predict_mean, dng_predict_std


def dngo_test():
    np.random.seed(42)
    module_list = create_single_model()
    dngo_model = DNGO(*module_list)
    x_train, y_train, x_test, y_test = rescaled_sinc()
    # 50 features in last hidden layer + 1 for bias
    Sigma_pre = np.identity(50 + 1)
    mu_pre = np.zeros((50 + 1))
    noise = 1
    dngo_model.fit_blr_model(mu_pre, Sigma_pre, x_train, y_train, noise)

    mu_post = dngo_model.mu_post
    Sigma_post = dngo_model.Sigma_post
    err_msg = "fit_blr_model function not implemented correctly"
    np.testing.assert_allclose(
        mu_post[:],
        dngo_fit_blr_mu,
        atol=1e-5,
        err_msg=err_msg)

    np.testing.assert_allclose(
        Sigma_post.flatten(),
        dngo_fit_blr_sigma,
        atol=1e-5,
        err_msg=err_msg)

    xlim = (-0, 1)
    grid = np.linspace(*xlim, 10)
    mean, std = dngo_model.predict_mean_and_std(
        torch.from_numpy(grid[:, None]).float())

    np.testing.assert_equal(mean.shape[0], 10, err_msg="mean has a wrong shape DNGO")
    np.testing.assert_equal(std.shape[0], 10, err_msg="std. dev. has a wrong shape DNGO")

    if len(mean.shape) == 1:
        mean = mean.reshape(-1, 1)
    if len(std.shape) == 1:
        std = std.reshape(-1, 1)

    err_msg = "predict_mean_and_std mean result is not correct"
    np.testing.assert_allclose(
        np.concatenate([mean[:, 0]]),
        np.concatenate([dng_predict_mean]),
        atol=1e-5,
        err_msg=err_msg)

    err_msg = "predict_mean_and_std std. dev. result is not correct"
    np.testing.assert_allclose(
        np.concatenate([std[:, 0]]),
        np.concatenate([dng_predict_std]),
        atol=1e-5,
        err_msg=err_msg)
