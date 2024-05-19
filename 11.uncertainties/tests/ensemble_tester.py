
import numpy as np
import torch

from lib.model import EnsembleFeedForward
from tests.results import ensemble_mean, ensemble_std, ensemble_pred
from lib.utilities import create_ensembled


def ensemble_test():

    n = 5
    ensembled_nets = create_ensembled(num_models=n)
    ensemble = EnsembleFeedForward(ensembled_nets=ensembled_nets)
    xlim = (-0, 1)
    grid = np.linspace(*xlim, 10)

    pred = ensemble.individual_predictions(
        torch.from_numpy(grid[:, None]).float())

    err_msg = "individual_predictions Ensemble not implemented correctly"
    np.testing.assert_allclose(
        pred,
        ensemble_pred,
        atol=1e-3,
        err_msg=err_msg)

    mean, std = ensemble.predict_mean_and_std(
        torch.from_numpy(grid[:, None]).float())

    np.testing.assert_equal(mean.shape[0], 10, err_msg="mean has a wrong shape Ensemble")
    np.testing.assert_array_equal(std.shape[0], 10, err_msg="std. dev. has a wrong shape Ensemble")
    if len(mean.shape) == 1:
        mean = mean.reshape(-1, 1)
    if len(std.shape) == 1:
        std = std.reshape(-1, 1)

    mean = mean.flatten()
    std = std.flatten()
    err_msg = "predict_mean_and_std mean of Ensemble not implemented correctly"
    np.testing.assert_allclose(
        mean,
        ensemble_mean,
        atol=1e-5,
        err_msg=err_msg)

    err_msg = "predict_mean_and_std std. dev. of Ensemble not implemented correctly"
    np.testing.assert_allclose(
        std,
        ensemble_std,
        atol=1e-5,
        err_msg=err_msg)
