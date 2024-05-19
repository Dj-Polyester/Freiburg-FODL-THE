# Some initial setup: Importing required libraries
import torch
import numpy as np
from lib.utilities import load_result, rescaled_sinc, create_single_model
from lib.model import DNGO
from lib.plot import plot_predictions, plot_uncertainty


def eval_dngo(dngo_model):

    x_train, y_train, x_test, y_test = rescaled_sinc()
    xlim = (-0, 1)
    grid = np.linspace(*xlim, 200)
    fvals = np.sinc(grid * 10 - 5)

    grid_preds = dngo_model(torch.from_numpy(
        grid[:, None]).float()).detach().numpy()
    pred_y = dngo_model(x_train.float()).detach().numpy()
    # 50 features in last hidden layer + 1 for bias
    Sigma_pre = np.identity(50 + 1)
    mu_pre = np.zeros((50 + 1, 1))
    noise = 1

    dngo_model.fit_blr_model(
        mu_pre,
        Sigma_pre,
        x_train,
        y_train,
        noise)
    plot_predictions(dngo_model, grid, fvals, grid_preds, pred_y, x_train, y_train)
    plot_uncertainty(dngo_model, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    np.random.seed(42)

    module_list = create_single_model()
    dngo_model = DNGO(*module_list)
    load_result(dngo_model, name="DNGO")
    eval_dngo(dngo_model)
