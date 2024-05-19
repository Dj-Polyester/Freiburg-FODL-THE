"""Plotting functions."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats
from typing import Union
from lib.blr import BLR
from lib.model import DNGO, EnsembleFeedForward


def plot_bayes_linear_regression(
    lbr_1d: BLR, X_train: np.array, y_train: np.array, num_samples: int = 100
) -> None:
    """Plot sample function from prior and posterior distribution.
    Additionally, plot the mean for prior and posterior distribution
    from which the samples were drawn from.

    Args:
        lbr_1d: Bayesian Linear Regression object
        X_train : Numpy array of shape (n, D) n - number of samples and
                  D -  dimension of sample
        Y_train : Numpy array of shape (n) n - number of samples and
        num_samples: number of samples to draw from prior and posterior
        noise: noise

    Returns:
        None

    Note:
        Prior mu and prior sigma are already defined in run_lin_regression.py
        Please use different colors for prior and posterior samples

    """

    # get quantities
    mu_pre = lbr_1d.Mu_pre
    Sigma_pre = lbr_1d.Sigma_pre

    # call linreg_bayes to get posterior mean and variance
    mu_post, Sigma_post = lbr_1d.linreg_bayes(X_train, y_train)

    # create multivariate guassian distribution once for prior and posterior
    distr_prior = scipy.stats.multivariate_normal(mu_pre, Sigma_pre)
    distr_post = scipy.stats.multivariate_normal(mu_post, Sigma_post)

    # can be useful for plotting the different sampled functions (alpha
    # parameter)
    pdf_max_val = distr_post.pdf(mu_post)

    # sample num_samples times from the prior and posterior distribution
    samples_prior = distr_prior.rvs(size=num_samples)
    samples_pdf_prior = distr_prior.pdf(samples_prior)

    samples_post = distr_post.rvs(size=num_samples)
    samples_pdf_post = distr_post.pdf(samples_post)

    print("Avg weight value (sampled from prior)", samples_prior.mean())
    x = np.linspace(-1.0, 1.0, 100)

    plt.rcParams["figure.figsize"] = (12.0, 8.0)
    plt.scatter(X_train[:, 0], y_train, c="r", s=100, label="Data points")
    # START TODO ########################
    # Use the samples from the prior distribution, i.e., the weights from the prior and plot the y values for the
    # x values just generated. This will results in multiple linear models
    # being plotted from the prior distribution.
    # Prior mu and prior sigma are already defined in run_lin_regression.py
    # Please use different colors for prior and posterior samples
    raise NotImplementedError
    # END TODO ########################

    print("Avg weight value (sampled from posterior)", samples_post.mean())

    # START TODO ########################
    # Now use the samples from the posterior distribution and plot the y values for the
    # x values. This will results in multiple linear models being plotted from
    # the posterior distribution.
    raise NotImplementedError
    # END TODO ########################

    plt.legend()

    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.show()


def plot_contour(
    lbr_2d: BLR, X_train: np.array, y_train: np.array, num_samples: int = 1000
) -> None:
    """Plot contour plot of the multivariate gaussian distribution for the prior and the posterior
    distribution determined by mean and variance

    Args:
        lbr_2d: Bayesian Linear Regression object for 2-d data
        num_samples: number of samples to draw from prior and posterior
        X_train : Numpy array of shape (n, D) n - number of samples and D -  dimension of sample
        y_train : Numpy array of shape (n) n - number of samples

    Returns:
        None

    Note:
        mu_pre and sigma_pre are already included in lbr_2d
    """
    distr_prior = None
    distr_post = None

    # START TODO #######################
    # Call linreg_bayes function to compute posterior mean and variance based on X_train any y_train
    # Use scipy.stats.multivariate_normal(.. , ..) to define "distr_prior" and
    # "distr_post"
    raise NotImplementedError
    # END TODO ########################

    samples = distr_post.rvs(size=num_samples)
    print(
        "Avg weight values (sampled from posterior)",
        samples[:, 0].mean(),
        samples[:, 1].mean(),
    )

    delta = 0.01
    x, y = np.mgrid[-2.1:2.1:delta, -2.0:2.0:delta]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    plt.figure(figsize=(12, 15))

    plt.subplot(211)
    plt.contourf(x, y, distr_prior.pdf(pos), cmap="Purples")
    plt.title("Prior Contour Lines")
    plt.xlabel("Weight dim 0 values")
    plt.ylabel("Weight dim 1 values")
    plt.colorbar()

    plt.subplot(212)
    plt.contourf(x, y, distr_post.pdf(pos), cmap="Purples")
    plt.title("Posterior Contour Lines")
    plt.xlabel("Weight dim 0 values")
    plt.ylabel("Weight dim 1 values")
    plt.colorbar()

    plt.subplots_adjust(hspace=0.325)
    plt.show()


def plot_predictions(
    model: DNGO,
    grid: np.array,
    fvals: np.array,
    grid_preds: np.array,
    y_train_pred: np.array,
    X_train: np.array,
    y_train: np.array,
) -> None:
    """Plot predictions of DNGO model

    Args:
        model: DNGO model
        grid: Grid of values that will be plotted
        fvals: Actual function values
        grid_preds: Model predictions for points in the grid
        y_train_pred: Predictions for the training data
        X_train: Numpy array of shape (n, D) n - number of samples and
                  D -  dimension of sample
        y_train: Numpy array of shape (n, D) n - number of samples and
                  D -  dimension of sample

    Returns:
        None

    Note:
        This plot shows a comparison between ordinary Deep Learning and DNGO.
        Under the label Predicted function values and Predicted train values lie the predictions of a conventional
        Neural Network.
        Under the Mean posterior prediction label lie the predictions of DNGO using the mean of the posterior of
        the weights.

    """

    xlim = (-0, 1)
    ylim = (-2, 2)
    plt.rc("font", size=15.0, family="serif")
    plt.rcParams["figure.figsize"] = (12.0, 8.0)
    # Get mean prediction from DNGO
    pred_y_mean, pred_y_std = model.predict_mean_and_std(
        torch.from_numpy(grid[:, None]).float()
    )
    # Sample posterior predictions from DNGO
    num_samples = 20
    distr_post = scipy.stats.multivariate_normal(
        model.mu_post.reshape(-1), model.Sigma_post
    )
    sampled_weights = distr_post.rvs(size=num_samples)
    grid_last_hidden_layer = (
        model.last_hidden_layer(torch.from_numpy(grid[:, None]).float())
        .detach()
        .numpy()
    )
    bias = np.ones((grid_last_hidden_layer.shape[0], 1))
    grid_last_hidden_layer = np.hstack([grid_last_hidden_layer, bias])
    grid_last_hidden_layer = grid_last_hidden_layer.T
    for i in range(num_samples):
        grid_pred_mean = np.dot(grid_last_hidden_layer.T, sampled_weights[i])
        plt.plot(grid, grid_pred_mean, c="r", alpha=0.05)
    plt.plot(
        [],
        [],
        c="r",
        label="Predicted function values for samples from posterior (DNGO)",
        alpha=0.2,
    )
    pred_y_mean = np.squeeze(pred_y_mean[:, 0])
    plt.plot(
        grid, pred_y_mean, c="r", label="Mean posterior prediction (DNGO)", alpha=0.5
    )

    plt.rc("font", size=15.0, family="serif")
    plt.rcParams["figure.figsize"] = (12.0, 8.0)

    plt.plot(grid, fvals, "k--", label="True function values")

    plt.plot(
        grid,
        grid_preds[:, 0],
        "b",
        alpha=0.2,
        label="Predicted function values (ordinary DL, not Bayesian)",
    )

    plt.plot(X_train.numpy(), y_train.numpy(), "ko", label="True train values")
    plt.plot(
        X_train.numpy(),
        y_train_pred,
        "bo",
        label="Predicted train values (ordinary DL, not Bayesian)",
    )
    plt.grid()
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(loc="lower right")
    plt.xlabel("x value")
    plt.ylabel("y value")

    plt.show()


def plot_uncertainty(
    model: Union[DNGO, EnsembleFeedForward],
    X_train: np.array,
    y_train: np.array,
    x_test: np.array,
    y_test: np.array,
) -> None:
    """Plot uncertainty of Ensemble or DNGO model

    Args:
        model: DNGO model
        X_train: Numpy array of shape (n1, D) n1 - number of samples and
                  D -  dimension of sample
        y_train: Numpy array of shape (n, D) n1 - number of samples and
                  D -  dimension of sample
        x_test: Numpy array of shape (n2, D) n2 - number of samples and
                  D -  dimension of sample
        y_test: Numpy array of shape (n2, D) n2 - number of samples and
                  D -  dimension of sample

    Returns:
        None

    Note:
        In addition to the grid values, the [train & test] true and predicted values are plotted as well
    """

    xlim = (-0, 1)
    ylim = (-2, 2)

    grid = np.linspace(*xlim, 200, dtype=np.float32)
    fvals = np.sinc(grid * 10 - 5)

    plt.rc("font", size=15.0, family="serif")
    plt.rcParams["figure.figsize"] = (12.0, 8.0)

    grid_mean, grid_std = model.predict_mean_and_std(torch.from_numpy(grid[:, None]))
    grid_mean = np.squeeze(grid_mean[:, 0])
    grid_std = np.squeeze(grid_std[:, 0])
    plt.plot(grid, fvals, "k--", label="True values")
    plt.plot(grid, grid_mean, "r--", label="Predicted values")
    plt.fill_between(
        grid,
        grid_mean + grid_std,
        grid_mean - grid_std,
        color="orange",
        alpha=0.8,
        label="Confidence band 1-std.dev.",
    )
    plt.fill_between(
        grid,
        grid_mean + 2 * grid_std,
        grid_mean - 2 * grid_std,
        color="orange",
        alpha=0.6,
        label="Confidence band 2-std.dev.",
    )
    plt.fill_between(
        grid,
        grid_mean + 3 * grid_std,
        grid_mean - 3 * grid_std,
        color="orange",
        alpha=0.4,
        label="Confidence band 3-std.dev.",
    )

    pred_y, pred_y_std = model.predict_mean_and_std(X_train)
    pred_y = np.squeeze(pred_y[:, 0])

    pred_y_test, pred_y_test_std = model.predict_mean_and_std(x_test)
    pred_y_test = np.squeeze(pred_y_test[:, 0])

    plt.plot(X_train.numpy(), y_train.numpy(), "ko", label="True train values")
    plt.plot(X_train.numpy(), pred_y, "ro", label="Predicted train values")

    plt.plot(x_test.numpy(), y_test.numpy(), "kx", label="True test values")
    plt.plot(x_test.numpy(), pred_y_test, "rx", label="Predicted test values")

    plt.grid()
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.legend(loc="lower right", fontsize="x-small")
    plt.xlabel("x value")
    plt.ylabel("y value")

    plt.show()


def plot_multiple_predictions(
    model: EnsembleFeedForward, X_train: np.array, y_train: np.array
) -> None:
    """Plot multiple predictions of Ensemble

    Args:

        model: EnsembleFeedForward model
        X_train: Numpy array of shape (n1, D) n1 - number of samples and
                  D -  dimension of sample
        y_train: Numpy array of shape (n, D) n1 - number of samples and
                  D -  dimension of sample

    Returns:
        None
    """

    xlim = (-0, 1)
    ylim = (-2, 2)
    grid = np.linspace(*xlim, 200, dtype=np.float32)
    fvals = np.sinc(grid * 10 - 5)

    plt.rc("font", size=15.0)
    plt.rcParams["figure.figsize"] = (12.0, 8.0)
    plt.plot(grid, fvals, "k--", label="True values")

    # START TODO ########################
    # Hint:
    # Use the individual_predictions() function of the ensemble to get individual predictions
    # (over a grid as in plot_uncertainty) for each of the members of the ensemble.
    # Plot all of these one by one using plot() with some transparency using its alpha parameter
    # so that you get a good visualization of the individual predictions
    # Also plot the mean predictions of the ensemble and remember to label the plotted data.
    # Hello
    # I did not understand what "mean predictions of the ensemble" is. Do not we already return it
    # in predict_mean_and_std as the prediction (i.e. pred_y variable below). Could you elaborate on this
    # in your feedback?
    # Regards
    preds = model.individual_predictions(X_train)
    preds = preds.squeeze()
    alpha_step_size = 1 / model.num_models
    for i in range(model.num_models):
        plt.plot(
            X_train.numpy(),
            preds[:, i],
            alpha=alpha_step_size * (i + 1),
            label=f"Predicted train values for ensemble {i+1}",
        )
    # END TODO ########################
    pred_y, pred_y_std = model.predict_mean_and_std(X_train)
    pred_y = np.squeeze(pred_y[:, 0])

    plt.plot(X_train.numpy(), y_train.numpy(), "ko", label="True train values")
    plt.plot(X_train.numpy(), pred_y, "ro", label="Predicted train values")
    plt.grid()
    plt.xlim(*xlim)
    plt.ylim(*ylim)

    plt.legend(loc="best")
    plt.xlabel("x value")
    plt.ylabel("y value")

    plt.show()
