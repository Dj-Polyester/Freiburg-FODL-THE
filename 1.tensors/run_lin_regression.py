import numpy as np
from lib.blr import BLR
from lib.plot import plot_bayes_linear_regression, plot_contour
from lib.utilities import data_1d, data_2d


def run_blr(lbr_1d: BLR, lbr_2d: BLR, noise) -> None:
    """
    Args:
        lbr_1d: Bayesian Linear Regression object for 1-d data
        lbr_2d: Bayesian Linear Regression object for 2-d data

    Returns:
        None

    Note:
        The function linreg_bayes in BLR object is used to obtain analytical
        values for posterior mean and covariance
    """

    X_train, y_train = data_1d()
    plot_bayes_linear_regression(lbr_1d, X_train, y_train)

    X_train, y_train = data_2d()
    plot_contour(lbr_2d, X_train, y_train, noise)


if __name__ == "__main__":
    np.random.seed(0)  # Dont change this

    # define 1-d bayesian linear regression
    Sigma_pre = np.array([[1.0]])
    mu_pre = np.array([0.0])
    noise = 1.0
    lbr_1d = BLR(mu_pre=mu_pre, sigma_pre=Sigma_pre, noise=noise)
    # define 2-d bayesian linear regression
    Sigma_pre = 10**-10 * np.array([[1.0, 0.0], [0.0, 1.0]])
    mu_pre = np.array([0.0, 0.0])
    lbr_2d = BLR(mu_pre=mu_pre, sigma_pre=Sigma_pre, noise=noise)

    # run 1-d and 2-d bayesian linear regression
    run_blr(lbr_1d, lbr_2d, noise)
