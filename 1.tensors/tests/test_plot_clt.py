import numpy as np

from lib.distributions import plot_clt


def test_plot_clt_sample_shape():
    samples, means = plot_clt(n_repetitions=2, sample_sizes=(3, 4), plot=False)
    correct_samples = [
        np.array([[0.30384489, 0.17278791, 0.32590539], [0.61492192, 0.38713495, 0.73062648]]),
        np.array([[0.44274929, 1.33721701, 2.54883597], [-0.37073656, 2.4252914, 0.72053609]]),
        np.array([[1.58045489, 0.44826025, 0.53803229, 0.87769071], [1.42783234, 0.20801901, 0.33991215, 1.10932605]]),
        np.array([[-0.32939545, 0.99854526, -0.31465268, 0.62038826],
                  [2.26521065, 1.12066774, 1.14794178, -1.75372579]])
    ]
    for sample, correct_sample in zip(samples, correct_samples):
        assert sample.shape == correct_sample.shape, "The sample arrays do not have the correct shape"


def test_plot_clt_mean_shape():
    samples, means = plot_clt(n_repetitions=2, sample_sizes=(3, 4), plot=False)
    correct_means = [
        np.array([0.26751273, 0.57756111]), np.array([1.44293409, 0.92503031]),
        np.array([0.86110954, 0.77127239]),
        np.array([0.24372135, 0.69502359])]
    for mean, correct_mean in zip(means, correct_means):
        assert mean.shape == correct_mean.shape, "The mean arrays do not have the correct shape"


def test_plot_clt_sample_values():
    samples, means = plot_clt(n_repetitions=2, sample_sizes=(3, 4), plot=False)
    correct_samples = [
        np.array([[0.30384489, 0.17278791, 0.32590539], [0.61492192, 0.38713495, 0.73062648]]),
        np.array([[0.44274929, 1.33721701, 2.54883597], [-0.37073656, 2.4252914, 0.72053609]]),
        np.array([[1.58045489, 0.44826025, 0.53803229, 0.87769071], [1.42783234, 0.20801901, 0.33991215, 1.10932605]]),
        np.array([[-0.32939545, 0.99854526, -0.31465268, 0.62038826],
                  [2.26521065, 1.12066774, 1.14794178, -1.75372579]])
    ]
    for sample, correct_sample in zip(samples, correct_samples):
        np.testing.assert_allclose(sample, correct_sample, err_msg="Calculation of samples not implemented correctly")


def test_plot_clt_mean_values():
    samples, means = plot_clt(n_repetitions=2, sample_sizes=(3, 4), plot=False)
    correct_means = [
        np.array([0.26751273, 0.57756111]), np.array([1.44293409, 0.92503031]),
        np.array([0.86110954, 0.77127239]),
        np.array([0.24372135, 0.69502359])]
    for mean, correct_mean in zip(means, correct_means):
        np.testing.assert_allclose(mean, correct_mean, err_msg="Calculation of mean not implemented correctly")


if __name__ == "__main__":
    test_plot_clt_sample_shape()
    test_plot_clt_mean_shape()
    test_plot_clt_sample_values()
    test_plot_clt_mean_values()
    print("Test complete.")
