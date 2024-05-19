import numpy as np
from lib.mcmc import *
import matplotlib.pyplot as plt


def plot_mcmc1d(
    target_pdf,
    proposal_pdf,
    proposal_sampler,
    start_sample,
    num_samples,
    burn_in,
    sampling_freq,
):
    x = np.linspace(-20, 20, num=200)
    y = []
    for _x in x:
        _y = target_pdf(_x) / (np.sqrt(2 * np.pi) * 2)
        y.append(_y)
    samples = metropolis_hastings(
        target_pdf,
        proposal_pdf,
        proposal_sampler,
        start_sample,
        num_samples,
        burn_in,
        sampling_freq,
    )
    samples = np.array(samples)
    bins = np.linspace(-20, 20, num=41)
    w = bins[1] - bins[0]
    height_bar = []
    x_bar = []
    for i in range(len(bins) - 1):
        height_bar.append(np.sum((samples >= bins[i]) & (samples < bins[i + 1])))
        x_bar.append((bins[i] + bins[i + 1]) / 2.0)
    height_bar = np.array(height_bar, dtype=np.float64)
    height_bar /= np.sum(height_bar)
    plt.bar(x_bar, height_bar, width=w, edgecolor="w", label="mcmc sampling")
    plt.plot(x, y, "r", label="GMM pdf")
    plt.legend()
    plt.show()


def plot_mcmc2d(
    target_pdf,
    proposal_pdf,
    proposal_sampler,
    start_sample,
    num_samples,
    burn_in,
    sampling_freq,
):
    X = np.linspace(-6, 6, num=200)
    Y = np.linspace(-6, 6, num=200)
    points = np.meshgrid(X, Y)
    N = len(X)
    Z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            x = points[0][i, j]
            y = points[1][i, j]
            p = [x, y]
            Z[i, j] = target_pdf(p)
    plt.contourf(X, Y, Z, levels=35)
    samples = metropolis_hastings(
        target_pdf,
        proposal_pdf,
        proposal_sampler,
        start_sample,
        num_samples,
        burn_in,
        sampling_freq,
    )
    samples = np.array(samples)
    plt.scatter(samples[:, 0], samples[:, 1], c="red", s=0.01, label="samples")
    plt.legend()
    plt.show()


def plot_mcmc_2gmm1d(
    start_sample: float, num_samples: int, burn_in: int, sampling_freq: int
):
    plot_mcmc1d(
        target_2gmm1d_pdf_upto_norm,
        proposal_gaussian1d_pdf,
        proposal_gaussian1d_sampler,
        start_sample,
        num_samples,
        burn_in,
        sampling_freq,
    )


def plot_mcmc_3gmm1d(
    start_sample: float, num_samples: int, burn_in: int, sampling_freq: int
):
    plot_mcmc1d(
        target_3gmm1d_pdf_upto_norm,
        proposal_gaussian1d_pdf,
        proposal_gaussian1d_sampler,
        start_sample,
        num_samples,
        burn_in,
        sampling_freq,
    )


def plot_mcmc_2gmm2d(
    start_sample: float, num_samples: int, burn_in: int, sampling_freq: int
):
    plot_mcmc2d(
        target_2gmm2d_pdf_upto_norm,
        proposal_gaussian2d_pdf,
        proposal_gaussian2d_sampler,
        start_sample,
        num_samples,
        burn_in,
        sampling_freq,
    )


if __name__ == "__main__":
    burn_in = 10000
    np.random.seed(123456)
    plot_mcmc_2gmm1d(
        start_sample=0.0, num_samples=1000, burn_in=burn_in, sampling_freq=10
    )
    np.random.seed(654321)
    plot_mcmc_3gmm1d(
        start_sample=0.0, num_samples=1000, burn_in=burn_in, sampling_freq=10
    )
    np.random.seed(321654)
    plot_mcmc_2gmm2d(
        start_sample=[2.999, 2.999], num_samples=10000, burn_in=burn_in, sampling_freq=2
    )
