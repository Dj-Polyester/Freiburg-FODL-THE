from sklearn.datasets import make_moons
from sklearn.datasets import make_swiss_roll, make_circles
import torch
import matplotlib.pyplot as plt
import math


def get_circles(n_samples=10000):
    x, _ = make_circles(noise=0.05, factor=0.5, random_state=1, n_samples=n_samples)
    print(x.shape)
    return x


def get_moons(n_samples=10000):
    x, _ = make_moons(n_samples=n_samples, noise=0.01)
    return x


def get_swiss_roll(n_samples=10000):
    x, _ = make_swiss_roll(n_samples=n_samples, noise=0.5)
    x = x[:, [0, 2]]
    x = (x - x.mean()) / x.std()
    return x


def get_dataset(dataset_name="moons", n_samples=10000):
    if dataset_name == "moons":
        return get_moons(n_samples=n_samples)
    elif dataset_name == "swiss_roll":
        return get_swiss_roll(n_samples=n_samples)
    elif dataset_name == "circles":
        return get_circles(n_samples=n_samples)
    else:
        raise NotImplementedError


def plot_dataset(x, dataset_name="moons"):
    plt.scatter(x[:, 0], x[:, 1])
    fig_name = dataset_name + ".png"
    plt.savefig(fig_name)
    plt.clf()


def cosine_schedule(diffusion_steps=1000):
    steps = diffusion_steps + 1
    s = 0.008
    x = torch.linspace(0, diffusion_steps, steps)
    alphas_cumprod = (
        torch.cos(((x / diffusion_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999)
    alphas = 1 - betas
    baralphas = torch.cumprod(alphas, dim=0)
    return baralphas, alphas, betas


def linear_schedule(diffusion_steps=1000):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, diffusion_steps)
    alphas = 1 - betas
    baralphas = torch.cumprod(alphas, dim=0)
    return baralphas, alphas, betas


def plot_bar_alpha():
    cosine, _, _ = cosine_schedule()
    linear, _, _ = linear_schedule()
    plt.plot(cosine, color="blue", label="cosine")
    plt.plot(linear, color="orange", label="linear")
    plt.legend()
    plt.savefig("linear_and_cosine_schedules.png")
    plt.clf()


def add_noise(batch, t, baralphas):
    eps = torch.randn(size=batch.shape).to(batch.device)
    noised = None
    # START TODO #
    # 1. Add noise to the batch using equation 4 from the DDPM paper https://arxiv.org/pdf/2006.11239.pdf
    # 2. Use the baralphas and the noise sampled in the previous step to compute the noised data
    # eps = torch.randn(size=batch.shape)

    noised = torch.sqrt(baralphas[t]) * batch + torch.sqrt(1 - baralphas[t]) * eps

    if noised is None:
        raise NotImplementedError("Please complete the steps in the TODO block")
    # END TODO #
    return noised, eps


def plot_noised(batch, t, baralphas):
    noiselevel = t
    noised, eps = add_noise(
        batch, torch.full([len(batch), 1], fill_value=noiselevel), baralphas
    )
    plt.scatter(noised[:, 0], noised[:, 1], marker="*", alpha=0.5)
    plt.scatter(batch[:, 0], batch[:, 1], alpha=0.5)
    plt.legend(["Noised data", "Original data"])
    save_name = "true_vs_noisy_{}.png".format(str(t))
    plt.savefig(save_name)
    plt.clf()


def plot_denoised_vs_true(batch, noised, t, baralphas):
    eps = torch.randn(size=batch.shape)
    denoised = (
        1 / torch.sqrt(baralphas[t]) * (noised - torch.sqrt(1 - baralphas[t]) * eps)
    )
    plt.scatter(batch[:, 0], batch[:, 1], alpha=0.5)
    plt.scatter(denoised[:, 0], denoised[:, 1], marker="1", alpha=0.5)
    plt.legend(["Original data", "Recovered original data"])
    plt.savefig("original_and_denoised.png")
    plt.clf()
