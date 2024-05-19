from lib.utils import get_dataset, add_noise, plot_dataset, cosine_schedule, linear_schedule, plot_bar_alpha
from lib.model_diffusion import SimpleDiffusion
from lib.train_diffusion import train
import argparse
import torch

import matplotlib.pyplot as plt

import matplotlib.animation as animation


def draw_frame(i):
    plt.clf()
    Xvis = Xgen_hist[i].cpu()
    fig = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker="1", animated=True)
    return fig,


def sample_ddpm(args, model, nsamples, nfeatures, baralphas, alphas, betas):
    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    baralphas = baralphas.to(args.device)
    alphas = alphas.to(args.device)
    betas = betas.to(args.device)
    with torch.no_grad():
        x = torch.randn(size=(nsamples, nfeatures)).to(args.device)
        xt = [x]
        for t in range(args.diffusion_steps-1, 0, -1):
            predicted_noise = model(x, torch.full(
                [nsamples, 1], t).to(args.device))
            # See DDPM paper between equations 11 and 12
            x = 1 / (alphas[t] ** 0.5) * (x - (1 - alphas[t]) /
                                          ((1-baralphas[t]) ** 0.5) * predicted_noise)
            if t > 1:
                # See DDPM paper section 3.2.
                # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                variance = betas[t]
                std = variance ** (0.5)
                x += std * \
                    torch.randn(size=(nsamples, nfeatures)).to(args.device)
            xt += [x]
        return x, xt


def main():
    parser = argparse.ArgumentParser("simple 2d diffusion")
    parser.add_argument('--dataset', type=str,
                        default='moons', help='choose dataset from ["moons", "swiss_roll", "circles"]')
    parser.add_argument('--batch_size', type=int,
                        default=2048, help='batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3, help='init learning rate')
    parser.add_argument('--device', type=str,
                        default="cuda", help='device to use')
    parser.add_argument('--schedule', type=str, default="cosine",
                        help='schedule to use for beta cosine or linear')
    parser.add_argument('--epochs', type=int, default=250,
                        help='num of training epochs')
    parser.add_argument('--n_samples', type=int, default=100000,
                        help='number of samples in dataset')
    parser.add_argument('--n_blocks', type=int, default=8,
                        help='number of linear layers in diffusion model')
    parser.add_argument('--diffusion_steps', type=int, default=100,
                        help='number of diffusion steps')
    parser.add_argument('--verbose', type=bool, default=True)
    args = parser.parse_args()
    args.verbose = True
    dataset = get_dataset(args.dataset, args.n_samples)
    dataset = torch.tensor(dataset, dtype=torch.float32).to(args.device)
    if args.schedule == "cosine":
        baralphas, alphas, betas = cosine_schedule(args.diffusion_steps)
    else:
        baralphas, alphas, betas = linear_schedule(args.diffusion_steps)
    model = SimpleDiffusion(nfeatures=2, nblocks=args.n_blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train(dataset, args, model, optimizer, baralphas)
    global Xgen_hist
    Xgen, Xgen_hist = sample_ddpm(
        args, model, 100000, 2, baralphas, alphas, betas)
    Xgen = Xgen.cpu()

    plt.scatter(dataset[:, 0].cpu(), dataset[:, 1].cpu(), alpha=0.5)
    plt.scatter(Xgen[:, 0], Xgen[:, 1], marker="1", alpha=0.25)
    plt.legend(["Original data", "Generated data"])
    plt.savefig("generated_samples.png")
    plt.clf()
    fig = plt.figure()
    anim = animation.FuncAnimation(
        fig, draw_frame, frames=args.diffusion_steps, interval=20, blit=True)
    anim_name = "{}_generation.mp4".format(args.dataset)
    print("Saving animation to {}".format(anim_name))
    anim.save(anim_name, fps=10)


if __name__ == "__main__":
    main()
