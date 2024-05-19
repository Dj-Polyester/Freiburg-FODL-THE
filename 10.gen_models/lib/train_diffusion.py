import argparse
import torch
from lib.utils import (
    get_dataset,
    add_noise,
    plot_dataset,
    cosine_schedule,
    linear_schedule,
    plot_bar_alpha,
)
from lib.model_diffusion import SimpleDiffusion
import matplotlib.pyplot as plt

import matplotlib.animation as animation


def train(dataset, args, model, optimizer, baralphas):
    batch_size = args.batch_size
    plot_bar_alpha()
    plot_dataset(dataset.cpu(), dataset_name=args.dataset)
    model = model.to(args.device)
    loss_fn = torch.nn.MSELoss()
    baralphas = baralphas.to(args.device)
    loss = None
    for epoch in range(args.epochs):
        epoch_loss = steps = 0
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            timesteps = torch.randint(0, args.diffusion_steps, size=[len(batch), 1]).to(
                args.device
            )  # sample random timesteps for every sample

            # START TODO #
            # 1. Compute the noisy samples at different timesteps using add_noise function
            # (Note you need to complete the add_noise function in utils.py)
            # 2. Use the defined diffusion model to predict the noise added at timestep t
            # 3. Compute the MSE between the predicted noise and the added noise (gausssian noise)

            noised, eps = add_noise(batch, timesteps, baralphas)
            pred_eps = model(noised, timesteps)
            loss = loss_fn(pred_eps, eps)

            if loss is None:
                raise NotImplementedError("Please complete the steps in the TODO block")
            # END TODO #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            steps += 1
        if args.verbose:
            print(f"Epoch {epoch} loss = {epoch_loss / steps}")

    return loss
