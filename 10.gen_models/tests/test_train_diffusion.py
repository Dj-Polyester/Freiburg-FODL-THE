import os


import matplotlib
import numpy as np
import torch

from lib.model_diffusion import SimpleDiffusion
import argparse
from lib.train_diffusion import train


def get_test_args():
    parser = argparse.ArgumentParser(
        description="tests for training diffusion models", allow_abbrev=False)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--device', type=str,
                        default="cpu", help='device to use')
    parser.add_argument('--schedule', type=str, default="cosine",
                        help='schedule to use for beta cosine or linear')
    parser.add_argument('--epochs', type=int, default=50,
                        help='num of training epochs')
    parser.add_argument('--n_samples', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=2,
                        help='number of linear layers in diffusion model')
    parser.add_argument('--diffusion_steps', type=int, default=10)
    parser.add_argument('--dataset', type=str,
                        default='random', help='choose random dataset')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('-p', '--pytest_args', nargs='+',
                        default=[], help='additional arguments for pytest')
    args, _ = parser.parse_known_args()

    return args


def test_train():
    # Seed the environment
    args = get_test_args()
    np.random.seed(0)
    torch.manual_seed(0)
    # sample random dataset

    dataset = torch.randn(args.n_samples, 2)
    # initialize random baralphas of dim 1000
    baralphas = torch.rand(args.diffusion_steps)
    # initialize dummy model
    model = SimpleDiffusion(nfeatures=2, nblocks=args.n_blocks)
    # initialize dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = train(dataset, args, model, optimizer, baralphas)
    assert np.isclose(loss.item(
    ), 0.0199, rtol=1.e-2), f"Current loss does not match the hardcoded loss value, please recheck the implementation"


if __name__ == '__main__':
    test_train()
    print("Test complete.")
