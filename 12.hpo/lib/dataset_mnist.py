from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def load_mnist_minibatched(
    batch_size: int,
    n_train: int = 8192,
    n_valid: int = 1024,
    valid_test_batch_size: int = 1024,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST data using torchvision.

    Returns:
        train data, validation data, test data
    """
    train_dataset = torchvision.datasets.MNIST(
        root="../data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="../data", train=False, transform=transforms.ToTensor()
    )
    train_sampler = SubsetRandomSampler(range(n_train))
    validation_sampler = SubsetRandomSampler(range(n_train, n_train + n_valid))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
    )
    validation_loader = DataLoader(
        dataset=train_dataset,
        batch_size=valid_test_batch_size,
        sampler=validation_sampler,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=valid_test_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader, test_loader
