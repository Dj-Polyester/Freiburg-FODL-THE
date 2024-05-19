import numpy as np
from typing import Tuple
import os
import torch.nn as nn
import torch
from pathlib import Path


# data generation
def data_1d() -> Tuple[np.array, np.array]:
    # generates y data according to the example in Figure 5.1.

    # the data generator; Input: np.array of shape (1)
    def generator(x):
        # data checking
        if isinstance(x, list):
            x = np.array(x)
        if len(x.shape) != 1 or len(x) != 1:
            raise ValueError('x should be a vector of size 1')
        return 1.5 * x[0] + np.random.normal(0.0, 0.1)

    # the data
    X_train = np.array([[-0.4], [-0.35], [-0.3], [-0.3],
                        [-0.2], [0.1], [0.2], [0.3], [0.4], [0.6]])
    y_train = np.array([generator(X_train[i]) for i in range(len(X_train))])
    return X_train, y_train


def data_2d(noise_rate: float = 0.1) -> Tuple[np.array, np.array]:
    # generates random data according to the function y = 1.5 * x1 + 0.5 * x2
    # + epsilon

    def generator(x):
        # data checking
        if isinstance(x, list):
            x = np.array(x)
        if len(x.shape) != 1 or len(x) != 2:
            raise ValueError('x should be a vector of size 2')
        return 1.5 * x[0] + 0.5 * x[1] + np.random.normal(0.0, noise_rate)

    data_n = 40
    X_train = np.reshape(np.random.random(data_n * 2), (data_n, 2))
    y_train = np.array([generator(X_train[i]) for i in range(len(X_train))])
    return X_train, y_train


def rescaled_sinc() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    np.random.seed(42)
    torch.manual_seed(0)
    n_train = 30
    x_train = np.random.uniform(0, 1, n_train)
    y_train = np.sinc(x_train * 10 - 5)
    x_train = torch.FloatTensor(x_train[:, None])
    y_train = torch.FloatTensor(y_train[:, ])

    x_test = np.random.uniform(0, 1, 20)
    y_test = np.sinc(x_test * 10 - 5)
    x_test = torch.from_numpy(x_test[:, None]).float()
    y_test = torch.from_numpy(y_test[:, ]).float()
    return x_train, y_train, x_test, y_test


def save_result(model: torch.nn, name: str = "") -> None:
    """Save object to disk as pickle file.

    Args:
        filename: Name of file in ./results directory to write object to.
        obj: The object to write to file.

    """
    # make sure save directory exists
    save_path = Path("results/")
    os.makedirs(save_path, exist_ok=True)

    # save the python objects as bytes
    torch.save(model.state_dict(), f"results/model_{name}")


def load_result(model: torch.nn, name: str = "") -> torch.nn:
    """Load object from pickled file.

    Args:
        filename: Name of file in ./results directory to load.

    """
    model_path = Path(f"results/model_{name}")
    model.load_state_dict(torch.load(model_path))


# The initialization of the weights turns out to be crucial for good training with the given hyperparameters,
# so we do that here.

def init_weights(module):
    # torch.manual_seed(0)
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=2)
        nn.init.constant_(module.bias, val=0.0)


def create_ensembled(num_models):

    ensembled_nets = []
    for i in np.arange(num_models):
        temp_model = nn.Sequential(*create_single_model()).apply(init_weights)
        ensembled_nets.append(temp_model)
    return ensembled_nets


def create_single_model():

    module_list = (nn.Linear(in_features=1, out_features=50, bias=True),
                   nn.Tanh(),
                   nn.Linear(in_features=50, out_features=50, bias=True),
                   nn.Tanh(),
                   nn.Linear(in_features=50, out_features=1, bias=True))

    return module_list
