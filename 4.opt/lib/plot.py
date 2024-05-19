"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np

from lib.lr_schedulers import PiecewiseConstantLR, CosineAnnealingLR
from lib.optimizers import Adam, SGD
from lib.network_base import Parameter
from lib.utilities import load_result


def plot_learning_curves() -> None:
    """Plot the performance of SGD, SGD with momentum, and Adam optimizers.

    Note:
        This function requires the saved results of compare_optimizers() above, so make
        sure you run compare_optimizers() first.
    """
    optim_results = load_result("optimizers_comparison")
    # START TODO ################
    # train result are tuple(train_costs, train_accuracies, eval_costs,
    # eval_accuracies). You can access the iterable via
    # optim_results.items()

    fig, axs = plt.subplots(1, 2)
    for optim_name, optim_result in optim_results.items():
        train_costs, train_accuracies, _, _ = optim_result

        axs[0].plot(train_costs, label=optim_name)
        axs[0].set_xlabel("Number of epochs")
        axs[0].set_ylabel("Training loss")
        axs[0].legend()

        axs[1].plot(train_accuracies, label=optim_name)
        axs[1].set_xlabel("Number of epochs")
        axs[1].set_ylabel("Training accuracy")
        axs[1].legend()

    plt.show()
    # END TODO ###################


def plot_lr_schedules() -> None:
    """Plot the learning rate schedules of piecewise and cosine schedulers."""
    num_epochs = 80
    base_lr = 0.1

    piecewise_scheduler = PiecewiseConstantLR(
        Adam([], lr=base_lr), [10, 20, 40, 50], [0.1, 0.05, 0.01, 0.001]
    )
    cosine_scheduler = CosineAnnealingLR(Adam([], lr=base_lr), num_epochs)

    # START TODO ################
    # plot piecewise lr and cosine lr
    piecewise_scheduler_lr = np.zeros(num_epochs)
    cosine_scheduler_lr = np.zeros(num_epochs)
    for i in range(num_epochs):
        piecewise_scheduler.step()
        piecewise_scheduler_lr[i] = piecewise_scheduler.optimizer.lr
        cosine_scheduler.step()
        cosine_scheduler_lr[i] = cosine_scheduler.optimizer.lr

    plt.plot(piecewise_scheduler_lr, label="Piecewise constant scheduler")
    plt.plot(cosine_scheduler_lr, label="Cosine delay scheduler")

    plt.xlabel("Number of epochs")
    plt.ylabel("Training loss")
    plt.legend()
    plt.show()
    # END TODO ################
