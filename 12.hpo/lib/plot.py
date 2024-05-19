"""Plotting functions."""
import matplotlib.pyplot as plt
import numpy as np


def plot_lr_vs_filter(results) -> None:
    """Plot the learning rate versus the number of filter.

    Args:
        results: Dictionary containing the final losses, hyperparameter configurations, and the validation errors
        across epochs

    Returns:
        None
    """

    final_errors = np.array(results["Losses"])
    num_filters = [
        sum([config[k] for k in config if k.startswith("num_filters")])
        for config in results["Config"]
    ]
    learning_rates = [config["lr"] for config in results["Config"]]
    sizes = 10 + 90 * final_errors
    plt.xscale("log")
    plt.xlabel("learning rate"), plt.ylabel("# filters")
    plt.title("size, color $\\propto$ validation error at epoch 9")
    plt.scatter(learning_rates, num_filters, s=sizes, c=final_errors)
    plt.colorbar()
    plt.show()


def plot_error_curves(results) -> None:
    """Plot the validation errors over time (epochs).

    Args:
        results: Structure of metrics

    Returns:
        None
    """

    for val_errors in results["Val_errors"]:
        plt.plot(val_errors)

    plt.xlabel("epochs"), plt.ylabel("validation error")
    plt.title("Learning curves for different hyperparameters")
    plt.xticks(np.arange(0, 9), [str(i) for i in range(1, 10)])
    plt.axvline(0), plt.axvline(2), plt.axvline(8)
    plt.show()


def plot_incumbents(results) -> None:
    """Plot the incumbents of a particular method
    Args:
        results: Dictionary containing the results of the runs

    Returns:
        None
    """
    plt.plot(results["Epochs"], results["Incumbents"])
    plt.xlabel("# Epochs")
    plt.ylabel("Validation Error")
    plt.title("Incumbents")
    plt.xlim((0, 180))
    plt.grid()
    plt.show()


def plot_comparison(
    optimizer1_results,
    optimizer2_results,
    optimizer1_name,
    optimizer2_name,
    optimizer3_results=None,
    optimizer3_name=None,
) -> None:
    """Plot the comparison of two different optimizer incumbents
    Args:
        optimizer1_results: Dictionary containing results of the first optimizer
        optimizer2_results: Dictionary containing results of the second optimizer

    Returns:
        None
    """
    plt.plot(
        optimizer1_results["Epochs"],
        optimizer1_results["Incumbents"],
        label=optimizer1_name,
    )
    plt.plot(
        optimizer2_results["Epochs"],
        optimizer2_results["Incumbents"],
        label=optimizer2_name,
    )
    if optimizer3_name is not None and optimizer3_results is not None:
        plt.plot(
            optimizer3_results["Epochs"],
            optimizer3_results["Incumbents"],
            label=optimizer3_name,
        )
    plt.xlabel("# Epochs")
    plt.ylabel("Validation Error")
    plt.yscale("log")
    plt.title("Incumbents")
    plt.legend()
    plt.grid()
    plt.show()
