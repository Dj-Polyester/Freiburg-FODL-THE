"""ReLU plot."""

import matplotlib.pyplot as plt
import numpy as np

from lib.activations import ReLU


def plot_relu() -> None:
    """Plot the ReLU function in the range (-4, 4).

    Returns:
        None
    """
    # START TODO #################
    # Create input data, run through ReLU and plot.
    # create input
    x = np.linspace(-4, +4, 101)

    # create sigmoid module and apply it to the input
    relu = ReLU()
    y = relu(x)  # equal to call of sigmoid.forward(x)

    # plot input and sigmoid output
    plt.plot(x, y)
    plt.xlabel("input")
    plt.ylabel("output")
    plt.grid(True)
    plt.show()
    # END TODO###################
