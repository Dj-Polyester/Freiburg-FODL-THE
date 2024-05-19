"""Weight distribution plot."""

import matplotlib.pyplot as plt
import numpy as np

from lib.utilities import load_result


def plot_distribution() -> None:
    """Plot the histograms of the weight values of the five models (before_training, no regularization,
    L1 regularization, L2 regularization and Dropout) from -1 to 1 with 100 bins.

    Returns:
        None
    """

    models = load_result("trained_models")

    fig, axs = plt.subplots(figsize=(12, 6), ncols=5, sharey=True)
    axs[0].set(yscale="log", ylabel="total frequency")
    for model, ax in zip(models.keys(), axs):
        ax.set(title=model, xlabel="value")
        # START TODO ################
        # Retrieve all the weights (exclude the biases!) of the parameters for each of the five models
        # (before training, no regularization, L1 regularization, L2 regularization and Dropout)
        # and then plot the histogram as specified

        real_model = models[model]
        model_params = np.hstack(
            [
                param.data.flatten()
                for param in real_model.parameters()
                if param.name == "W"
            ]
        )
        # I tried both normalizing and limiting the values between -1 and 1.
        # However, the information I got was less doing either of them.
        # Therefore I decided not doing both of them.

        # normalize between -1 and 1
        # model_params = (model_params - model_params.min()) / (
        #     model_params.max() - model_params.min()
        # )
        # limit between -1 and 1
        # ax.set_xlim(-1, 1)
        ax.hist(model_params, bins=100)
        # END TODO ################
    fig.tight_layout()
    plt.show()
