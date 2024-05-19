import matplotlib.pyplot as plt
import numpy as np


def plot_data(
    data: np.ndarray,
    rows: int = 5,
    cols: int = 4,
    plot_border: bool = True,
    title: str = "",
) -> None:
    """Plot the given image data.

    Args:
        data: image data shaped (n_samples, channels, width, height).
        rows: number of rows in the plot .
        cols: number of columns in the plot.
        plot_border: add a border to the plot of each individual digit.
                     If True, also disable the ticks on the axes of each image.
        title: add a title to the plot.

    Returns:
        None

    Note:

    """
    # START TODO ################
    # useful functions: plt.subplots, plt.suptitle, plt.imshow

    fig, axs = plt.subplots(nrows=rows, ncols=cols)

    # Uncomment 2 lines below for randomization
    # nsize = rows * cols
    # indices = np.sort(np.random.choice(len(data), nsize))

    # For auto margin
    fig.tight_layout()
    for index, ax in enumerate(axs.flatten()):
        # Uncomment the line below for randomization
        # index = indices[i]

        # Data is 3d, reduce it to 2d
        dataSqueezed = data[index].squeeze()
        # I wanted to show the user the size of the images on axes
        ax.set_yticks([0, dataSqueezed.shape[0] - 1])
        ax.set_xticks([0, dataSqueezed.shape[1] - 1])
        if not plot_border:
            ax.axis("off")
        ax.imshow(dataSqueezed)

    plt.suptitle(title)
    plt.show()
    # END TODO ################
