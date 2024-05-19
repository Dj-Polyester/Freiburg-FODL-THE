from lib.plot import plot_error_curves, plot_incumbents, plot_lr_vs_filter
from lib.utilities import get_results


def plot_results(results) -> None:
    """Plot the results from different runs.

    Args:
        results: Structure containing results of the runs

    Returns:
        None

    """
    # Get structured results
    losses_and_config = get_results(results)

    # Plot learning rate versus number of filter
    plot_lr_vs_filter(losses_and_config)

    # Plot the validation errors over time (epochs)
    plot_error_curves(losses_and_config)

    # Plot the incumbents
    plot_incumbents(losses_and_config)


if __name__ == "__main__":
    plot_results(f"results/random_search")
