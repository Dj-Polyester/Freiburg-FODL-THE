import argparse

from lib.plot import plot_error_curves, plot_incumbents
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

    # Plot the validation errors over time (epochs)
    plot_error_curves(losses_and_config)

    # Plot the incumbents
    plot_incumbents(losses_and_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--searcher",
        type=str,
        choices=["hyperband", "priorband"],
        required=True,
        help='Specify either "hyperband" or "priorband".',
    )
    args = parser.parse_args()

    searcher = args.searcher
    plot_results(f"results/{searcher}")
