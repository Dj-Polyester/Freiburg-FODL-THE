import argparse

from lib.plot import plot_comparison
from lib.utilities import get_results, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--searcher1",
        type=str,
        choices=["random_search", "hyperband", "priorband"],
        required=True,
        help='Specify either "hyperband" or "priorband".',
    )
    parser.add_argument(
        "--searcher2",
        type=str,
        choices=["random_search", "hyperband", "priorband"],
        required=True,
        help='Specify either "hyperband" or "priorband".',
    )
    parser.add_argument(
        "--searcher3",
        type=str,
        choices=["random_search", "hyperband", "priorband"],
        required=False,
        help='Specify either "hyperband" or "priorband".',
    )
    args = parser.parse_args()

    searcher1 = args.searcher1
    searcher2 = args.searcher2
    searcher3 = args.searcher3

    if searcher1 == "random_search":
        searcher1 = "random_search_biggerSpace"
    if searcher2 == "random_search":
        searcher2 = "random_search_biggerSpace"
    if searcher3 == "random_search":
        searcher3 = "random_search_biggerSpace"
    set_seed(124)

    if searcher3 is not None:
        losses_and_config_s1 = get_results(f"results/{searcher1}")
        losses_and_config_s2 = get_results(f"results/{searcher2}")
        losses_and_config_s3 = get_results(f"results/{searcher3}")
        plot_comparison(
            losses_and_config_s1,
            losses_and_config_s2,
            searcher1,
            searcher2,
            optimizer3_results=losses_and_config_s3,
            optimizer3_name=searcher3,
        )
    else:
        losses_and_config_s1 = get_results(f"results/{searcher1}")
        losses_and_config_s2 = get_results(f"results/{searcher2}")
        plot_comparison(
            losses_and_config_s1,
            losses_and_config_s2,
            searcher1,
            searcher2,
        )


if __name__ == "__main__":
    main()
