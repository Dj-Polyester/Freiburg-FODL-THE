import argparse
import logging

import neps
from lib.multi_fidelity_pipeline import get_pipeline_space, run_pipeline
from lib.utilities import set_seed


def main():
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
    set_seed(115)
    pipeline_space = get_pipeline_space()

    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=f"results/{searcher}",
        overwrite_working_directory=True,
        max_cost_total=20 * 9,
        searcher=f"{searcher}",
    )
    previous_results, pending_configs = neps.status(f"results/{searcher}")
    # neps.plot(f"results/{searcher}")


if __name__ == "__main__":
    main()
