import logging

import neps
from lib.multi_fidelity_pipeline import get_pipeline_space, run_pipeline
from lib.utilities import set_seed


def main():
    set_seed(115)
    pipeline_space = get_pipeline_space()
    del pipeline_space["epochs"]

    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/random_search_biggerSpace",
        overwrite_working_directory=True,
        max_cost_total=20 * 9,
        searcher="random_search",
    )
    previous_results, pending_configs = neps.status("results/random_search_biggerSpace")
    # neps.plot("results/random_search_biggerSpace")


if __name__ == "__main__":
    main()
