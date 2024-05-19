import logging

import neps
from lib.rs_pipeline import get_pipeline_space, run_pipeline
from lib.utilities import set_seed


def main():
    set_seed(115)
    logging.basicConfig(level=logging.INFO)

    pipeline_space = get_pipeline_space()
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/random_search",
        overwrite_working_directory=True,
        max_evaluations_total=20,
        searcher="random_search",
    )
    previous_results, pending_configs = neps.status("results/random_search")
    # neps.plot("results/random_search")


if __name__ == "__main__":
    main()
