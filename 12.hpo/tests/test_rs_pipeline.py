import neps
import numpy as np
from lib.rs_pipeline import get_pipeline_space, run_pipeline
from lib.utilities import get_results, set_seed


def test_rs_pipeline():
    set_seed(115)
    err_msg = "run_pipeline in bo_rs_pipeline not implemented correctly"
    pipeline_space = get_pipeline_space()
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/test_random_search",
        overwrite_working_directory=True,
        max_evaluations_total=1,
        loss_value_on_error=0.1,
        searcher="random_search",
    )
    results = get_results("results/test_random_search")

    np.testing.assert_allclose(
        results["Losses"][0], 0.94921875, atol=1e-5, err_msg=err_msg
    )
    np.testing.assert_allclose(
        results["Config"][0]["lr"], 1.50159861e-05, atol=1e-5, err_msg=err_msg
    )
    np.testing.assert_equal(results["Config"][0]["num_filters_1"], 6, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["num_filters_2"], 4, err_msg=err_msg)
    np.testing.assert_array_equal(
        results["Val_errors"][0],
        np.array(
            [
                0.951171875,
                0.951171875,
                0.951171875,
                0.951171875,
                0.951171875,
                0.951171875,
                0.951171875,
                0.94921875,
                0.94921875,
            ]
        ),
        err_msg=err_msg,
    )


if __name__ == "__main__":
    test_rs_pipeline()
    print("Test complete.")
