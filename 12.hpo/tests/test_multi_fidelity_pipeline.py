import neps
import numpy as np
from lib.multi_fidelity_pipeline import get_pipeline_space, run_pipeline
from lib.utilities import get_results, set_seed


def test_hyperband_pipeline():
    set_seed(115)
    err_msg = "run_pipeline in hyperband_pipeline not implemented correctly"
    pipeline_space = get_pipeline_space()
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/test_hyperband",
        overwrite_working_directory=True,
        max_evaluations_total=1,
        loss_value_on_error=0.1,
        searcher="hyperband",
    )
    results = get_results("results/test_hyperband")
    np.testing.assert_allclose(
        results["Losses"][0], 0.908203, atol=1e-5, err_msg=err_msg
    )
    np.testing.assert_allclose(
        results["Config"][0]["lr"], 0.0007656032, atol=1e-5, err_msg=err_msg
    )
    np.testing.assert_allclose(
        results["Val_errors"][0][0], 0.908203, atol=1e-5, err_msg=err_msg
    )
    np.testing.assert_equal(results["Config"][0]["epochs"], 1, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["num_filters_1"], 5, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["num_filters_2"], 17, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["num_filters_3"], 9, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["optimizer"], "SGD", err_msg=err_msg)


def test_priorband_pipeline():
    set_seed(115)
    err_msg = "run_pipeline in priorband_pipeline not implemented correctly"
    pipeline_space = get_pipeline_space()
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/test_priorband",
        overwrite_working_directory=True,
        max_evaluations_total=1,
        loss_value_on_error=0.1,
        searcher="priorband",
    )
    results = get_results("results/test_priorband")
    np.testing.assert_allclose(
        results["Losses"][0], 0.136719, atol=1e-2, err_msg=err_msg
    )
    np.testing.assert_allclose(
        results["Config"][0]["lr"], 1e-3, atol=1e-5, err_msg=err_msg
    )
    np.testing.assert_allclose(
        results["Val_errors"][0][0], 0.136719, atol=1e-2, err_msg=err_msg
    )
    np.testing.assert_equal(results["Config"][0]["epochs"], 1, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["num_filters_1"], 12, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["num_filters_2"], 12, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["num_filters_3"], 28, err_msg=err_msg)
    np.testing.assert_equal(results["Config"][0]["optimizer"], "Adam", err_msg=err_msg)


if __name__ == "__main__":
    test_hyperband_pipeline()
    test_priorband_pipeline()
    print("Test complete.")
