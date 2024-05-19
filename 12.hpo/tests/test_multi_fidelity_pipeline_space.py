import numpy as np
from lib.multi_fidelity_pipeline import get_pipeline_space


def test_multi_fidelity_pipeline_space():
    err_msg = "Priorband pipeline space is not implemented correctly"
    true_keys = [
        "num_filters_1",
        "num_filters_2",
        "num_filters_3",
        "lr",
        "optimizer",
        "epochs",
    ]
    true_lower_bounds = [4, 4, 4, 1e-6, "Adam", 1]
    true_upper_bounds = [32, 32, 32, 1e-1, "Adam", 9]
    true_defaults = [12, 12, 28, 1e-3, "Adam", None]
    default_confidence = [0.25, 0.25, 0.25, 0.25, 4, 0.5]
    true_optimizer_choices = ["Adam", "SGD"]
    pipeline_space = get_pipeline_space()

    pipeline_space = get_pipeline_space()
    for i, key in enumerate(pipeline_space.keys()):
        np.testing.assert_string_equal(key, true_keys[i])
        np.testing.assert_equal(
            pipeline_space[key].lower, true_lower_bounds[i], err_msg=err_msg
        )
        np.testing.assert_equal(
            pipeline_space[key].upper, true_upper_bounds[i], err_msg=err_msg
        )
        np.testing.assert_equal(
            pipeline_space[key].default, true_defaults[i], err_msg=err_msg
        )
        np.testing.assert_equal(
            pipeline_space[key].default_confidence_score,
            default_confidence[i],
            err_msg=err_msg,
        )
    np.testing.assert_equal(pipeline_space[true_keys[0]].log, True)
    np.testing.assert_equal(pipeline_space[true_keys[1]].log, True)
    np.testing.assert_equal(pipeline_space[true_keys[2]].log, True)
    np.testing.assert_equal(pipeline_space[true_keys[3]].log, True)
    np.testing.assert_equal(
        pipeline_space[true_keys[4]].choices, true_optimizer_choices, err_msg=err_msg
    )
    np.testing.assert_equal(
        pipeline_space[true_keys[5]].is_fidelity, True, err_msg=err_msg
    )


if __name__ == "__main__":
    test_multi_fidelity_pipeline_space()
    print("Test complete.")
