import numpy as np
from lib.rs_pipeline import get_pipeline_space


def test_rs_pipeline_space():
    err_msg = "Pipeline space in bo pipeline not implemented correctly"
    true_keys = ["lr", "num_filters_1", "num_filters_2"]
    true_lower_bounds = [1e-6, 2, 2]
    true_upper_bounds = [1e-0, 8, 8]
    pipeline_space = get_pipeline_space()
    for i, key in enumerate(pipeline_space.keys()):
        np.testing.assert_string_equal(key, true_keys[i])
        np.testing.assert_equal(
            pipeline_space[key].lower, true_lower_bounds[i], err_msg=err_msg
        )
        np.testing.assert_equal(
            pipeline_space[key].upper, true_upper_bounds[i], err_msg=err_msg
        )
    np.testing.assert_equal(pipeline_space[true_keys[0]].log, True)
    np.testing.assert_equal(pipeline_space[true_keys[1]].log, False)
    np.testing.assert_equal(pipeline_space[true_keys[2]].log, False)


if __name__ == "__main__":
    test_rs_pipeline_space()
    print("Test complete.")
