import numpy as np
from lib.conv_model import get_conv_model


def test_convnet1():
    expected_repr = """Sequential(
                          (conv1): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=same)
                          (relu1): ReLU()
                          (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                          (conv2): Conv2d(2, 2, kernel_size=(3, 3), stride=(1, 1), padding=same)
                          (relu2): ReLU()
                          (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                          (flatten): Flatten(start_dim=1, end_dim=-1)
                          (linear): Linear(in_features=98, out_features=10, bias=True)
                          (log_softmax): LogSoftmax(dim=1)
                        )
                    """

    convnet = get_conv_model([2, 2])
    repr_str = str(convnet)

    line_by_line_expected_repr = expected_repr.split("\n")[1:-2]
    line_by_line_repr_str = repr_str.split("\n")[1:-2]

    err_msg = "get_conv_model not implemented correctly"

    for expected, given in zip(line_by_line_expected_repr, line_by_line_repr_str):
        expected = expected.strip().split(": ")[1]
        given = given.strip().split(": ")[1]
        np.testing.assert_equal(given, expected, err_msg=err_msg)


if __name__ == "__main__":
    test_convnet1()
    print("Test complete.")
