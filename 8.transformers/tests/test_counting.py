import numpy as np
import torch

from lib.counting import CountingModel


def test_counting_model_forward():
    torch.manual_seed(0)

    batch_size = 2
    d_in = 2
    model = CountingModel(d_in, d_in - 1, 4)
    x = torch.rand(batch_size, 1, d_in)
    out, attention_weights = model(x)

    expected_out = np.array(
        [
            [[0.02352506, -0.00647369], [-0.21286917, 0.22818303]],
            [[0.15463954, 0.03659275], [-0.17983317, 0.34440082]],
        ],
        dtype=float,
    )

    expected_weights = np.array(
        [[[0.52600944], [0.42547926]], [[0.5409462], [0.46046096]]]
    )

    err_msg = "Attention forward pass not implemented correctly"
    np.testing.assert_allclose(
        out.detach().numpy(), expected_out, err_msg=err_msg, rtol=1e-4
    )
    np.testing.assert_allclose(
        attention_weights.detach().numpy(), expected_weights, err_msg=err_msg, rtol=1e-4
    )


def test_counting_model_backward():
    torch.manual_seed(0)
    torch.set_printoptions(precision=6)
    batch_size = 2
    d_in = 2
    model = CountingModel(d_in, d_in - 1, 4)
    x = torch.rand(batch_size, 1, d_in)
    out, attention_weights = model(x)
    loss_function = torch.nn.MSELoss()
    expected_out = torch.tensor(
        [
            [[0.40, 0.75], [0.11, 0.42]],
            [[0.40, 0.75], [0.39, 0.74]],
        ]
    )
    torch.set_printoptions(precision=9)
    loss = loss_function(out, expected_out)
    model.zero_grad()
    loss.backward()

    expected_grads = np.array(
        [
            [
                [
                    [0.5439603, -0.31151295, -0.229666, -0.00311968],
                    [0.1365527, -0.1701021, -0.04166017, 0.07216725],
                ]
            ],
            [-0.13061608, -0.13299255, 0.01616891, 0.10634904],
            [0.2728275, -0.13390714, 0.0175142, 0.05266769],
            [
                [0.00096021, -0.00032015],
                [0.00177802, -0.00037316],
                [0.00097361, 0.00027986],
                [0.00175723, 0.00054542],
            ],
            [0.00042343, 0.00149893, 0.00239653, 0.0044566],
            [
                [0.12046233, 0.05148267],
                [-0.08562218, -0.03438438],
                [-0.04600086, -0.02152687],
                [0.0111607, 0.00442859],
            ],
            [0.35137174, -0.24256088, -0.14025462, 0.03144376],
            [
                [0.24185222, -0.15922914, 0.37070832, -0.45333138],
                [0.17896396, -0.3947002, 0.6528858, -0.43714952],
            ],
            [-0.37863445, -0.51432425],
        ],
        dtype=object,
    )

    err_msg = "Attention forward pass not implemented correctly"
    for i, param in enumerate(model.parameters()):
        np.testing.assert_allclose(
            param.grad.detach().numpy(), expected_grads[i], rtol=1e-3, err_msg=err_msg
        )


if __name__ == "__main__":
    test_counting_model_forward()
    test_counting_model_backward()
    print("Test complete.")
