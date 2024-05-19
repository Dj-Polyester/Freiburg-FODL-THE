import numpy as np
import torch
from lib.conv_model import get_conv_model
from lib.dataset_mnist import load_mnist_minibatched
from lib.train_epoch import training
from lib.utilities import evaluate_accuracy, set_seed


def test_train_epoch():
    set_seed(115)
    err_msg = "Training function is not implemented correctly"
    train_loader, validation_loader, test_loader = load_mnist_minibatched(
        batch_size=32, n_train=4096, n_valid=512
    )
    val_errors = list()
    num_filters_per_layer = [3, 3]
    model = get_conv_model(num_filters_per_layer)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = torch.nn.NLLLoss()
    epochs = 1
    for epoch in range(epochs):
        val_errors = training(
            model, optimizer, criterion, train_loader, validation_loader
        )
    train_accuracy = evaluate_accuracy(model, train_loader)
    test_accuracy = evaluate_accuracy(model, test_loader)

    np.testing.assert_allclose(
        val_errors, np.array(0.931641), atol=1e-5, err_msg=err_msg
    )
    np.testing.assert_allclose(train_accuracy, 0.082275, atol=1e-5, err_msg=err_msg)
    np.testing.assert_allclose(test_accuracy, 0.0869, atol=1e-5, err_msg=err_msg)


if __name__ == "__main__":
    test_train_epoch()
    print("Test complete.")
