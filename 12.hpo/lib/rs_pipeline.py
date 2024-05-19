import time

import neps
import torch
from lib.conv_model import get_conv_model
from lib.dataset_mnist import load_mnist_minibatched
from lib.train_epoch import training
from lib.utilities import evaluate_accuracy


def get_pipeline_space() -> dict:
    """Define a NePS hyperparameter search-space.

    hyperparameters:
        lr              from 1e-6 to 1e-0 (log, float)
        num_filters_1   from    2 to    8 (int)
        num_filters_2   from    2 to    8 (int)


    Returns:
        Dictionary containing the NePS pipeline space

    Note:
        Please name the hyperparameters and order them as given above (needed for testing)

    """

    # START TODO #################
    pipeline_space = {
        "lr": neps.FloatParameter(1e-6, 1e-0, log=True),
        "num_filters_1": neps.IntegerParameter(2, 8),
        "num_filters_2": neps.IntegerParameter(2, 8),
    }
    # END TODO ###################
    return pipeline_space


def run_pipeline(lr, num_filters_1, num_filters_2):
    """Train and evaluate a model given some configuration

    Args:
        lr: Learning rate passed to the optimizer to train the neural network
        num_filters_1: Number of filters for the first conv layer
        num_filters_2: Number of filters for the second conv layer

    Returns:
        Dictionary of loss and info_dict which contains additional information about the runs.

    Note:
        Keep in mind that we want to minimize the error (not loss function of
        the training procedure), for that we use (1-val_accuracy) as our val_error.

    """
    start = time.time()
    train_loader, validation_loader, test_loader = load_mnist_minibatched(
        batch_size=32, n_train=4096, n_valid=512
    )
    # define loss
    criterion = torch.nn.NLLLoss()
    max_epochs = 9
    val_errors = []

    # Populate a list specifying the number of filters for each convolutional layer
    num_filters_per_layer = [num_filters_1, num_filters_2]

    # START TODO ################
    # 1. Use the number of filters to create the model
    # 2. Define the optimizer (use SGD optimizer)
    model = get_conv_model([num_filters_1, num_filters_2])
    optim = torch.optim.SGD(model.parameters(), lr)
    # END TODO ################
    # 1. For each epoch until max_epochs is reached get the validation errors from
    # your training function and append them to the val_errors list
    for epoch in range(max_epochs):
        print("  Epoch {} / {} ...".format(epoch + 1, max_epochs).ljust(2))
        # START TODO ################
        val_error = training(
            model,
            optim,
            criterion,
            train_loader,
            validation_loader,
        )
        val_errors.append(val_error)
        # END TODO ################
    train_accuracy = evaluate_accuracy(model, train_loader)
    test_accuracy = evaluate_accuracy(model, test_loader)
    end = time.time()
    print(
        "  Epoch {} / {} Val Error: {}".format(
            max_epochs, max_epochs, val_errors[-1]
        ).ljust(2)
    )

    return {
        "loss": val_errors[-1],
        "info_dict": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_time": end - start,
            "val_errors": val_errors,
            "cost": max_epochs,
        },
    }
