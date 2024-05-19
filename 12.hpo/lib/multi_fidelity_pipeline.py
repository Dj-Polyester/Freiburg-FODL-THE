import time

import neps
import torch
from lib.conv_model import get_conv_model
from lib.dataset_mnist import load_mnist_minibatched
from lib.train_epoch import training
from lib.utilities import evaluate_accuracy
from neps.utils.common import load_checkpoint, save_checkpoint


def get_pipeline_space() -> dict:
    """Define a hyperparameter search-space.

    Note: When using PriorBand an assumption about the best value for the hyperparamater
    has to be made. In the NePS library, this is commonly referred to as "default".

    hyperparameters:
      num_filters_1   from    4 to   32 (int, log, default = 12, default_confidence = "medium")
      num_filters_2   from    4 to   32 (int, log, default = 12, default_confidence = "medium")
      num_filters_3   from    4 to   32 (int, log,  default = 28, default_confidence = "medium)
      lr              from 1e-6 to 1e-1 (float, log, default = 1e-2, default_confidence = "medium")
      optimizer            Adam or  SGD (categorical, order is important for tests, default = Adam
                                            default_confidence = "medium")
      epochs          from 1 to 9 (fidelity parameter)

    Returns:
        Pipeline space dictionary

    Note:
        Please name the hyperparameters and order them as given above (needed for testing)
    """

    # START TODO ################
    pipeline_space = {
        "num_filters_1": neps.IntegerParameter(
            4,
            32,
            log=True,
            default=12,
            default_confidence="medium",
        ),
        "num_filters_2": neps.IntegerParameter(
            4,
            32,
            log=True,
            default=12,
            default_confidence="medium",
        ),
        "num_filters_3": neps.IntegerParameter(
            4,
            32,
            log=True,
            default=28,
            default_confidence="medium",
        ),
        # Test wants lr = 1e-3 whereas the function suggests 1e-2
        "lr": neps.FloatParameter(
            1e-6,
            1e-1,
            log=True,
            default=1e-3,
            default_confidence="medium",
        ),
        "optimizer": neps.CategoricalParameter(
            ["Adam", "SGD"],
            default="Adam",
            default_confidence="medium",
        ),
        "epochs": neps.IntegerParameter(1, 9, is_fidelity=True),
    }
    # END TODO ################
    return pipeline_space


def run_pipeline(
    pipeline_directory,
    previous_pipeline_directory,
    num_filters_1,
    num_filters_2,
    num_filters_3,
    lr,
    optimizer,
    epochs=9,
) -> dict:
    """Evaluate a function with the given parameters and return a loss.
        NePS tries to minimize the returned loss. In our case the function is
        the training and validation of a model, the budget is the number of
        epochs and the val_error(1-validation_accuracy) which we use as our loss.

    Args:
        num_filters_1: Number of filters for the first conv layer
        num_filters_2: Number of filters for the second conv layer
        num_filters_3: Number of filters for the third conv layer
        lr: Learning rate passed to the optimizer to train the neural network
        optimizer: Optimizer used to train the neural network ("Adam" or "SGD")
        epochs: Number of epochs to train the model(if not set by NePS it is by default 9)
        pipeline_directory: Directory where the trained model will be saved
        previous_pipeline_directory: Directory containing stored model of previous HyperBand iteration

    Returns:
        Dictionary of loss and info_dict which contains additional information about the runs.

    Note:
        Please notice that the optimizer is determined by the pipeline space.
    """
    start = time.time()

    # START TODO ################
    # retrieve the number of filters and create the model
    model = get_conv_model([num_filters_1, num_filters_2, num_filters_3])
    # END TODO ################
    train_loader, validation_loader, test_loader = load_mnist_minibatched(
        batch_size=32, n_train=4096, n_valid=512
    )
    # define loss
    criterion = torch.nn.NLLLoss()

    # Define the optimizer, make sure to name the variable `optim` for compatability
    # with the load_checkpoint function below.
    # START TODO ################
    optim = getattr(torch.optim, optimizer)(model.parameters(), lr=lr)
    # END TODO ##################

    # make use of checkpointing to resume training models on higher fidelities
    previous_state = load_checkpoint(  # predefined function from neps
        directory=previous_pipeline_directory,
        model=model,
        optimizer=optim,
    )
    # adjusting run budget based on checkpoint
    if previous_state is not None:
        start_epoch = previous_state["epochs_already_trained"]
    else:
        start_epoch = 0

    val_errors = list()

    for epoch in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(epoch + 1, epochs).ljust(2))
        # Call the training function, get the validation errors and append them to val errors
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

    save_checkpoint(
        directory=pipeline_directory,
        model=model,
        optimizer=optim,
        values_to_save={
            "epochs_already_trained": epochs,
        },
    )
    end = time.time()
    print(
        "  Epoch {} / {} Val Error: {}".format(epochs, epochs, val_errors[-1]).ljust(2)
    )
    return {
        "loss": val_errors[-1],
        "info_dict": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "val_errors": val_errors,
            "train_time": end - start,
            "cost": epochs - start_epoch,
        },
        "cost": epochs - start_epoch,
    }
