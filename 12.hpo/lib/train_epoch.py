from lib.utilities import evaluate_accuracy


def training(model, optimizer, criterion, train_loader, validation_loader) -> float:
    """
    Function that trains the model for one epoch and evaluates the model on the validation set. Used by the searcher.

    Args:
        model (nn.Module): Model to be trained.
        optimizer (torch.nn.optim): Optimizer used to train the weights (depends on the pipeline space).
        criterion (nn.modules.loss) : Loss function to use.
        train_loader (torch.utils.Dataloader): Data loader containing the training data.
        validation_loader (torch.utils.Dataloader): Data loader containing the validation data.

    Returns:
        (float) validation error for the epoch.
    """
    # START TODO ################
    # Train
    for X, y in train_loader:
        y_hat = model(X)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Evaluate
    validation_accuracy = evaluate_accuracy(model, validation_loader)
    # END TODO ################

    return 1 - validation_accuracy
