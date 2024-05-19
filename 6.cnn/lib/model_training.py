"""Helper functions for model training."""

from typing import Tuple

import numpy as np

from lib.model_evaluation import accuracy, evaluate


def train(model, loss_fn, optimizer, x_train, y_train, x_val, y_val,
          num_epochs, batch_size, scheduler=None
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Train a model

    Args:
        model: Model to train.
        loss_fn: Loss function to use.
        optimizer: Optimizer to use.
        x_train: Train data.
        y_train: Train labels.
        x_val: Validation data.
        y_val: Validation labels.
        num_epochs: Number of epochs.
        batch_size: Batch size.
        scheduler: Learning rate scheduler to use.

    Returns:
        4-tuple of (train costs, train accuracies, evaluation costs, evaluation accuracies), each of which is
        a list with num_epochs entries.

    """
    # initialize arrays to store losses and accuracies
    train_costs, train_accuracies = np.zeros(num_epochs), np.zeros(num_epochs)
    eval_costs, eval_accuracies = np.zeros(num_epochs), np.zeros(num_epochs)

    # calculate how many minibatches we get out of the training dataset given the batch size
    num_train_batches = len(x_train) // batch_size
    assert len(x_train) % batch_size == 0, (
        f"Training dataset size of {len(x_train)} is not divisible by batch size {batch_size}.")

    # create indices of the training data to shuffle it later
    train_idx = np.arange(len(x_train))

    for epoch in range(num_epochs):
        print("Epoch {} / {}:".format(epoch + 1, num_epochs))
        training_predictions = []

        # shuffle training data order
        np.random.shuffle(train_idx)
        x_train_shuffled = x_train[train_idx]
        y_train_shuffled = y_train[train_idx]

        # train for one epoch
        model.train()
        for batch_num in range(num_train_batches):
            # get the minibatch data given the current minibatch number
            minibatch_start = batch_num * batch_size
            minibatch_end = (batch_num + 1) * batch_size
            x_batch = x_train_shuffled[minibatch_start:minibatch_end]
            y_batch = y_train_shuffled[minibatch_start:minibatch_end]

            # zero gradients
            optimizer.zero_grad()

            # do the forward pass, remember the predictions and calculate the loss
            y_batch_predicted = model(x_batch)
            training_predictions.append(y_batch_predicted)
            loss = loss_fn(y_batch_predicted, y_batch)

            # do the backward pass
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step()

            # aggregate loss for this epoch
            train_costs[epoch] += loss

        # normalize loss over number of steps done
        train_costs[epoch] /= num_train_batches

        # step the LR scheduler if needed
        if scheduler:
            scheduler.step()

        # set model to evaluation mode
        model.eval()

        # concatenate the minibatch training predictions back together and calculate the accuracy
        training_predictions = np.concatenate(training_predictions, axis=0)
        train_accuracies[epoch] = accuracy(y_train_shuffled, training_predictions)
        print("  Training Accuracy: {:.4f}".format(train_accuracies[epoch]))
        print("  Training Cost: {:.4f}".format(train_costs[epoch]))

        # evaluate
        eval_accuracies[epoch], eval_costs[epoch] = evaluate(x_val, y_val, model, loss_fn, batch_size)
        print("  Eval Accuracy: {:.4f}".format(eval_accuracies[epoch]))
    return train_costs, train_accuracies, eval_costs, eval_accuracies
