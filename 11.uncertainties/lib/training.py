import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple
import logging

np.random.seed(0)
torch.manual_seed(0)


def train_evaluate_model(
        model,
        optimizer,
        loss_func,
        scheduler,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs,
        batch_size):
    train_losses = []
    test_losses = []

    for epoch in range(int(epochs)):
        if (epoch + 1) % 1000 == 0:
            logging.info("Epoch {} / {} ...".format(epoch + 1, epochs))
        model.train()

        ix = np.arange(len(x_train))
        np.random.shuffle(ix)

        shuffled_data = zip(minibatched(x_train[ix], batch_size), minibatched(y_train[ix], batch_size))

        for i, (x, y) in enumerate(shuffled_data):
            optimizer.zero_grad()
            output = model(x).view(-1,)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()

        train_loss = evaluate_loss(model, loss_func, x_train, y_train)
        test_loss = evaluate_loss(model, loss_func, x_test, y_test)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if (epoch + 1) % 1000 == 0:
            logging.info('Train loss: {} and Test loss: {}'.format(train_losses[-1], test_losses[-1]))
        scheduler.step()

    return train_losses, test_losses


def minibatched(data: np.ndarray, batch_size: int) -> List[np.ndarray]:
    """Mini-batchifies data"""
    assert len(data) % batch_size == 0, ("Data length {} is not multiple of batch size {}"
                                         .format(len(data), batch_size))
    return data.reshape(-1, batch_size, *data.shape[1:])


def evaluate_loss(model: nn.Module, loss_func: nn.Module,
                  x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Evaluates given loss function for given data for the given model"""
    model.eval()
    with torch.no_grad():
        output = model(x).view(-1,)
        loss = loss_func(output, y)
    return loss.numpy()
