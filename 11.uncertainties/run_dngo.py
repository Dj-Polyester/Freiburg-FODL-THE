# Some initial setup: Importing required libraries

import torch
import numpy as np
from lib.training import train_evaluate_model
from lib.utilities import rescaled_sinc, save_result, init_weights, create_single_model
from lib.model import DNGO
import logging

logging.basicConfig(level=logging.INFO)


def run_dngo(dngo_model, epochs, batch_size, lr):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(dngo_model.parameters(), lr=lr)

    def lambda1(_): return 1.

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    x_train, y_train, x_test, y_test = rescaled_sinc()
    logging.info('Start Training DNGO model')
    train_evaluate_model(model=dngo_model, optimizer=optimizer,
                         loss_func=criterion, scheduler=scheduler,
                         x_train=x_train, y_train=y_train, x_test=x_test,
                         y_test=y_test, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    np.random.seed(42)
    # play around with the hyperparameters if you like
    lr = 1e-2
    epochs = 10000
    n_train = 30
    batch_size = n_train // 2

    module_list = create_single_model()
    dngo_model = DNGO(*module_list).apply(init_weights)

    run_dngo(dngo_model, epochs, batch_size, lr)
    save_result(dngo_model, name="DNGO")
