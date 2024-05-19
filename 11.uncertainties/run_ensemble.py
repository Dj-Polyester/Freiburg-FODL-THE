"""
    Starting from create_ensembled function to create an ensemble of differently
    initialized base NNs (based on normal distribution). Performing same training
    procedure for each network in the ensemble as done for DNGO.
"""

import torch

import numpy as np
from lib.training import train_evaluate_model
from lib.utilities import rescaled_sinc, save_result, create_ensembled
from lib.model import EnsembleFeedForward
import logging

logging.basicConfig(level=logging.INFO)


def run_ensemble(n_models, ensembled_nets, epochs, batch_size, lr):
    x_train, y_train, x_test, y_test = rescaled_sinc()

    ensemble = EnsembleFeedForward(
        ensembled_nets=ensembled_nets)

    num_models = n_models
    for i in np.arange(num_models):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(
            ensemble.ensembled_nets[i].parameters(), lr=lr)

        def lambda1(_): return 1.

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda1)

        logging.info(
            'Start Training for model {}/{} of Ensemble'.format(i + 1, num_models))

        train_evaluate_model(model=ensemble.ensembled_nets[i], optimizer=optimizer,
                             loss_func=criterion, scheduler=scheduler,
                             x_train=x_train, y_train=y_train, x_test=x_test,
                             y_test=y_test, epochs=epochs, batch_size=batch_size)

    return ensemble


if __name__ == '__main__':
    np.random.seed(42)
    n_models = 5
    lr = 1e-2
    n_train = 30
    epochs = 10000
    batch_size = n_train // 2
    ensembled_nets = create_ensembled(num_models=n_models)
    ensemble = run_ensemble(n_models, ensembled_nets, epochs, batch_size, lr)
    for i in range(n_models):
        save_result(ensemble.ensembled_nets[i], name=f"ensemble_{i}")
