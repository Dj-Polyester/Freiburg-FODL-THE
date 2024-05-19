# Some initial setup: Importing required libraries
import numpy as np
from lib.utilities import load_result, rescaled_sinc, create_ensembled
from lib.model import EnsembleFeedForward
from lib.plot import plot_multiple_predictions, plot_uncertainty


def eval_ensemble(ensemble_model):
    x_train, y_train, x_test, y_test = rescaled_sinc()
    plot_multiple_predictions(ensemble_model, x_train, y_train)
    plot_uncertainty(ensemble_model, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    np.random.seed(42)
    num_models = 5
    ensembled_nets = create_ensembled(num_models=num_models)
    ensemble = EnsembleFeedForward(ensembled_nets=ensembled_nets)
    for i in range(num_models):
        load_result(ensemble.ensembled_nets[i], name=f"ensemble_{i}")

    eval_ensemble(ensemble)
