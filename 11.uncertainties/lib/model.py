import torch
import torch.nn as nn
import numpy as np
from lib.blr import BLR
from typing import Tuple

np.random.seed(0)
torch.manual_seed(0)


class DNGO(nn.Sequential):
    """
    This class takes the same arguments as a Sequential model, i.e. a sequence of layers e.g.
                nn.Linear(in_features=1, out_features=50, bias=True),
                nn.Tanh(),
                nn.Linear(in_features=50, out_features=50, bias=True),
                nn.Tanh(),
                nn.Linear(in_features=50, out_features=1, bias=True)
    and extends it with multiple functionalities to be bayesian about the last layer of the network.
    """

    def __init__(self, *args, **kwargs):
        """Init for the class"""
        super(DNGO, self).__init__(*args, **kwargs)
        self.last_hidden_layer_saved = False
        (
            self.mu_post,
            self.Sigma_post,
            self.last_hidden_layer,
            self.blr,
            self.last_hidden_layer_weights,
        ) = (None, None, None, None, None)

    def last_hidden_layer_features(self, x: torch.Tensor) -> np.array:
        """
        Use last hidden layer of the network shown on top of class
        to get outputs individually by only this layer. Additionally,
        the last hidden layer is saved to  last_hidden_layer variable

        Args:
            x:  inputs for the last layer of shape
                (n, D) n - number of samples and
                D - dimension of sample
        Return:
            output of the last layer (n, 50) n - number of samples and
                50 - out_features of the last hidden layer
        """

        if not self.last_hidden_layer_saved:
            self.last_hidden_layer = nn.Sequential(*list(self.children())[:-1])
            self.last_hidden_layer_saved = True

        return self.last_hidden_layer(x).detach().numpy()

    def fit_blr_model(
        self,
        mu_pre: np.array,
        Sigma_pre: np.array,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        noise: float,
    ) -> None:
        """Extract features out of the last layer, compute the
            posterior distribution and set posterior mean and posterior variance.

        Args:

            mu_pre: Numpy array of shape (D_hidden+1, 1) D_hidden - (hidden output features of
                    last hidden layer + 1 for bias)
            Sigma_pre: Numpy array of shape (D_hidden+1, D_hidden+1) D_hidden -
                    (hidden output features of last hidden layer +1 for bias)
            x_train: Torch tensor of shape (n, D) n - number of samples and
                      D -  dimension of sample
            y_train: Torch tensor of shape (n, D) n - number of samples and
                      D -  dimension of sample
            noise: noise
        Returns:
            None

        """

        # START TODO #######################
        # Use last_hidden_layer_features() above to generate the learned features from the last hidden layer of the NN
        # and then perform BLR to get the posterior distribution for weights (using the same method as used
        # in the first exercise for the 1-D and 2-D linear cases). Store these as variables of the class so they can be
        # reused later.
        # (keep in mind using bias = True)
        x_train_mapped = self.last_hidden_layer_features(x_train)
        self.blr = BLR(mu_pre, Sigma_pre, noise, True)
        self.blr.linreg_bayes(x_train_mapped, y_train)
        self.mu_post = self.blr.Mu_post
        self.Sigma_post = self.blr.Sigma_post
        # END TODO ########################

    def set_last_hidden_layer_weights(self) -> None:
        """
        Needed to implement the bonus part such that the last_hidden_layer_weights is used as mu_prior for linreg_bayes
        """
        last_hidden_layer_weights = self.state_dict()["4.weight"].T.detach().numpy()
        bias = self.state_dict()["4.bias"].detach().numpy()

        self.last_hidden_layer_weights = np.vstack(
            [last_hidden_layer_weights, bias]
        )  # for bias

    def predict_mean_and_std(self, x: torch.Tensor) -> Tuple[np.array, np.array]:
        """Compute the mean and std. of the output of the
            last hidden layer given x

        Args:
            x : Numpy array of shape (n, D) n - number of samples and
                      D -  dimension of sample
        Returns:
            predicted mean: Numpy array of shape (n, 1) n - number of samples and

            predicted std: Numpy array of shape (n, 1) n - number of samples and

        Note:
            Use the last hidden layer outputs for x in order to get the new data points
            for which you want to compute the new values accordingly to eqn. 2.9
            Make use of the posterior_predictive function you were supposed to implement
            in the previous exercise and return the result of it.


        """

        # START TODO ########################
        # Call last_hidden_layer_features(...) which returns an np.array directly
        x_mapped = self.last_hidden_layer_features(x)
        x_mean, x_std = self.blr.posterior_predictive(x_mapped)
        # END TODO ########################
        return x_mean, x_std


class EnsembleFeedForward:
    """Holds an ensemble of NNs which are used to get prediction means and uncertainties.

    Args:
        num_models: the number of NNs in the ensemble
        ensembled_nets: list of nn.Sequential NNs belonging to the ensemble
    """

    def __init__(self, ensembled_nets: list):
        """Init for the class"""
        self.ensembled_nets = ensembled_nets
        self.num_models = len(ensembled_nets)

    def individual_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return the individual predictions for each model

        Args:
            x : Torch tensor of shape (n, D) n - number of samples and
                      D -  dimension of sample
        Returns:
            The individual predictions of the NNs: Torch tensor of
            shape (n, 1, m) n - number of samples and
            m - number of models in ensemble

        Note:
            Iterate over the list of base NNs held by the ensemble and collect the predictions for each
            NN.
        """
        preds = []
        # START TODO ########################
        # just append predictions to preds and use torch.cat to combine them
        for net in self.ensembled_nets:
            preds.append(net(x))
        # in the tester, the ensmbled_pred is 3d, so I decided to use stack instead of cat
        preds = torch.stack(preds, dim=2)
        # END TODO ########################
        return preds.detach().numpy()

    def predict_mean_and_std(self, x: torch.Tensor) -> Tuple[np.array, np.array]:
        """Compute the mean and std. of each point x.

        Args:
            x : Numpy array of shape (n, D) n - number of samples and
                      D -  dimension of sample
        Returns:
            predicted mean of the ensembled networks: Numpy array of
            shape (n, 1) n - number of samples

            predicted std. deviation of the ensembled networks: Numpy array of
            shape (n, 1) n - number of samples

        """

        # START TODO ########################
        # Iterate over the list of base NNs held by the ensemble and collect the predictions for each NN. Then take the
        # mean and std. dev. over the predictions and return them
        preds = self.individual_predictions(x)
        preds = torch.from_numpy(preds)
        mean = preds.mean(dim=2)
        std = preds.std(dim=2)
        # END TODO ########################

        return mean.detach().numpy(), std.detach().numpy()
