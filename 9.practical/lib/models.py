"""CNN models to train"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1(nn.Module):
    """
    The CNN model with 3 filters, kernel size 5, and padding 2
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # initialize required parameters / layers needed to build the network
        num_of_filters = 3
        kernel_size = 5
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=num_of_filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Flatten(),
            nn.Linear(16 * 16 * num_of_filters, 10),
        )
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape [batch_size, *feature_dim] (minibatch of data)
        Returns:
            scores: Pytorch tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.layers(x)
        # END TODO #############
        return x


class ConvNet2(nn.Module):
    """
    The CNN model with 16 filters, kernel size 5, and padding 2
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        num_of_filters = 16
        kernel_size = 5
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=num_of_filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Flatten(),
            nn.Linear(16 * 16 * num_of_filters, 10),
        )
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.layers(x)
        # END TODO #############
        return x


class ConvNet3(nn.Module):
    """
    The CNN model with 16 filters, kernel size 3, and padding 1
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # Define the layers need to build the network
        num_of_filters = 16
        kernel_size = 3
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=num_of_filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Flatten(),
            nn.Linear(16 * 16 * num_of_filters, 10),
        )
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.layers(x)
        # END TODO #############
        return x


class ConvNet4(nn.Module):
    """
    The CNN model with 16 filters, kernel size 3, padding 1 and batch normalization
    """

    def __init__(self):
        super().__init__()
        # START TODO #############
        # Define the layers need to build the network
        num_of_filters = 16
        kernel_size = 3
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=num_of_filters,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            # num_features is the channel size
            nn.BatchNorm2d(num_features=num_of_filters),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Flatten(),
            nn.Linear(16 * 16 * num_of_filters, 10),
        )
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        x = self.layers(x)
        # END TODO #############
        return x


class ConvNet5(nn.Module):
    """Your custom CNN"""

    def __init__(self):
        super().__init__()

        # START TODO #############

        # This network is composed of a number of convolutional blocks
        # (definition of which is given below), and
        # a single linear layer. The convolutional blocks
        # half the width (which is equal to the height) of
        # each activation. On the contrary,
        # they double the number of channels with increasing blocks.
        # Therefore, Number of channels and the size (width or height)
        # of the layer with index `index` (zero-indexing is used)
        # is given below in respective functions

        def calc_out_num_channels(index):
            return 2 ** (initial_num_channels_power + index)

        def calc_out_size(index):
            return int(initial_size * 2**-index)

        # The parameters that are not tuned (constant) are given below
        # number of channels of image (rgb)
        initial_num_channels = 3
        # image width
        initial_size = 32
        # kernel size is held constant
        kernel_size = 3

        # Tuned parameters are given below
        num_conv_blocks = 2
        # Initial number of channels is the number of channels
        # of the first activation layer, the layer coming just after
        # the input. It is 2**initial_num_channels_power
        initial_num_channels_power = 5

        last_num_channels = calc_out_num_channels(num_conv_blocks - 1)
        last_activation_size = calc_out_size(num_conv_blocks)
        if last_activation_size < 1:
            raise ValueError(
                f"Last activation size {last_activation_size} is smaller than 1"
            )

        # A convolutional block
        # Note that stride of 1 is used and the padding preserves the size
        # The halving is done by the MaxPool module.
        def convblock(index):
            # zero-indexing is used
            out_channels = calc_out_num_channels(index)
            in_channels = initial_num_channels if index == 0 else out_channels // 2
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                # num_features is the channel size
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                ),
            )

        self.convlayers = nn.ModuleList([convblock(i) for i in range(num_conv_blocks)])
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                last_activation_size * last_activation_size * last_num_channels,
                10,
            ),
        )
        # END TODO #############

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: The input tensor with shape (batch_size, *feature_dim)
            The input to the network will be a minibatch of data

        Returns:
            scores: PyTorch Tensor of shape (N, C) giving classification scores for x
        """
        # START TODO #############
        for convlayer in self.convlayers:
            y = convlayer(x)
            x = y
        x = self.linear(x)
        # END TODO #############
        return x
