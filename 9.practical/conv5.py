import torch
import torch.nn as nn
import torch.nn.functional as F


def seqconv(self, x):
    for convlayer in self.convlayers:
        y = convlayer(x)
        x = y
    return x


def convblock(
    convblock_variant,
    index,
    kernel_size,
    padding,
    initial_num_channels,
    calc_out_num_channels,
    pooldegree,
    initial_num_channels_power,
):
    # zero-indexing is used
    out_channels = calc_out_num_channels(initial_num_channels_power, index)
    in_channels = (
        initial_num_channels
        if index == 0
        else calc_out_num_channels(initial_num_channels_power, index - 1)
    )

    if pooldegree == None:
        return nn.Sequential(
            convblock_variant(
                kernel_size,
                padding,
                in_channels,
                out_channels,
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )
    return convblock_variant(
        kernel_size,
        padding,
        in_channels,
        out_channels,
    )


# A convolutional block for a simple network
# The halving is done by the MaxPool module.
def convblock_normal(
    kernel_size,
    padding,
    in_channels,
    out_channels,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        ),
        # num_features is the output channel size
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
    )


# A convolutional block for a mobile network
def convblock_mobile(
    kernel_size,
    padding,
    in_channels,
    out_channels,
):
    # Separable depthwise
    return nn.Sequential(
        # Depthwise convolution
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        ),
        # 1x1 convolution
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        ),
        # num_features is the output channel size
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
    )


# A convolutional block for a mobile network
def convblock_mobile2(
    kernel_size,
    padding,
    in_channels,
    out_channels,
):
    # Separable depthwise
    return nn.Sequential(
        # Depthwise convolution
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        ),
        # num_features is the output channel size
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU(),
        # 1x1 convolution
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        ),
        # num_features is the output channel size
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
    )


def calcParams(
    preserve_size_after_conv_layer,
    pooldegree=None,
    # kernel size is held constant
    kernel_size=3,
):
    if preserve_size_after_conv_layer:
        padding = (kernel_size - 1) // 2
        if pooldegree == None:
            calc_out_num_channels = lambda initial_num_channels_power, index: 2 ** (
                initial_num_channels_power + index
            )
            calc_out_size = lambda initial_size, index: initial_size // 2**index
        else:
            calc_out_num_channels = (
                lambda initial_num_channels_power, _: 2**initial_num_channels_power
            )
            calc_out_size = lambda initial_size, _: initial_size
    else:
        padding = 0
        calc_out_num_channels = lambda initial_num_channels_power, index: 2 ** (
            initial_num_channels_power + index
        )
        if pooldegree == None:
            calc_out_size = lambda initial_size, index: initial_size // 2**index - 2
        else:
            calc_out_size = lambda initial_size, index: initial_size - 2 * index
    return padding, calc_out_size, calc_out_num_channels


def simpleconv(
    self,
    # Tuned parameters are given below
    convblock_variant,
    preserve_size_after_conv_layer,
    num_conv_blocks,
    # Initial number of channels is the number of channels
    # of the first activation layer, the layer coming just after
    # the input. It is 2**initial_num_channels_power for the first
    # implementation, 2**initial_num_channels_power-2 for the second
    initial_num_channels_power,
    pooldegree=None,
    # The parameters that are not tuned (constant) are given below
    # number of channels of image (rgb)
    initial_num_channels=3,
    # image width
    initial_size=32,
    # kernel size is held constant
    kernel_size=3,
):
    # This network is composed of a number of convolutional blocks
    # (definition of which is given below), and
    # a single linear layer.
    # It has 2 variants. Both variants use the same architecture,
    # except the first variant uses zero-padding to preserve the
    # size (width or height) after the coınvolutional layer.
    # half the width (which is equal to the height) of
    # each activation. On the contrary,
    # they double the number of channels with increasing blocks.
    # Therefore, Number of channels and the size (width or height)
    # of the layer with index `index` (zero-indexing is used)
    # is given below in respective functions
    self.pooldegree = pooldegree

    padding, calc_out_size, calc_out_num_channels = calcParams(
        preserve_size_after_conv_layer,
        pooldegree,
    )

    self.convlayers = nn.ModuleList(
        [
            convblock(
                convblock_variant,
                i,
                kernel_size,
                padding,
                initial_num_channels,
                calc_out_num_channels,
                self.pooldegree,
                initial_num_channels_power,
            )
            for i in range(num_conv_blocks)
        ]
    )
    last_num_channels = calc_out_num_channels(
        initial_num_channels_power,
        num_conv_blocks - 1,
    )
    last_activation_size = calc_out_size(
        initial_size,
        num_conv_blocks,
    )

    if self.pooldegree != None:
        self.pool = nn.AvgPool2d(
            kernel_size=self.pooldegree,
            stride=self.pooldegree,
        )
        last_activation_size //= self.pooldegree
    if last_activation_size < 1:
        raise ValueError(
            f"Last activation size {last_activation_size} is smaller than 1"
        )

    self.linear = nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            last_activation_size * last_activation_size * last_num_channels,
            10,
        ),
    )


class ConvNet5(nn.Module):
    """Your custom CNN"""

    def __init__(self, **kwargs):
        super().__init__()

        # START TODO #############
        # achieves 55%
        # simpleconv(self, num_conv_blocks=2, initial_num_channels_power=5)
        simpleconv(self, **kwargs)
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
        x = seqconv(self, x)
        if self.pooldegree != None:
            x = self.pool(x)
        x = self.linear(x)
        # END TODO #############
        return x
