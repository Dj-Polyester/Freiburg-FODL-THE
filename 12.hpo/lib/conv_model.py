from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


def get_conv_model(num_filters_per_layer: List[int]) -> nn.Module:
    """
    Builds a deep convolutional model with varying number of convolutional
    layers (and # filters per layer) for MNIST input using pytorch.

    Args:
        num_filters_per_layer (list): List specifying the number of filters for each convolutional layer

    Returns:
        convolutional model with desired architecture

    Note:
        for each element in num_filters_per_layer:
            convolution (conv_kernel_size, num_filters, stride=1, padding="same") (use nn.Conv2d(..))
            relu (use nn.ReLU())
            max_pool(pool_kernel_size) (use nn.MaxPool2d(..))

        flatten layer (already given below)
        linear layer
        log softmax as final activation
    """
    assert (
        len(num_filters_per_layer) > 0
    ), "len(num_filters_per_layer) should be greater than 0"
    pool_kernel_size = 2
    conv_kernel_size = 3

    # OrderedDict is used to keep track of the order of the layers
    layers = OrderedDict()

    # START TODO ################
    num_blocks = len(num_filters_per_layer)
    in_channels = 1
    for i in range(num_blocks):
        out_channels = num_filters_per_layer[i]
        layers[f"conv{i+1}"] = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_size,
            padding="same",
        )
        layers[f"relu{i+1}"] = nn.ReLU()
        layers[f"pool{i+1}"] = nn.MaxPool2d(kernel_size=pool_kernel_size)
        in_channels = out_channels
    conv_in_width = 28
    conv_last_out_channels = in_channels
    # Each block halves the width
    conv_out_width = int(conv_in_width / 2**num_blocks)
    conv_output_size = conv_out_width**2 * conv_last_out_channels
    # END TODO ################

    layers["flatten"] = nn.Flatten()
    layers["linear"] = nn.Linear(conv_output_size, 10)
    layers["log_softmax"] = nn.LogSoftmax(dim=1)

    return nn.Sequential(layers)
