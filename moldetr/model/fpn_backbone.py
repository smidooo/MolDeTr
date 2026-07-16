"""This module contains the FPN class. """
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from dataclasses import dataclass, field

from moldetr.model.classes_and_interfaces import DataclassModule
from moldetr.model.resnet_block import ResidualBlock


@dataclass(unsafe_hash=True)
class FPN_BB(DataclassModule):
    """This class implements the FPN architecture.
    The FPN architecture is a convolutional neural network that is based on the Feature Pyramid Network architecture and uses a ResNet block as convolutional layer.
    The FPN architecture is used for the prediction of the multiplet parameters and classes.
    The FPN architecture consists of a convolutional and a fully connected layers.
    Args:
        number_of_classes (int): Number of classes.
        kernel_size (int): Kernel size of the convolutional layers.
        num_params (int): Number of regression parameters.
        retrain_seed (int): Seed for the initialization of the fully connected layers.
        num_multiplet_pred (int): Number of multiplets to predict.
        pyramid_layers (int): Number of layers in the pyramid.
    """

    _conv1D: nn.ModuleList = field(default_factory=nn.ModuleList, init=False)
    _conv_x1: nn.ModuleList = field(default_factory=nn.ModuleList, init=False)
    _conv_x3: nn.ModuleList = field(default_factory=nn.ModuleList, init=False)
    _upsample: nn.ModuleList = field(default_factory=nn.ModuleList, init=False)
    _max_pool: nn.ModuleList = field(default_factory=nn.ModuleList, init=False)
    _conv_x1_end: nn.Sequential = field(init=False)
    input_length: int = 4096
    number_of_classes: int = 7
    kernel_size: int = 11
    num_params: int = 5
    retrain_seed: int = 12345
    num_multiplet_pred: int = 30
    pyramid_layers: int = 9
    channel_dim_up: int = 256
    pool_size: int = 8
    cnn_output_dimension: int = 256
    _activation: nn.Module = nn.LeakyReLU(inplace=True)
    _activation_name: str = "leaky_relu"

    def __post_init__(self):
        total_layers = self.pyramid_layers + 1
        kernel_sizes = [3, 5, 7, 9]
        layers_per_kernel_size = total_layers // len(kernel_sizes)

        self._conv1D = nn.ModuleList()
        for i in range(total_layers):
            kernel_index = i // layers_per_kernel_size
            kernel_size = kernel_sizes[min(kernel_index, len(kernel_sizes) - 1)]  # Ensure index is within bounds
            padding = kernel_size // 2
            stride = 2 if i > 0 else 1
            in_channels = 1 if i == 0 else self.channel_dim_up
            out_channels = self.channel_dim_up

            self._conv1D.append(
                ResidualBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
            )
        self._conv_x1 = nn.ModuleList(
            [
                ResidualBlock(
                    self.channel_dim_up,
                    self.channel_dim_up,
                    kernel_size=1,
                    stride=1,
                )
                # nn.Conv1d(2 ** (i + 1), self.channel_dim_up, 1, stride=1)
                for _ in range(self.pyramid_layers + 1)
            ]
        )
        self._conv_x3 = nn.ModuleList(
            [
                ResidualBlock(
                    self.channel_dim_up,
                    self.channel_dim_up,
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                )
                # nn.Conv1d(self.channel_dim_up, self.self.channel_dim_up, kernel_size=3, padding=1)
                for _ in range(self.pyramid_layers + 1)
            ]
        )
        self._upsample = nn.ModuleList(
            [nn.Upsample(scale_factor=2) for _ in range(self.pyramid_layers)]
        )
        self._max_pool = nn.ModuleList(
            [
                nn.AdaptiveMaxPool1d(self.pool_size)
                for _ in range(self.pyramid_layers + 1)
            ]
        )
        self._conv_x1_end = nn.Sequential(
            self._activation,
            nn.Conv1d(
                (int(math.log2(self.input_length // self.pool_size)) + 1)
                * self.channel_dim_up,
                self.cnn_output_dimension,
                1,
            ),
        )
        self.spatial_downsample = nn.ModuleList([])
        for i in range(
            int(
                math.log2(self.input_length // self.pool_size),
            ),
            0,
            -1,
        ):
            self.spatial_downsample.append(
                nn.ModuleList(
                    [
                        ResidualBlock(
                            self.channel_dim_up,
                            self.channel_dim_up,
                            stride=2,
                            kernel_size=5,
                            padding=5 // 2,
                        )
                        for _ in range(i)
                    ]
                )
            )

        self.init_all_weights()

    def init_all_weights(self):
        """Initializes the weights of the FPN architecture."""
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            """Initialize the weights of the Resnet architecture."""
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
            for l in m.children():
                init_weights(l)

        self.apply(init_weights)

    def forward(self, x) -> torch.Tensor:
        """Forward pass of the FPN architecture. The input is a 1D tensor of shape (batch_size, 1, signal_length). The output is a 1D tensor of shape (batch_size, number of prediction objects, number of classes+number of parameters).
        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, 1, signal_length).
        Returns: Output tensor of shape (batch_size, number of prediction objects, number of classes+number of parameters).
        """
        # print(f"input shape: {x.shape}")

        assert (
            x.size()[-1] >= 2**self.pyramid_layers
        ), f"Input is length is to short, must be minimum {2**self.pyramid_layers} but is {x.size()[-1]}"
        C = []
        P = []
        for conv in self._conv1D:
            x = conv(x)
            # print(f"conv block: {x.shape})")
            C.append(x)
        # print(f"last conv block: {x.shape})")
        last_idx = len(C) - 1
        x = self._conv_x1[last_idx](x)

        for i in range(last_idx - 1, -1, -1):
            x = self._conv_x1[i](C[i]) + self._upsample[i](x)

            y = self._conv_x3[i](x)
            P.append(y)

        P = P[-1:-8:-2]

        return P

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
