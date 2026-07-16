"""Residual Block class with either batch normalization or instance normalization. """

import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """Residual Block class with either batch normalization or instance normalization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        instance_norm=False,
        stride=1,
        padding=0,
        kernel_size=3,
    ):
        """Initialize the Residual Block class.
        Args:
        initial_in_channels (int): Number of input channels
        initial_out_channels (int): Number of output channels
        instance_norm (bool): Whether to use instance normalization or not, otherwise batch normalization is used
        downsampling (bool): Whether to use downsampling or not
        stride (int): Stride for the convolutional layers
        padding (int): Padding for the convolutional layers
        kernel_size (int): Kernel size for the convolutional layers
        """
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    padding=self.padding,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    bias=True,
                ),
                # nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = None
        if instance_norm:
            self.block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    padding_mode="zeros",
                    stride=1,
                    bias=True,
                ),
                #                 nn.BatchNorm1d(out_channels),
                nn.InstanceNorm1d(out_channels),
                nn.LeakyReLU(),
                #             nn.Dropout(0.03),
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    padding_mode="zeros",
                    stride=self.stride,
                    bias=True,
                ),
                nn.InstanceNorm1d(out_channels)
                #                 nn.BatchNorm1d(out_channels),
                #             nn.Dropout(0.03)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    padding_mode="zeros",
                    stride=1,
                    bias=True,
                ),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),
                #             nn.InstanceNorm1d(out_channels),
                #             nn.Dropout(0.03),
                nn.Conv1d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    padding_mode="zeros",
                    stride=self.stride,
                    bias=True,
                ),
                #             ,nn.InstanceNorm1d(out_channels)
                nn.BatchNorm1d(out_channels),
                #             nn.Dropout(0.03)
            )

    def forward(self, x):
        """Forward pass of the Residual Block
        Args:
        x (torch.Tensor): Input tensor
        Returns:
        torch.Tensor: Output tensor
        """
        if self.stride != 1:
            # return F.avg_pool1d(
            #     F.leaky_relu(
            #         self.block(x) + (x if self.skip is None else self.skip(x))
            #     ),
            #     kernel_size=self.stride,
            #     stride=self.stride,
            # )

            return F.leaky_relu(
                self.block(x) + (x if self.skip is None else self.skip(x))
            )

        return F.leaky_relu(self.block(x) + (x if self.skip is None else self.skip(x)))
