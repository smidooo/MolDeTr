"""Base dataclass-module interfaces shared across the model components."""

"""Base dataclass-module interfaces shared across the model components."""

from dataclasses import dataclass
from typing import Protocol
import torch.nn as nn


@dataclass
class DataclassModule(nn.Module):
    def __new__(cls, *args, **k):
        inst = super().__new__(cls)
        nn.Module.__init__(inst)
        return inst


@dataclass
class DataclassProtocolClass(Protocol):
    def __post_init__(self):
        ...

    def forward(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        ...
