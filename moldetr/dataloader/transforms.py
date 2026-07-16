"""Input/target normalization transforms for spectra and multiplet parameters."""

import dataclasses
import torch
import typing
from multipledispatch import dispatch


@dataclasses.dataclass()
class Transform(typing.Protocol):
    """Protocol class for transformations."""

    extrema: dict = dataclasses.field(default_factory=dict)

    def transform(self, value: float, index: int) -> float:
        ...

    def untransform(self, value: torch.Tensor | float, index: int) -> torch.Tensor:
        ...


@dataclasses.dataclass()
class Normalize:
    """This class is used to normalize the data."""

    extrema: dict = dataclasses.field(default_factory=dict)

    def transform(self, value: float, index: int) -> float:
        """Returns the normalized value."""
        extremum = list(self.extrema.values())[index]
        return (value - extremum[0]) / (extremum[1] - extremum[0])

    @dispatch(torch.Tensor, int)
    # @overload untransform for torch.Tensor and int
    def untransform(self, value: torch.Tensor, index: int) -> torch.Tensor:
        """Returns the unnormalized value. This method is used for the training set."""
        extremum = list(self.extrema.values())[index]
        return value[..., index] * (extremum[1] - extremum[0]) + extremum[0]

    @dispatch(float, int)
    # @overload untransform for float and int
    def untransform(self, value: float, index: int) -> float:
        """Returns the unnormalized value. This method is used for the test set."""
        extremum = list(self.extrema.values())[index]
        return value * (extremum[1] - extremum[0]) + extremum[0]



