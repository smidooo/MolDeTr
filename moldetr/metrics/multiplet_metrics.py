"""This file contains the metrics for the multiple object detection task."""

import fastai.metrics  # noqa: F401 — accessed as fastai.metrics.accuracy_multi / .accuracy below
import torch
from typing import Callable, Optional

from moldetr.loss.helper_funcs import (
    get_src_permutation_idx,
)
from moldetr.loss.individual_losses import LossPartial
from moldetr.matcher.matcher import MatcherPartial

# create a callable type for the metrics
MetricPartial = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def accuracy_with_empty_object(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    matching_partial: MatcherPartial,
    number_of_classes: int,
    n_groups: int = 1,
) -> torch.Tensor:
    """Calculate the accuracy of the model with the empty object class.
    Args:
    outputs: tensor containing the output of the model
    targets: tensor containing the target of the model
    number_of_classes: number of classes in the data set
    Returns:
    Metric: accuracy of the model with the empty object class"""

    outputs = [group.squeeze(1) for group in outputs.chunk(n_groups, dim=1)]
    targets_classes_onehot = []
    output_classes = []
    for output in outputs:
        indices = matching_partial(output, targets)
        output_index = get_src_permutation_idx(indices)

        targets_permuted = torch.cat([t[J] for t, (_, J) in zip(targets["targets"], indices)])
        output_class = output[..., :number_of_classes]
        target_class_o = targets_permuted[..., 0].to(dtype=torch.int64)

        target_classes = torch.full(
            output_class.shape[:2],
            number_of_classes,
            dtype=torch.int64,
            device=target_class_o.device,
        )

        target_classes[output_index] = target_class_o
        target_classes_onehot = torch.zeros(
            [output_class.shape[0], output_class.shape[1], output_class.shape[2] + 1],
            dtype=output_class.dtype,
            layout=output_class.layout,
            device=output_class.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        targets_classes_onehot.append(target_classes_onehot)
        output_classes.append(output_class)
    output_class = torch.cat(output_classes)
    target_classes_onehot = torch.cat(targets_classes_onehot)
    return fastai.metrics.accuracy_multi(output_class, target_classes_onehot)


def accuracy_without_empty_object(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    matching_partial: MatcherPartial,
    number_of_classes: int,
    n_groups: int = 1,
) -> torch.Tensor:
    """Calculate the accuracy of the model without the empty object class.
    Args:
    outputs: tensor containing the output of the model
    targets: tensor containing the target of the model
    number_of_classes: number of classes in the data set
    Returns:
    Metric: accuracy of the model without the empty object class"""
    outputs = [group.squeeze(1) for group in outputs.chunk(n_groups, dim=1)]
    target_classes = []
    output_classes = []
    for output in outputs:
        indices = matching_partial(output, targets)
        output_index = get_src_permutation_idx(indices)

        targets_permuted = torch.cat([t[J] for t, (_, J) in zip(targets["targets"], indices)])
        target_class_o = targets_permuted[..., 0].to(dtype=torch.int64)

        output = output[output_index]  # .transpose(1, 2)
        output_class = output[..., :number_of_classes]  # transpose(1, 2)
        target_classes.append(target_class_o)
        output_classes.append(output_class)

    output_class = torch.cat(output_classes)
    target_class_o = torch.cat(target_classes)
    return fastai.metrics.accuracy(output_class, target_class_o)


def parameter_loss_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    matching_partial: MatcherPartial,
    parameter_loss_callable: LossPartial,
    parameter_loss_weighting: Optional[torch.Tensor],
    n_groups: int = 1,
) -> torch.Tensor:
    """Calculate the parameter loss of the model.
    Args:
    callable: callable function
    *args: arguments for the callable function
    **kwargs: keyword arguments for the callable function
    Returns:
    Metric: parameter loss of the model"""
    loss = 0.0
    outputs = [group.squeeze(1) for group in outputs.chunk(n_groups, dim=1)]

    num_multiplets_in_batch = sum(targets["num_targets"])

    if parameter_loss_weighting is None:
        parameter_loss_weighting = 1

    for output in outputs:
        indices = matching_partial(
            output,
            targets,
        )

        loss += parameter_loss_callable(
            output,
            targets,
            indices=indices,
        )
    return parameter_loss_weighting * (loss / (num_multiplets_in_batch * n_groups))


def single_parameter_loss_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    matching_partial: MatcherPartial,
    single_parameter_loss_callable: LossPartial,
    parameter_loss_weighting: Optional[torch.Tensor],
    n_groups: int = 1,
) -> torch.Tensor:
    """Calculate the parameter loss of the model.
    Args:
    callable: callable function
    *args: arguments for the callable function
    **kwargs: keyword arguments for the callable function
    Returns:
    Metric: parameter loss of the model"""
    loss = 0.0
    outputs = [group.squeeze(1) for group in outputs.chunk(n_groups, dim=1)]

    num_multiplets_in_batch = sum(targets["num_targets"])
    if parameter_loss_weighting is None:
        parameter_loss_weighting = 1

    for output in outputs:
        indices = matching_partial(
            output,
            targets,
        )

        loss += single_parameter_loss_callable(
            output,
            targets,
            indices=indices,
        )

    return parameter_loss_weighting * (loss / (num_multiplets_in_batch * n_groups))


def classification_loss_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    matching_partial: MatcherPartial,
    classification_loss_callable: LossPartial,
    classification_loss_weighting: Optional[torch.Tensor],
    n_groups: int = 1,
) -> torch.Tensor:
    """Calculate the classification loss of the model.
    Args:
    callable: callable function
    *args: arguments for the callable function
    **kwargs: keyword arguments for the callable function
    Returns:
    Metric: classification loss of the model"""
    loss = 0.0
    outputs = [group.squeeze(1) for group in outputs.chunk(n_groups, dim=1)]

    num_multiplets_in_batch = sum(targets["num_targets"])
    if classification_loss_weighting is None:
        classification_loss_weighting = 1

    for output in outputs:
        indices = matching_partial(
            output,
            targets,
        )

        loss += classification_loss_callable(
            output,
            targets,
            indices=indices,
        )

    return classification_loss_weighting * (loss / (num_multiplets_in_batch * n_groups))


def giou_loss_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    matching_partial: MatcherPartial,
    giou_loss_callable: LossPartial,
    giou_loss_weighting: Optional[torch.Tensor],
    n_groups: int = 1,
) -> torch.Tensor:
    """Calculate the giou loss of the model.
    Args:
    callable: callable function
    *args: arguments for the callable function
    **kwargs: keyword arguments for the callable function
    Returns:
    Metric: giou loss of the model"""
    loss = 0.0
    outputs = [group.squeeze(1) for group in outputs.chunk(n_groups, dim=1)]

    num_multiplets_in_batch = sum(targets["num_targets"])
    if giou_loss_weighting is None:
        giou_loss_weighting = 1
    for output in outputs:
        indices = matching_partial(
            output,
            targets,
        )

        loss += giou_loss_callable(
            output,
            targets,
            indices=indices,
        )

    return giou_loss_weighting * (loss / (num_multiplets_in_batch * n_groups))


# precision=fastai.metrics.PrecisionMulti(sigmoid=False)
# recall=fastai.metrics.RecallMulti(sigmoid=False)
# AUC=fastai.metrics.RocAucMulti(sigmoid=False)
# APscore=fastai.metrics.APScoreMulti(sigmoid=False)
