"""
This file contains the combined loss function for the multiplet detection model.
"""
import torch
from typing import Optional

from moldetr.loss.individual_losses import LossPartial
from moldetr.matcher.matcher import MatcherPartial


# def group_matching(
#     outputs: torch.Tensor,
#     targets: torch.Tensor,
#     n_groups: Optional[int],
#     matching_partial: MatcherPartial,
# ):
#     group_outputs = outputs.chunk(n_groups, dim=0)
#     batch_size = outputs.shape[0]
#     number_of_targets = len(targets["targets"])
#     indices = []
#     offset_prediction = 0
#     offset_target = 0
#     for group_output in group_outputs:
#         group_indices = matching_partial(
#             group_output,
#             targets,
#         )
#         # Shift the indices by the offset
#         shifted_indices = [
#             (
#                 i + offset_prediction,
#                 j + offset_target,
#             )
#             for i, j in group_indices
#         ]
#         indices.extend(shifted_indices)
#
#         # Update the offset for the next group
#         offset_prediction += batch_size
#         offset_target += number_of_targets
#
#     return indices


def combined_loss_func(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    parameter_loss_partial: LossPartial,
    classification_loss_partial: LossPartial,
    giou_loss_partial: LossPartial,
    matching_partial: MatcherPartial,
    loss_weighting: Optional[torch.Tensor],
    n_groups: Optional[int],
    reduction: str = "sum",
) -> torch.Tensor:
    """Calculate the combined loss of the model output and target multiplets. This is the (weighted) sum of the parameter loss, giou loss and the classification loss.  The loss is weighted by the loss weighting. The parameter loss is weighted by the parameter weighting. The classification loss is weighted by the class weighting.  The loss is calculated for the matching between the output and the target multiplets. The matching is calculated by the Hungarian algorithm. The loss is calculated for the given parameter indices.
    Args:
        outputs: tensor containing the output of the model
        targets: tensor containing the target of the model
        parameter_loss: loss function for the parameters
        classification_loss: loss function for the classes
        n_classes: number of classes in the data set

        parameter_indices: indices of the parameters in the output
        loss_weighting: weighting of the loss
        parameter_weighting: weighting of the parameters
        class_weighting: weighting of the classes
    Returns: combined loss of the model output and target multiplets
    """

    def group_loss(group, targets, reduction, indices):
        # indices = matching_partial(group, targets)
        return (
            loss_weighting.classification_loss_weighting
            * classification_loss_partial(
                group,
                targets,
                reduction=reduction,
                indices=indices,
            )
            + loss_weighting.parameter_loss_weighting
            * parameter_loss_partial(
                group,
                targets,
                reduction=reduction,
                indices=indices,
            )
            + loss_weighting.giou_loss_weighting
            * giou_loss_partial(
                group,
                targets,
                reduction=reduction,
                indices=indices,
            )
        )

    num_multiplets_in_batch = sum(targets["num_targets"])

    # #
    if reduction in ("sum", "mean"):
        total_loss = torch.tensor(0.0, device=outputs[0].device)

    else:  # reduction == "none":
        total_loss = torch.zeros(outputs.shape[0]*outputs.shape[1], device=outputs[0].device)

    batch_size = len(targets["targets"])


    # check outputs
    # assert (outputs[0, ...] != outputs[1, ...]).all()
    # assert (outputs[0, ...] != outputs[batch_size, ...]).all()
    outputs = [group.squeeze(1) for group in outputs.chunk(n_groups, dim=1)]

    # outputs=outputs.chunk(n_groups, dim=1)
    # outputs=ou.transpose(0,1)
    costs=None
    group_indices = []
    group_costs = []
    for index, group in enumerate(outputs):
        # print(f"group outputs shape: {group.shape}")
        if reduction in ("none"):

            indices,costs = matching_partial(group, targets, return_cost=True)
            group_costs.append(costs)

        else:
            indices = matching_partial(group, targets)
        group_indices.append(indices)
        if reduction in ("sum", "mean"):
            total_loss += (
                group_loss(
                    group,
                    targets,
                    reduction=reduction,
                    indices=indices,
                )
                / num_multiplets_in_batch
            )

        else:  # reduction == "none":
            total_loss[index * batch_size : (index + 1) * batch_size] = group_loss(
                group,
                targets,
                reduction=reduction,
                indices=indices,

            )

    return (
        total_loss / n_groups
        if reduction in ("mean", "sum")
        else (total_loss / n_groups, group_indices, group_costs)
    )
