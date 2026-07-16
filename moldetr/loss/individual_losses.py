"""Individual loss terms: focal classification, GIoU, and parameter-regression losses."""

"""Individual loss terms: focal classification, GIoU, and parameter-regression losses."""

from typing import Any
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from moldetr.config import RegParamIndices
from moldetr.dataloader.transforms import Transform
from moldetr.loss.helper_funcs import get_src_permutation_idx

LossPartial = Callable[
    [torch.Tensor, torch.Tensor, str],
    torch.Tensor,
]
CalculateGIOU = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def calculate_inter_union(
    output_lines, target_lines
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function calculates the intersection, union and enclosing line of two 1D multipelts.
    Args:
        output_lines: tensor containing the output multiplets
        target_lines: tensor containing the target multiplets
    Returns:    tuple[torch.Tensor, torch.Tensor, torch.Tensor]: intersection, union and enclosing line of the two multiplets

    """
    # assert torch.all(output_lines[0] < output_lines[1])
    # assert torch.all(target_lines[0] < target_lines[1])
    # make sure that the lower bound is always positive or zero

    x_left_max = torch.maximum(output_lines[0], target_lines[0])

    x_right_min = torch.minimum(output_lines[1], target_lines[1])

    inter_line = torch.maximum(x_right_min - x_left_max, torch.zeros_like(x_right_min))

    union_line = (
        output_lines[1]
        - output_lines[0]
        + target_lines[1]
        - target_lines[0]
        - inter_line
    )
    x_left_min = torch.minimum(output_lines[0], target_lines[0])
    x_right_max = torch.maximum(output_lines[1], target_lines[1])
    enclosing_line = torch.maximum(x_right_max - x_left_min, union_line)
    if torch.any(union_line > enclosing_line):
        print(
            f"Elements in union_line are larger than elements in enclosing_line.: {torch.gt(union_line, enclosing_line)}\n, "
            f"union: {union_line[torch.gt(union_line, enclosing_line)]}\n, enclosing: {enclosing_line[torch.gt(union_line, enclosing_line)]}\n, "
            f"intersection: {inter_line[torch.gt(union_line, enclosing_line)]}\n, output_0: {output_lines[0][torch.gt(union_line, enclosing_line)]}\n, "
            f"output_1: {output_lines[1][torch.gt(union_line, enclosing_line)]}\n, target_0: {target_lines[0][torch.gt(union_line, enclosing_line)]}\n, "
            f"target_1: {target_lines[1][torch.gt(union_line, enclosing_line)]}"
        )

    return (inter_line, union_line, enclosing_line)


def calculate_giou(
    outputs_params: torch.Tensor,
    targets_params_permuted: torch.Tensor,
    parameter_indices: RegParamIndices,
    transform: Transform,
    eps: float = 1e-7,
):
    # untransform the bounding box width and center position
    width_bounding_line_pred = transform.untransform(
        outputs_params, parameter_indices["bounding_box_range_in_points"]
    )
    center_position_pred = transform.untransform(
        outputs_params, parameter_indices["center_position_in_points"]
    )

    width_bounding_line_target = transform.untransform(
        targets_params_permuted, parameter_indices["bounding_box_range_in_points"]
    )
    center_position_target = transform.untransform(
        targets_params_permuted, parameter_indices["center_position_in_points"]
    )

    # calculate output lines and target lines
    offset_for_small_objects = 50
    output_lines = (
        center_position_pred - width_bounding_line_pred / 2 - offset_for_small_objects,
        center_position_pred + width_bounding_line_pred / 2 + offset_for_small_objects,
    )
    target_lines = (
        center_position_target
        - width_bounding_line_target / 2
        - offset_for_small_objects,
        center_position_target
        + width_bounding_line_target / 2
        + offset_for_small_objects,
    )

    (
        inter,
        union,
        line_c,
    ) = calculate_inter_union(output_lines, target_lines)
    iou = inter / union
    assert iou.all() >= 0.0
    assert iou.all() <= 1.0

    giou = iou - ((line_c - union) / (line_c + eps))

    return giou


def giou_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    indices: Optional[list[tuple[Any, Any]]],
    calculate_giou: CalculateGIOU,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    This function calculates the generalized intersection over union loss.
    The loss is calculated for each multiplet in the output and target.
    The multiplets are matched using the Hungarian algorithm. The loss is then calculated for each matching multiplet.
    Args:
        outputs: tensor containing the output of the model
        targets: tensor containing the target of the model
        n_classes: number of classes in the data set
        loss_weighting: weighting of the loss
        indices: indices of the matching between the output and the target
        parameter_indices: indices of the parameters in the output and target
        eps: epsilon to avoid division by zero
    Returns: giou loss of the model output and target multiplets for the given indices and parameter indices

    """

    output_index = get_src_permutation_idx(indices)
    outputs = outputs[output_index]
    #   only allow positive bounding box line and center positions
    outputs_params = outputs[..., n_classes:]
    targets_permuted = torch.cat(
        [t[J] for t, (_, J) in zip(targets["targets"], indices)]
    )
    targets_params_permuted = targets_permuted[..., 1:]

    giou = calculate_giou(outputs_params, targets_params_permuted)

    loss = 1 - giou
    if reduction == "none":
        sample_losses = loss.split([len(t) for t in targets["targets"]])
        loss_list = []
        for loss in sample_losses:
            loss_list.append(loss.mean())
        return torch.stack(loss_list)
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError(f"Unknown reduction {reduction}")

def parameter_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    parameter_weighting: Optional[torch.Tensor],
    indices: Optional[list[tuple[Any, Any]]],
    reduction: str = "sum",
) -> torch.Tensor:
    """Calculate the parameter loss. This is the l1 loss of the parameters of the multiplets.
    Args:
        outputs: tensor containing the output of the model
        targets: tensor containing the target of the model
        n_classes: number of classes in the data set
        loss_weighting: weighting of the loss
        parameter_weighting: weighting of the parameters
        indices: indices of the matching between the output and the target
    Returns: l1 loss of the model output and target multiplets for the given indices and parameter indices
    """

    output_index = get_src_permutation_idx(indices)
    output = outputs[output_index]
    targets_permuted = torch.cat(
        [t[J] for t, (_, J) in zip(targets["targets"], indices)]
    )
    output_param = output[..., n_classes:]
    target_param = targets_permuted[..., 1:]

    if reduction == "none":
        loss = weighted_l1_loss(output_param, target_param, parameter_weighting).sum(-1)
        sample_losses = loss.split([len(t) for t in targets["targets"]])
        loss_list = []
        for loss in sample_losses:
            loss_list.append(loss.mean())
        return torch.stack(loss_list)

    elif reduction == "sum":
        return weighted_l1_loss(output_param, target_param, parameter_weighting).sum()
    elif reduction == "mean":
        return weighted_l1_loss(output_param, target_param, parameter_weighting).mean()
    else:
        raise ValueError(
            f"reduction {reduction} is not supported. Please use 'none', 'sum' or 'mean'"
        )

def weighted_l1_loss(
        output: torch.Tensor, target: torch.Tensor, weight
) -> torch.Tensor:
    """Calculate the weighted l1 loss.
    Args:
        output: tensor containing the output of the model
        target: tensor containing the target of the model
        weight: weighting of the loss
    Returns: weighted l1 loss of the model output and target multiplets for the given indices and parameter indices
    """

    return weight * torch.abs(output - target)

def single_parameter_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    parameter_weighting: Optional[torch.Tensor],
    indices: Optional[list[tuple[Any, Any]]],
    reduction: str = "sum",
) -> torch.Tensor:
    """Calculate the parameter loss. This is the l1 loss of the parameters of the multiplets.
    Args:
        outputs: tensor containing the output of the model
        targets: tensor containing the target of the model
        n_classes: number of classes in the data set
        loss_weighting: weighting of the loss
        parameter_weighting: weighting of the parameters
        indices: indices of the matching between the output and the target
    Returns: l1 loss of the model output and target multiplets for the given indices and parameter indices
    """

    output_index = get_src_permutation_idx(indices)
    output = outputs[output_index]
    targets_permuted = torch.cat(
        [t[J] for t, (_, J) in zip(targets["targets"], indices)]
    )
    output_param = output[..., n_classes:]
    target_param = targets_permuted[..., 1:]
    with open("loss.txt", "w") as f:
        f.write(
            "%s = %.2f, %s = %.2f, %s = %.2f, %s = %.2f, %s = %.2f , %s = %.2f, %s = %.2f\n"
            % (
                "position_loss",
                float(
                    parameter_weighting[0]
                    * F.l1_loss(output_param[..., 0], target_param[..., 0])
                ),
                "line_width_loss",
                float(
                    parameter_weighting[1]
                    * F.l1_loss(output_param[..., 1], target_param[..., 1])
                ),
                "bounding_box_range_loss",
                float(
                    parameter_weighting[2]
                    * F.l1_loss(output_param[..., 2], target_param[..., 2])
                ),
                "coupling_constant_1_loss",
                float(
                    parameter_weighting[3]
                    * F.l1_loss(output_param[..., 3], target_param[..., 3])
                ),
                "coupling_constant_2_loss",
                float(
                    parameter_weighting[4]
                    * F.l1_loss(output_param[..., 4], target_param[..., 4])
                ),
                "coupling_constant_3_loss",
                float(
                    parameter_weighting[5]
                    * F.l1_loss(output_param[..., 5], target_param[..., 5])
                ),
                "coupling_constant_4_loss",
                float(
                    parameter_weighting[6]
                    * F.l1_loss(output_param[..., 6], target_param[..., 6])
                ),

            )
        )

    index_ = 0
    return parameter_weighting[index_] * F.l1_loss(
        output_param[..., index_], target_param[..., index_], reduction=reduction
    )

def sigmoid_focal_loss(inputs, targets, class_weights=[30000, 10000, 4400, 600, 600], alpha=0.25, gamma=2,
                       reduction="sum"):
    """
    Modified focal loss function that includes class-specific weighting for targets with an additional dimension,
    with added print statements for debugging tensor dimensions.
    Args:
        inputs: A float tensor of shape [batch_size, number_of_objects, num_classes] containing the logits.
        targets: A float tensor of shape [batch_size, number_of_objects, num_classes] containing the one-hot encoded classes.
        class_weights: List of class weights for inversely proportional weighting.
        alpha: Weighting factor to balance positive vs negative examples.
        gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        reduction: Method for reducing loss over the batch and objects ('none', 'sum', 'mean').
    Returns:
        Loss tensor
    """

    # Convert class weights to a tensor and calculate inversely proportional weights
    class_weights = torch.tensor(class_weights, device=inputs.device, dtype=inputs.dtype)
    class_weights = 1.0 / torch.sqrt(class_weights)
    # class_weights=1.0/class_weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)


    # Sigmoid activation to get probabilities
    prob = inputs.sigmoid()

    # Compute the binary cross-entropy loss without reduction
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")


    # Compute the focal loss components
    p_t = prob * targets + (1 - prob) * (1 - targets)
    focal_loss_factor = ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    # Compute weighted loss
    weighted_loss = ce_loss * focal_loss_factor * alpha_t


    # Apply class-specific weights
    loss = weighted_loss * class_weights.unsqueeze(0).unsqueeze(1)  # Adjust for broadcasting

    if reduction == "none":
        return loss.sum(-1).sum(-1)
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    else:
        raise ValueError("Invalid reduction mode: {}".format(reduction))


def classification_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    indices: Optional[list[tuple[Any, Any]]],
    loss_function: LossPartial,
    reduction: str = "sum",
) -> torch.Tensor:
    """Calculate the classification loss. This is the cross entropy loss of the classes of the multiplets.
    Args:
        outputs: tensor containing the output of the model
        targets: tensor containing the target of the model
        n_classes: number of classes in the data set
        class_weighting: weighting of the classes
        indices: indices of the matching between the output and the target
    Returns: cross entropy loss of the model output and target multiplets for the given indices and parameter indices
    """

    output_index = get_src_permutation_idx(indices)

    targets_permuted = torch.cat(
        [t[J] for t, (_, J) in zip(targets["targets"], indices)]
    )
    output_class = outputs[..., :n_classes]
    target_class_o = targets_permuted[..., 0].to(dtype=torch.int64)

    target_classes = torch.full(
        output_class.shape[:2],
        n_classes,
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
    return (
        loss_function(output_class, target_classes_onehot, reduction=reduction)
        * output_class.shape[1]
    )
