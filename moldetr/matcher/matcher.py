"""
# Modified from Deformation DETR
# ------------------------------------------------------------------------
# Deformable DETR found at https://github.com/fundamentalvision/Deformable-DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# ------------------------------------------------------------------------
This file contains the matching function which implements a bipartity matching using the Hungarian algorithm.
"""

from typing import Optional, Callable

import torch
from scipy.optimize import linear_sum_assignment

from moldetr.config import CostWeighting
from moldetr.loss.individual_losses import CalculateGIOU

# define types for the matcher function with preapplyed parameters
MatcherPartial = Callable[
    [
        torch.Tensor,
        torch.Tensor,
    ],
    list[tuple[torch.Tensor, torch.Tensor]],
]

# define types for the matcher function without preapplyed parameters
Matcher = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        int,
        Optional[torch.Tensor],
    ],
    list[tuple[torch.Tensor, torch.Tensor]],
]


# TODO: Implement the GIoU cost
def giou_cost(
    out_param: torch.Tensor, tgt_param: torch.Tensor, calculate_giou: CalculateGIOU
) -> torch.Tensor:
    """This function implements the GIoU cost between the output and target multiplets.
    Parameters  ----------
    out_param: tensor containing the output parameters
    tgt_param: tensor containing the target parameters
    Returns: GIoU cost between the output and target multiplets

    """
    out_param_matrix = out_param.unsqueeze(1).repeat(1, tgt_param.size(0), 1)
    # print(f"Size out_param_matrix: {out_param_matrix.size()}")
    tgt_param_matrix = tgt_param.unsqueeze(0).repeat(out_param.size(0), 1, 1)
    # print(f"Size tgt_param_matrix: {tgt_param_matrix.size()}")
    giou = calculate_giou(out_param_matrix, tgt_param_matrix)
    # print(f"Size giou_cost: {giou_cost.size()}")
    return 1 - giou


def matching(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    calculate_giou: CalculateGIOU,
    n_classes: int,
    cost_weighting: CostWeighting,
    parameter_cost_weights: torch.Tensor,
    return_cost: bool = False,
) -> list[tuple[torch.tensor, torch.tensor]]:
    """This function implements a bipartity matching using the Hungarian algorithm.
    Parameters  ----------
    outputs: tensor containing the output of the model
    targets: tensor containing the target of the model
    n_classes: number of classes in the data set
    loss_weigthing: tensor containing the loss weighting for each class

    Returns: list of tuples containing the indices of the inputs and targets to be permuted
    -------

    """

    # Assuming `outputs` is the variable holding your model's output just before matching
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"
    assert not torch.isinf(outputs).any(), "Model output contains Infs"

    with torch.no_grad():
        bs, num_queries = outputs.shape[:2]
        # We flatten to compute the cost matrices in a batch
        out_class_prob = outputs[..., :n_classes].flatten(0, 1).sigmoid()
        out_param = outputs[..., n_classes:].flatten(0, 1)

        # Also concat the target labels and boxes

        tgt = torch.cat([target for target in targets["targets"]], dim=0)
        tgt_class = tgt[..., 0].long()
        tgt_param = tgt[..., 1:]

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (
            (1 - alpha)
            * (out_class_prob * gamma)
            * (-(1 - out_class_prob + 1e-8).log())
        )
        pos_cost_class = (
            alpha * ((1 - out_class_prob) ** gamma) * (-(out_class_prob + 1e-8).log())
        )
        cost_class = pos_cost_class[:, tgt_class] - neg_cost_class[:, tgt_class]


        # Compute the GIoU cost between boxes
        cost_giou = giou_cost(out_param, tgt_param, calculate_giou)



        # Compute the L1 cost between boxes
        if parameter_cost_weights is not None:
            cost_param = torch.cdist(
                parameter_cost_weights * out_param,
                parameter_cost_weights * tgt_param,
                p=1,
            ) / tgt_param.size(1)
        else:
            cost_param = torch.cdist(out_param, tgt_param, p=1) / tgt_param.size(1)





    # Final cost matrix
    if cost_weighting is not None:
        cost_param = cost_param * cost_weighting.parameter_cost_weighting
        cost_giou = cost_giou * cost_weighting.giou_cost_weighting

    with open("cost.txt", "w") as f:
        f.write(
            "%s = %.2f, %s = %.2f, %s = %.2f \n"
            % (
                "cost_class",
                float(cost_class.mean()),
                "cost_param",
                float(cost_param.mean()),
                "cost_giou",
                float(cost_giou.mean()),
            ),
        )

    # Compute the final cost matrix
    C = cost_class + cost_param + cost_giou


    C = C.view(bs, num_queries, -1).cpu()

    # Proceed with the matching as usual
    sizes = targets["num_targets"]
    indices = [
        linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
    ]
    indices = [
        (
            torch.as_tensor(i, dtype=torch.int64),
            torch.as_tensor(j, dtype=torch.int64),
        )
        for num, (i, j) in enumerate(indices)
    ]

    if return_cost:
        return indices, {"cost": C, "cost_class": cost_class, "cost_param": cost_param, "cost_giou": cost_giou}
    else:
        return indices
