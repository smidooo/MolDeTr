"""Model utilities: parameter embedding, MLP helper, and parameter grouping."""

"""Model utilities: parameter embedding, MLP helper, and parameter grouping."""

from dataclasses import dataclass, field
import copy
import torch
import torch.nn as nn

from moldetr.model.classes_and_interfaces import DataclassModule
from moldetr.model.dnn import MLP


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@dataclass(unsafe_hash=True)
class ParamEmbedding(DataclassModule):
    num_params: int
    hidden_dim: int
    num_decoder_layers: int

    dec_pred_parameter_embed_share: bool = False
    _param_embed: nn.Module = field(default_factory=nn.Module, init=False)
    parameter_embed: nn.ModuleList = field(default_factory=nn.ModuleList, init=False)

    def __post_init__(self):
        self._param_embed = MLP(
            self.hidden_dim, self.hidden_dim, self.num_params, num_layers=3
        )

        nn.init.constant_(self._param_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self._param_embed.layers[-1].bias.data, 0)
        if self.dec_pred_parameter_embed_share:
            self.parameter_embed = nn.ModuleList(
                [
                    copy.deepcopy(self._param_embed)
                    for _ in range(self.num_decoder_layers)
                ]
            )
            nn.init.constant_(self.parameter_embed[0].layers[-1].bias.data[1:], -2.0)
        else:
            nn.init.constant_(self._param_embed.layers[-1].bias.data[1:], -2.0)
            self.parameter_embed = nn.ModuleList(
                [self._param_embed for _ in range(self.num_decoder_layers)]
            )


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_groups(model_without_ddp: nn.Module):
    param_groups = [
        [
            p
            for n, p in model_without_ddp.named_parameters()
            if not match_name_keywords(n, ["backbone"])
            and not match_name_keywords(n, ["reference_points", "sampling_offsets"])
            and p.requires_grad
        ],
        [
            p
            for n, p in model_without_ddp.named_parameters()
            if match_name_keywords(n, ["backbone"]) and p.requires_grad
        ],
        [
            p
            for n, p in model_without_ddp.named_parameters()
            if match_name_keywords(n, ["reference_points", "sampling_offsets"])
            and p.requires_grad
        ],
    ]
    return param_groups
