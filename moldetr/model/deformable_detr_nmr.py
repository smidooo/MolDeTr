"""This module contains the DETR class. """
import copy
import math
import torch
import torch.nn as nn
from dataclasses import dataclass, field

from moldetr.model.classes_and_interfaces import DataclassModule, DataclassProtocolClass
from moldetr.model.utils import inverse_sigmoid


@dataclass(unsafe_hash=True)
class Deformable_DETR_NMR(DataclassModule):
    """This class implements the DETR architecture for 1D images.
    This DETR uses an FPN as a backbone.
    Args:
        _backbone (nn.Module): Backbone for transformer.
        _transformer (nn.Module): Transformer.
        _class_embed (nn.Module): Class embedding.
        _num_classes (int): Number of classes.
        _num_params (int): Number of regression parameters.
        _num_queries (int): Number of multiplets to predict.
        _network_head (DNN): Network head for transformer.
        hidden_dim (int): Hidden dimension of the transformer.

    """

    backbone: DataclassProtocolClass
    positional_encoding: DataclassProtocolClass
    transformer: DataclassProtocolClass

    parameter_embed: nn.ModuleList
    num_classes: int = 7
    num_params: int = 4
    num_queries: int = 100
    hidden_dim: int = 512
    backbone_output_dim: int = 256
    n_groups: int = 1
    d_model: int = 512
    n_levels: int = 4
    channel_size: int = 256

    dec_pred_class_embed_share: bool = False
    _class_embed: nn.Module = field(default_factory=nn.Module, init=False)

    class_embed: nn.ModuleList = field(default_factory=nn.ModuleList, init=False)

    def __post_init__(self):
        # maybe use assertions for compatibility of components
        self.query_embed = nn.Embedding(
            self.n_groups * self.num_queries, self.hidden_dim * 2
        )

        # self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim * 2)
        self._class_embed = nn.Linear(self.hidden_dim, self.num_classes)

        if self.dec_pred_class_embed_share:
            self.class_embed = nn.ModuleList(
                [
                    copy.deepcopy(self._class_embed)
                    for _ in range(self.transformer.num_decoder_layers)
                ]
            )
        else:
            self.class_embed = nn.ModuleList(
                [self._class_embed for _ in range(self.transformer.num_decoder_layers)]
            )

        # if self.two_stage_add_query_num > 0:
        #     self.init_ref_points(self.two_stage_add_query_num)
        # input projection
        self.input_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                    ),
                    nn.GroupNorm(16, self.hidden_dim),
                )
                # for  in_channels in [1024,512,256,64]
                for in_channels in [self.backbone_output_dim]*4
            ]
        )

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_projections:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._class_embed.bias.data = torch.ones(self.num_classes) * bias_value

    # def init_ref_points(self, use_num_queries):
    #     self.refpoint_embed = nn.Embedding(use_num_queries, 1)
    #
    #     if self.random_refpoints_x:
    #         self.refpoint_embed.weight.data[:, :1].uniform_(0, 1)
    #         self.refpoint_embed.weight.data[:, :1] = inverse_sigmoid(
    #             self.refpoint_embed.weight.data[:, :1]
    #         )
    #         self.refpoint_embed.weight.data[:, :1].requires_grad = False

    def forward_group(
        self,
        group_query,
        srcs=None,
        poss=None,
    ):
        hs, references = self.transformer(srcs, poss, group_query)

        # squeeze parameter predictions into [0,1] range and add offset for box
        # parameter prediction relative to query reference
        output_class_list = []
        outputs_coord_list = []
        for (
            layer_reference,
            layer_parameter_embed,
            layer_cls_embed,
            layer_hs,
        ) in zip(references, self.parameter_embed, self.class_embed, hs):
            # get output class
            output_class = layer_cls_embed(layer_hs)

            # get output parameters
            reference_ = layer_reference
            reference_unsig_ = inverse_sigmoid(reference_)
            tmp = layer_parameter_embed(layer_hs)
            assert reference_unsig_.shape[-1] == 1
            tmp[..., :1] += reference_unsig_
            layer_output = tmp.sigmoid()
            outputs_coord_list.append(layer_output)
            output_class_list.append(output_class)

        outputs_coord_list = torch.stack(outputs_coord_list)
        outputs_class = torch.stack(output_class_list)
        outputs = torch.cat(
            [
                outputs_class[-1],
                outputs_coord_list[-1],
            ],
            dim=-1,
        )
        return outputs

    def forward(self, x):
        # first they do some things to the input. Might be good to adapt the input to be nestedtensor
        # as they say it is faster. However, we do not need a mask (only used for segmentation), so we might not need it

        assert not torch.isnan(x).any(), "Input data contains NaNs"
        assert not torch.isinf(x).any(), "Input data contains Infs"

        bs = x.shape[0]

        srcs = self.backbone(x)

        srcs = [proj(src) for proj, src in zip(self.input_projections, srcs)]
        poss = [
            self.positional_encoding.get_positional_encoding(src.transpose(1, 2))
            for src in srcs
        ]

        group_queries = self.query_embed.weight.reshape(
            self.n_groups, self.num_queries, self.hidden_dim * 2
        )
        # group_queries = self.query_embed.weight.reshape(
        #     self.num_queries, self.hidden_dim * 2
        # )
        outputs = self.forward_group(group_queries, srcs, poss)

        outputs = outputs.view( self.n_groups,bs, self.num_queries, -1)
        outputs=outputs.transpose(0,1)


        # outputs = self.forward_group(self.query_embed.weight, srcs, poss)

        return outputs
