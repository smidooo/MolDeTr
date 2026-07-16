"""Deformable-attention transformer encoder/decoder for 1D NMR spectra."""

"""Deformable-attention transformer encoder/decoder for 1D NMR spectra."""

import copy
from dataclasses import field, dataclass
from typing import Optional

import torch
import torch.nn as nn

from moldetr.model.classes_and_interfaces import DataclassModule
from moldetr.model.ops.modules.ms_deform_attn import MSDeformAttn
from moldetr.model.utils import inverse_sigmoid


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@dataclass(unsafe_hash=True)
class DeformableTransformerEncoderLayer(DataclassModule):
    d_model: int
    n_head: int
    n_levels: int
    n_points: int
    dropout_ratio: float = 0.1
    dim_feedforward: int = 1024
    self_attn: nn.Module = field(default_factory=nn.ModuleList, init=False)
    linear1: nn.Module = field(default_factory=nn.ModuleList, init=False)
    linear2: nn.Module = field(default_factory=nn.ModuleList, init=False)
    norm1: nn.Module = field(default_factory=nn.ModuleList, init=False)
    norm2: nn.Module = field(default_factory=nn.ModuleList, init=False)
    dropout: nn.Module = field(default_factory=nn.ModuleList, init=False)
    dropout1: nn.Module = field(default_factory=nn.ModuleList, init=False)
    dropout2: nn.Module = field(default_factory=nn.ModuleList, init=False)
    activation: nn.Module = field(default_factory=nn.ModuleList, init=False)

    def __post_init__(self):
        # self.self_attn = nn.MultiheadAttention(
        #     self.d_model, self.n_head, dropout=self.dropout_ratio
        # )
        self.self_attn = MSDeformAttn(
            self.d_model, self.n_levels, self.n_head, self.n_points
        )
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.dropout = nn.Dropout(self.dropout_ratio)
        self.dropout1 = nn.Dropout(self.dropout_ratio)
        self.dropout2 = nn.Dropout(self.dropout_ratio)

        self.activation = nn.ReLU()

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,  # for tgt
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


@dataclass(unsafe_hash=True)
class DeformableTransformerDecoderLayer(DataclassModule):
    d_model: int
    n_head: int
    n_levels: int
    n_points: int
    dropout_ratio: float = 0.1
    dim_feedforward: int = 1024

    self_attn: nn.Module = field(default_factory=nn.ModuleList, init=False)
    multihead_attn: nn.Module = field(default_factory=nn.ModuleList, init=False)
    linear1: nn.Module = field(default_factory=nn.ModuleList, init=False)
    linear2: nn.Module = field(default_factory=nn.ModuleList, init=False)
    norm1: nn.Module = field(default_factory=nn.ModuleList, init=False)
    norm2: nn.Module = field(default_factory=nn.ModuleList, init=False)
    norm3: nn.Module = field(default_factory=nn.ModuleList, init=False)
    dropout: nn.Module = field(default_factory=nn.ModuleList, init=False)
    dropout1: nn.Module = field(default_factory=nn.ModuleList, init=False)
    dropout2: nn.Module = field(default_factory=nn.ModuleList, init=False)
    dropout3: nn.Module = field(default_factory=nn.ModuleList, init=False)
    activation: nn.Module = field(default_factory=nn.ModuleList, init=False)

    def __post_init__(self):
        self.self_attn = nn.MultiheadAttention(
            self.d_model, self.n_head, dropout=self.dropout_ratio
        )
        self.cross_attn = MSDeformAttn(
            self.d_model, self.n_levels, self.n_head, self.n_points
        )
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.dropout1 = nn.Dropout(self.dropout_ratio)
        self.dropout2 = nn.Dropout(self.dropout_ratio)
        self.dropout3 = nn.Dropout(self.dropout_ratio)

        self.activation = nn.ReLU()

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt: Optional[torch.Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[torch.Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[torch.Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_reference_points: Optional[torch.Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[torch.Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        memory_level_start_index: Optional[torch.Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[torch.Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[torch.Tensor] = None,  # pos for memory
    ):
        # self attention
        # q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(
            # q.transpose(0, 1),
            # k.transpose(0, 1),

            tgt.transpose(0, 1),
            tgt.transpose(0, 1),
            tgt.transpose(0, 1),
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)  # Add & norm
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos),
            tgt_reference_points,
            memory,
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)  # add & norm
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


@dataclass(unsafe_hash=True)
class DeformableTransformerEncoder(DataclassModule):
    encoder_layer: nn.Module
    num_layers: int

    def __post_init__(self):
        self.layers = _get_clones(self.encoder_layer, self.num_layers)

        # self.norm is used in detr, not sure if necessary

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (W_,) in enumerate(spatial_shapes):
            ref_x = torch.linspace(
                0.5, W_ - 0.5, W_, dtype=torch.float32, device=device
            )

            ref_x = ref_x.reshape(-1) / (W_)

            reference_points_list.append(ref_x)
        reference_points = torch.cat(reference_points_list, 0)
        reference_points = reference_points[:, None]
        return reference_points

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]
            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """
        output = src
        # preparation and reshape

        reference_points = self.get_reference_points(spatial_shapes, device=src.device)[
            None, :, None, :
        ].repeat(src.shape[0], 1, spatial_shapes.shape[0], 1)

        for _, layer in enumerate(self.layers):
            # main process

            output = layer(
                src=output,
                pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@dataclass(unsafe_hash=True)
class DeformableTransformerDecoder(DataclassModule):
    decoder_layer: nn.Module
    num_layers: int
    d_model: int
    norm: nn.Module
    param_embed: nn.ModuleList
    spin_attention: nn.Module
    query_dim: int = 1
    query_scale: Optional[float] = None

    def __post_init__(self):
        self.layers = _get_clones(self.decoder_layer, self.num_layers)
        self.spin_attention_layers = _get_clones(self.spin_attention, self.num_layers)

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[torch.Tensor],
        refpoints_unsigmoid: Optional[torch.Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[torch.Tensor] = None,  # num_levels
        spatial_shapes: Optional[torch.Tensor] = None,  # bs, num_levels, 2
        query_pos: Optional[torch.Tensor] = None,  # num_queries, bs, 256
    ):
        output = tgt
        reference_points = refpoints_unsigmoid.sigmoid()
        intermediate = []
        intermediate_reference_points = [reference_points]

        for layer_id, (layer,spin_attention_layer) in enumerate(zip(self.layers, self.spin_attention_layers)):


            assert reference_points.shape[-1] == 1
            reference_points_input = reference_points[:, :, None].repeat(
                1, 1, spatial_shapes.shape[1], 1
            )
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                # tgt_query_sine_embed=query_sine_embed,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
            )
            # output=output.permute(1,0,2)
            # output=spin_attention_layer(output)
            # output=output.permute(1,0,2)

            tmp = self.param_embed[layer_id](output)

            delta_unsig = tmp[..., :1]
            outputs_unsig = delta_unsig + inverse_sigmoid(reference_points)
            new_reference_points = tmp
            new_reference_points[..., :1] = outputs_unsig
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()[..., :1]
            intermediate.append(output)
            intermediate_reference_points.append(reference_points)
        return [
            [itm_out for itm_out in intermediate],
            [itm_refpoint for itm_refpoint in intermediate_reference_points],
        ]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@dataclass(unsafe_hash=True)
class DeformableTransformer(DataclassModule):
    param_embed: nn.ModuleList
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout_ratio: float = 0.1
    n_levels: int = 4
    n_points: int = 4
    random_refpoints_x: bool = True
    encoder_layer: nn.Module = field(default_factory=nn.Module, init=False)
    decoder_layer: nn.Module = field(default_factory=nn.Module, init=False)
    encoder: nn.Module = field(default_factory=nn.Module, init=False)
    decoder: nn.Module = field(default_factory=nn.Module, init=False)
    spin_attention:  nn.Module= field(default_factory=nn.Module, init=False)
    two_stage: bool = False

    def __post_init__(self):
        self.encoder_layer = DeformableTransformerEncoderLayer(
            d_model=self.d_model,
            n_head=self.nhead,
            dropout_ratio=self.dropout_ratio,
            dim_feedforward=self.dim_feedforward,
            n_levels=self.n_levels,
            n_points=self.n_points,
        )
        self.encoder = DeformableTransformerEncoder(
            self.encoder_layer, self.num_encoder_layers
        )
        self.decoder_layer = DeformableTransformerDecoderLayer(
            d_model=self.d_model,
            n_head=self.nhead,
            dropout_ratio=self.dropout_ratio,
            dim_feedforward=self.dim_feedforward,
            n_levels=self.n_levels,
            n_points=self.n_points,
        )
        decoder_norm = nn.LayerNorm(self.d_model)

        self.spin_encoder_layer=nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout_ratio)
        self.spin_attention=nn.TransformerEncoder(self.spin_encoder_layer, num_layers=self.num_decoder_layers)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=self.num_decoder_layers,
            d_model=self.d_model,
            norm=decoder_norm,
            param_embed=self.param_embed,
            spin_attention=self.spin_attention
        )



        self.level_embed = nn.Parameter(torch.Tensor(self.n_levels, self.d_model))

        self.reference_points = nn.Linear(self.d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        nn.init.constant_(self.reference_points.bias.data, 0.0)
        if self.n_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    # can add pos embedding here as argument
    def forward(
        self,
        srcs,
        pos_embeds,
        query_embed=None,
    ):
        """
        Input:
            - srcs: List of multi features [bs, ci,  wi]


            - pos_embeds: List of multi pos embeds [bs, ci,  wi]


        """
        # prepare input for encoder
        assert query_embed is not None
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, w = src.shape
            spatial_shape = (w,)
            spatial_shapes.append(spatial_shape)

            src = src.transpose(1, 2)  # bs, w, c

            pos_embed = pos_embed  # bs, w, c
            if self.n_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)

        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{w}, c
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        # level_start_index = torch.cat(
        #     (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        # )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        memory = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
        )

        # prepare input for decoder
        bs, _, c = memory.shape


        query_embed, tgt = torch.split(query_embed, c, dim=-1)
        #
        # # original code for single group
        # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        # reference_points = self.reference_points(query_embed)

        # Prepare group tensors
        n_groups = query_embed.shape[0]

        # Expand tgt tensor
        tgt = tgt.unsqueeze(1).expand(-1, bs, -1, -1).reshape(bs * n_groups, -1, c)

        # Expand query_embed tensor
        query_embed = (
            query_embed.unsqueeze(1)
            .expand(-1, bs, -1, -1)
            .reshape(bs * n_groups, -1, c)
        )

        # Expand reference_points tensor
        reference_points = self.reference_points(query_embed)

        # Expand memory tensor
        memory = (
            memory.unsqueeze(0)
            .expand(n_groups, -1, -1, -1)
            .reshape(bs * n_groups, -1, c)
        )

        hs, inter_references = self.decoder(
            tgt=tgt,
            memory=memory,
            pos=lvl_pos_embed_flatten,
            refpoints_unsigmoid=reference_points,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            query_pos=query_embed,
        )

        # assert (hs[-1][0, ...] != hs[-1][1, ...]).all()
        # assert (hs[-1][0, ...] != hs[-1][bs, ...]).all()

        # hs[-1]=self.spin_attention(hs[-1])

        return (
            hs,
            inter_references,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)




