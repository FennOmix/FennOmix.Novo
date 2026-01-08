"依然依赖于depthcharge"
import torch
from torch import nn
from depthcharge.components.encoders import FloatEncoder, PositionalEncoder
from typing import Optional
"to test: decoder 迭代细化 Iterative Non-Autoregressive, INAR"
class PeptideDecoderHead(nn.Module):
    """
    Deep Learning-based Decoder
    只负责生成概率分布
    """
    def __init__(
            self,
            dim_model: int = 512,
            n_head: int = 8,
            dim_feedforward: int = 1024,
            n_layers: int = 9,
            drop_out: float = 0.0,
            pos_encoder: bool = True,
            max_charge: int = 5,
            max_length: int = 14,
            num_classes: int = 20,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.max_length = max_length
        self.num_classes = num_classes
        self.dim_model = dim_model

        "位置编码"
        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()
        "precursors 编码"
        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.mass_encoder = FloatEncoder(dim_model)

        "Transformer NAR Decoder"
        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=drop_out,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(layer, num_layers=n_layers)
        self.final = torch.nn.Linear(dim_model, num_classes + 1) # +1 for PAD

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
            self,
            precursors: torch.Tensor,
            memory: torch.Tensor,
            memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        forward：生成 logits

        Args:
            precursors: [B, 5] 前体物信息
            memory: [B, L_mem, D] 编码器输出
            memory_key_padding_mask: [B, L_mem] 掩码

        Returns:
            logits: [B, L_tgt, num_classes+1] 原始 logits
        """
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        # precursors = masses + charges[:, None, :]
        precursors = charges[:, None, :]  # 去掉mass信息
        tgt = precursors.repeat(1, self.max_length + 1, 1) # [B, L, D]
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.pos_encoder(tgt)

        dec_out = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        #生成 logits
        logits=self.final(dec_out) # [B, L ,num_classes+1]
        return logits
