import torch
from torch import nn

from fennomix_novo.encoders.base_encoders import FloatEncoder, PositionalEncoder


class PeptideNARDecoder(nn.Module):
    """
    Deep Learning-based Decoder
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
        nn.Module.__init__(self)
        self.max_length = max_length
        self.num_classes = num_classes
        self.dim_model = dim_model
        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()
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
        self.final = torch.nn.Linear(dim_model, num_classes + 1)  # +1 for PAD

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        memory: torch.Tensor,
        precursors: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        forward: generate logits

        Args:
            precursors: [B, 5] precursor_mass, precursor_charge
            memory: [B, L_mem, D] encoder output
            memory_key_padding_mask: [B, L_mem] mask

        Returns:
            logits: [B, L_tgt, num_classes+1] logits
        """
        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]
        # precursors = charges[:, None, :]  # remove precursor mass, may be useful for tims tof daa
        tgt = precursors.repeat(1, self.max_length + 1, 1)  # [B, L, D]
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.pos_encoder(tgt)

        dec_out = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.final(dec_out)  # [B, L ,num_classes+1]
        return logits
