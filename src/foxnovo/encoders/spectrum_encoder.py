import torch

from foxnovo.encoders.base_encoders import PeakEncoder, TokenizerEncoder


class SpectrumEncoder(torch.nn.Module):
    def __init__(
        self,
        dim_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: bool = True,
        dim_intensity: int | None = None,
        peaks_max_int: int = 3000,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.peaks_max_int = peaks_max_int
        self.tokenizer_encoder = TokenizerEncoder(
            dim_model=self.dim_model, peaks_max_int=self.peaks_max_int, padding_idx=0
        )
        if peak_encoder:
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))
        torch.nn.Linear(dim_model * 4, dim_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )
        self.tokenizer_encoder.input_proj = torch.nn.Linear(4 * dim_model, dim_model)

    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        int_token_embedding, dec_token_embedding = self.tokenizer_encoder(spectra)
        zeros = ~spectra.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)
        peaks = self.peak_encoder(spectra)
        x1 = torch.cat([peaks, int_token_embedding], dim=-1)
        x2 = torch.cat([peaks, dec_token_embedding], dim=-1)
        fused_spectra = torch.cat([x1, x2], dim=-1)
        peaks = self.tokenizer_encoder.input_proj(fused_spectra)
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)
        peaks = torch.cat([latent_spectra, peaks], dim=1)
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask

    @property
    def device(self):
        return next(self.parameters()).device
