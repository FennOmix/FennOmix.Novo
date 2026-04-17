import math

import einops
import numpy as np
import torch

"""same with depthcharge.components.encoders"""

"""Simple encoders for input into Transformers and the like."""


class FloatEncoder(torch.nn.Module):
    """Encode floating point values using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MassEncoder"""
        super().__init__()

        # Error checking:
        if min_wavelength <= 0:
            raise ValueError("'min_wavelength' must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("'max_wavelength' must be greater than 0.")

        # Get dimensions for equations:
        d_sin = math.ceil(dim_model / 2)
        d_cos = dim_model - d_sin

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, dim_model).float() - d_sin) / (d_cos - 1)
        sin_term = base * (scale**sin_exp)
        cos_term = base * (scale**cos_exp)

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X, **kwargs):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_masses, dim_model)
            The encoded features for the mass spectra.
        """
        sin_mz = torch.sin(X[:, :, None] / self.sin_term)
        cos_mz = torch.cos(X[:, :, None] / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self.sin_term.device


class PeakEncoder(torch.nn.Module):
    """Encode mass spectrum.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    dim_intensity : int, optional
        The number of features to use for intensity. The remaining features
        will be used to encode the m/z values.
    min_wavelength : float, optional
        The minimum wavelength to use.
    max_wavelength : float, optional
        The maximum wavelength to use.
    learned_intensity_encoding : bool, optional
        Use a learned intensity encoding as opposed to a sinusoidal encoding.
        Note that for the sinusoidal encoding, this encoder expects values
        between [0, 1].
    """

    def __init__(
        self,
        dim_model,
        dim_intensity=None,
        min_wavelength=0.001,
        max_wavelength=10000,
        learned_intensity_encoding=True,
    ):
        """Initialize the MzEncoder"""
        super().__init__()
        self.dim_model = dim_model
        self.dim_mz = dim_model
        self.learned_intensity_encoding = learned_intensity_encoding
        if dim_intensity is not None:
            if dim_intensity >= dim_model:
                raise ValueError("'dim_intensity' must be less than 'dim_model'")

            self.dim_mz -= dim_intensity
            self.dim_intensity = dim_intensity
        else:
            self.dim_intensity = dim_model

        self.mz_encoder = FloatEncoder(
            dim_model=self.dim_mz,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

        if self.learned_intensity_encoding:
            self.int_encoder = torch.nn.Linear(1, self.dim_intensity, bias=False)
        else:
            self.int_encoder = FloatEncoder(
                dim_model=self.dim_intensity,
                min_wavelength=1e-6,
                max_wavelength=1,
            )

    def forward(self, X, **kwargs):
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectra, n_peaks, dim_model)
            The encoded features for the mass spectra.
        """
        m_over_z = X[:, :, 0]
        encoded = self.mz_encoder(m_over_z)

        int_input = X[:, :, [1]] if self.learned_intensity_encoding else X[:, :, 1]

        intensity = self.int_encoder(int_input)
        if self.dim_intensity == self.dim_model:
            return encoded + intensity

        return torch.cat([encoded, intensity], dim=2)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class PositionalEncoder(FloatEncoder):
    """The positional encoder for sequences.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float, optional
        The shortest wavelength in the geometric progression.
    max_wavelength : float, optional
        The longest wavelength in the geometric progression.
    """

    def __init__(self, dim_model, min_wavelength=1, max_wavelength=10000):
        """Initialize the MzEncoder"""
        super().__init__(
            dim_model=dim_model,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

    def forward(self, X, **kwargs):
        """Encode positions in a sequence.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_sequence, n_features)
            The first dimension should be the batch size (i.e. each is one
            peptide) and the second dimension should be the sequence (i.e.
            each should be an amino acid representation).

        Returns
        -------
        torch.Tensor of shape (batch_size, n_sequence, n_features)
            The encoded features for the mass spectra.
        """
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X


class TokenizerEncoder(torch.nn.Module):
    def __init__(self, dim_model: int = 512, peaks_max_int: int = 3000, padding_idx: int = 0):
        super().__init__()
        self.dim_model = dim_model
        self.peaks_max_int = peaks_max_int
        self.padding_idx = padding_idx
        self.mz_token = list(range(0, 1000))
        self.mz_token.extend(list(range(1000, self.peaks_max_int * 1000 + 1, 1000)))
        mz2idx_map = torch.full((max(self.mz_token) + 1,), fill_value=padding_idx, dtype=torch.long)
        for idx, val in enumerate(self.mz_token):
            mz2idx_map[val] = idx
        self.register_buffer("token_lookup", mz2idx_map)
        vocab_size = len(self.mz_token)
        self.int_mz_embedding = torch.nn.Embedding(vocab_size, dim_model, padding_idx=padding_idx)
        self.dec_mz_embedding = torch.nn.Embedding(1000, dim_model, padding_idx=padding_idx)
        self.input_proj = torch.nn.Linear(dim_model * 4, dim_model)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def peaks2token(self, spectra) -> tuple[torch.Tensor, torch.Tensor]:
        spectra_mz = spectra[:, :, 0].squeeze(-1)
        int_part = torch.floor(spectra_mz) * 1000
        dec_part = torch.clamp(torch.round((spectra_mz - torch.floor(spectra_mz)) * 1000), max=999)
        int_part = int_part.clamp(0, self.token_lookup.size(0) - 1).long()
        dec_part = dec_part.clamp(0, self.token_lookup.size(0) - 1).long()
        int_token = self.token_lookup[int_part]
        dec_token = self.token_lookup[dec_part]
        int_token_embedding = self.int_mz_embedding(int_token)
        dec_token_embedding = self.dec_mz_embedding(dec_token)
        return int_token_embedding, dec_token_embedding

    def forward(self, spectra: torch.Tensor, **kwargs) -> torch.Tensor:
        int_token_embedding, dec_token_embedding = self.peaks2token(spectra)
        return int_token_embedding, dec_token_embedding
