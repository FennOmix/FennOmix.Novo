"Based on Depthcharge"

import torch
import torch.nn as nn
from .base_encoders import FloatEncoder, PeakEncoder
from typing import Dict, Type, Optional


class PeakEncodeStrategy(nn.Module):
    """所有峰编码策略的基类，定义统一接口"""
    def __init__(self, peak_encoder: nn.Module):
        super().__init__()
        self.peak_encoder = peak_encoder

    def forward(
            self,
            spectra: torch.Tensor,
            precursors: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
        raise NotImplementedError("子类必须实现forward方法")


class BasePeakEncodeStrategy(PeakEncodeStrategy):
    def forward(self, spectra: torch.Tensor, precursors: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.peak_encoder(spectra)  # 直接调用原peak_encoder


class PeaksTokenizerOnlyStrategy(PeakEncodeStrategy):
    def __init__(
            self,
            peak_encoder: nn.Module,
            dim_model: int,  # 仅用于新增层初始化
            peaks_max_int: int = 3000):
        super().__init__(peak_encoder)
        self.peaks_max_int = peaks_max_int

        self.mz_token = list(range(0, 1000))
        self.mz_token.extend(range(1000, self.peaks_max_int * 1000 + 1, 1000))
        mz2idx_map = torch.full((max(self.mz_token) + 1,), fill_value=0, dtype=torch.long)
        for idx, val in enumerate(self.mz_token):
            mz2idx_map[val] = idx
        self.register_buffer("token_lookup", mz2idx_map)
        vocab_size = len(self.mz_token)
        # int_mz_embedding/dec_mz_embedding/input_proj
        self.int_mz_embedding = torch.nn.Embedding(vocab_size, dim_model, padding_idx=0)
        self.dec_mz_embedding = torch.nn.Embedding(1000, dim_model, padding_idx=0)
        self.input_proj = torch.nn.Linear(4 * dim_model, dim_model)
    def peaks2token(self, spectra:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectra_mz = spectra[:, :, 0].squeeze(-1)
        int_part = torch.floor(spectra_mz) * 1000
        dec_part = torch.clamp(torch.round((spectra_mz - torch.floor(spectra_mz)) * 1000), max=999)
        int_part = int_part.clamp(0, self.token_lookup.size(0) - 1).long()
        dec_part = dec_part.clamp(0, self.token_lookup.size(0) - 1).long()
        int_token = self.token_lookup[int_part]
        dec_token = self.token_lookup[dec_part]
        return self.int_mz_embedding(int_token), self.dec_mz_embedding(dec_token)

    def forward(self, spectra: torch.Tensor, precursors: Optional[torch.Tensor] = None) -> torch.Tensor:
        int_emb, dec_emb = self.peaks2token(spectra)
        return self.input_proj(torch.cat([self.peak_encoder(spectra), int_emb, self.peak_encoder(spectra), dec_emb], dim=-1))

# 3. Peaks Tokenizer + Base模式：
class PeaksTokenizerWithBaseStrategy(PeakEncodeStrategy):
    def __init__(self, peak_encoder: nn.Module, dim_model: int, peaks_max_int: int = 3000):
        super().__init__(peak_encoder)
        self.tokenizer_strategy = PeaksTokenizerOnlyStrategy(peak_encoder, dim_model, peaks_max_int)
        self.input_proj = torch.nn.Linear(4 * dim_model, dim_model)

    def forward(self, spectra: torch.Tensor, precursors: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_feat = self.peak_encoder(spectra)
        int_emb, dec_emb = self.tokenizer_strategy.peaks2token(spectra)
        x1 = torch.cat([base_feat, int_emb], dim=-1)
        x2 = torch.cat([base_feat, dec_emb], dim=-1)
        fused = torch.cat([x1, x2], dim=-1)
        return self.input_proj(fused)

# 4. Helix+Base模式：diff_encoder/fusion_proj
class HelixWithBaseStrategy(PeakEncodeStrategy):
    def __init__(self, peak_encoder: nn.Module, dim_model: int):
        super().__init__(peak_encoder)
        self.diff_encoder = FloatEncoder(dim_model=dim_model)
        self.fusion_proj = torch.nn.Linear(2 * dim_model, dim_model)

    def forward(self, spectra: torch.Tensor, precursors: Optional[torch.Tensor] = None) -> torch.Tensor:
        if precursors is None:
            raise ValueError("Precursors are necessary in Helix Encoder Model")
        base_feat = self.peak_encoder(spectra)
        precursor_mz_expanded = precursors[:, [0]].expand(-1, spectra.shape[1])
        mz_diff = spectra[:, :, 0] - precursor_mz_expanded
        diff_feat = self.diff_encoder(mz_diff)
        fused = torch.cat([base_feat, diff_feat], dim=-1)
        return self.fusion_proj(fused)

class PeakEncodeStrategyRegistry:
    _strategies: Dict[str, Type[PeakEncodeStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[PeakEncodeStrategy]):
        if not issubclass(strategy_cls, PeakEncodeStrategy):
            raise TypeError(f"{strategy_cls} 必须继承自 PeakEncodeStrategy")
        cls._strategies[name] = strategy_cls

    @classmethod
    def get_strategy(cls, name: str) -> Type[PeakEncodeStrategy]:
        if name not in cls._strategies:
            raise ValueError(f"未知策略: {name}，可用策略: {list(cls._strategies.keys())}")
        return cls._strategies[name]

PeakEncodeStrategyRegistry.register("base", BasePeakEncodeStrategy)
PeakEncodeStrategyRegistry.register("peaks_tokenizer_only", PeaksTokenizerOnlyStrategy)
PeakEncodeStrategyRegistry.register("peaks_tokenizer_with_base", PeaksTokenizerWithBaseStrategy)
PeakEncodeStrategyRegistry.register("helix_with_base", HelixWithBaseStrategy)
class SpectrumEncoder(nn.Module):
    def __init__(
        self,
        dim_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: bool = True,
        dim_intensity: Optional[int] = None,
        encode_strategy: str = "base",
        peaks_max_int: int = 3000,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.encode_strategy_name = encode_strategy
        self.peaks_max_int = peaks_max_int

        if peak_encoder:
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)

        strategy_cls = PeakEncodeStrategyRegistry.get_strategy(encode_strategy)
        if encode_strategy == "base":
            self.encode_strategy = strategy_cls(self.peak_encoder)
        elif encode_strategy in ["peaks_tokenizer_only", "peaks_tokenizer_with_base"]:
            self.encode_strategy = strategy_cls(self.peak_encoder, dim_model, peaks_max_int)
        elif encode_strategy == "helix_with_base":
            self.encode_strategy = strategy_cls(self.peak_encoder, dim_model)

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))

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

    def _generate_mask(self, spectra: torch.Tensor) -> torch.Tensor:
        zeros = ~spectra.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]
        return torch.cat(mask, dim=1)

    def forward(
            self,
            spectra: torch.Tensor,
            precursors: Optional[torch.Tensor] = None) -> tuple[
        torch.Tensor, torch.Tensor]:
        mask = self._generate_mask(spectra)

        peak_feat = self.encode_strategy(spectra, precursors)

        latent_spectra = self.latent_spectrum.expand(peak_feat.shape[0], -1, -1)
        peaks = torch.cat([latent_spectra, peak_feat], dim=1)

        latent = self.transformer_encoder(peaks, src_key_padding_mask=mask)

        return latent, mask

    @classmethod
    def add_new_strategy(cls, name: str, strategy_cls: Type[PeakEncodeStrategy]):
        PeakEncodeStrategyRegistry.register(name, strategy_cls)

    @property
    def device(self):
        return next(self.parameters()).device

#---------------------------------------------------------------------------------------
class SpectrumEncoder_ori(torch.nn.Module):
    "耦合高，简单粗暴，但已跑通"
    """A Transformer encoder for input mass spectra.

    Parameters
    ----------
    dim_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : bool, optional
        Use positional encodings m/z values of each peak.
    dim_intensity: int or None, optional
        The number of features to use for encoding peak intensity.
        The remaining (``dim_model - dim_intensity``) are reserved for
        encoding the m/z value.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        peak_encoder=True,
        dim_intensity=None,
        peaks_tokenizer = True,
        peaks_helix = True,
        peaks_max_int = 3000,
    ):
        """Initialize a SpectrumEncoder"""
        super().__init__()
        "------------------peaks_tokenizer---------------------"
        self.peaks_tokenizer = peaks_tokenizer
        if self.peaks_tokenizer:
            self.peaks_max_int = peaks_max_int

            # 构建 mz_token（整数部分 + 小数部分） → 统一 token space
            self.mz_token = list(range(0, 1000))  # 小数部分 token
            self.mz_token.extend(range(1000, self.peaks_max_int * 1000 + 1, 1000))  # 整数部分 token

            # 构建查找表（token value → token index）
            mz2idx_map = torch.full((max(self.mz_token) + 1,), fill_value=0, dtype=torch.long)
            for idx, val in enumerate(self.mz_token):
                mz2idx_map[val] = idx
            self.register_buffer("token_lookup", mz2idx_map)  # 自动迁移到 device

            # 保存 token 数目
            vocab_size = len(self.mz_token)

            # 嵌入层，注意 padding_idx=0 避免训练时更新填充值
            self.int_mz_embedding = torch.nn.Embedding(vocab_size, dim_model, padding_idx=0)
            self.dec_mz_embedding = torch.nn.Embedding(1000, dim_model, padding_idx=0)

            # 输入投影层，用于融合 peak+token 表达
            self.input_proj = torch.nn.Linear(4 * dim_model, dim_model)
        "------------------peaks_tokenizer---------------------"
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))
        if peak_encoder:
            self.peak_encoder = PeakEncoder(
                dim_model,
                dim_intensity=dim_intensity,
            )
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)

        "--------------------helix---------------------"
        #母离子差值编码器(与原始 m/z 编码器参数一致，保证编码空间统一)
        self.diff_encoder = FloatEncoder(
            dim_model=dim_model
        )
        #特征融合投影层（将“原始峰编码 + 差值编码”拼接后的 2*dim_model 压缩到 dim_model）
        self.fusion_proj = torch.nn.Linear(2 * dim_model, dim_model)
        "---------------------helix---------------------"

        # The Transformer layers:
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

    "------------------peaks_tokenizer---------------------"
    def peaks2token(self, spectra):
        spectra_mz = spectra[:, :, 0].squeeze(-1)  # (B, L)

        # 分别取整数和小数部分
        int_part = torch.floor(spectra_mz) * 1000  # 转为你的 token 范围
        dec_part = torch.clamp(torch.round((spectra_mz - torch.floor(spectra_mz)) * 1000), max=999)

        # 转为 long 并 clip 在有效范围
        int_part = int_part.clamp(0, self.token_lookup.size(0) - 1).long()
        dec_part = dec_part.clamp(0, self.token_lookup.size(0) - 1).long()

        # 快速查找 token index，使用预设查找表（已 register_buffer）
        int_token = self.token_lookup[int_part]
        dec_token = self.token_lookup[dec_part]

        # 送入 embedding 层
        int_token_embedding = self.int_mz_embedding(int_token)
        dec_token_embedding = self.dec_mz_embedding(dec_token)
        return int_token_embedding, dec_token_embedding

    "------------------peaks_tokenizer---------------------"

    def forward(self, spectra, peaks_tokenizer, precursors, peaks_helix):
        """The forward pass.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        "------------------peaks_tokenizer---------------------"
        "mz to int and dec part and cat with spectra > transofrmer encoder"
        "mz > int, dec > token > int_embedding, dec_embedding > cat with casanovo encoded spectra respectively > cat x1, x2 > linear to peaks(4D > 1D)"
        if peaks_tokenizer:
            int_token_embedding, dec_token_embedding = self.peaks2token(spectra)
            zeros = ~spectra.sum(dim=2).bool()
            mask = [
                torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
                zeros,
            ]
            mask = torch.cat(mask, dim=1)
            peaks = self.peak_encoder(spectra)

            # Add the spectrum representation to each input:
            x1 = torch.cat([peaks,int_token_embedding], dim=-1)
            x2 = torch.cat([peaks,dec_token_embedding], dim=-1)
            fused_spectra = torch.cat([x1, x2],dim=-1)
            peaks = self.input_proj(fused_spectra)
            latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)
            peaks = torch.cat([latent_spectra, peaks], dim=1)

        elif peaks_helix:
            # 1. 生成 Padding Mask（与原逻辑一致）
            zeros = ~spectra.sum(dim=2).bool()  # (n_spectra, n_peaks)：True 表示填充峰
            mask = [
                torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),  # 全局特征无填充
                zeros,
            ]
            mask = torch.cat(mask, dim=1)  # (n_spectra, 1 + n_peaks)

            # 2. 原始峰编码（与原逻辑一致）
            peaks_encoded = self.peak_encoder(spectra)  # (n_spectra, n_peaks, dim_model)

            # 3. 新增：计算峰与母离子的差值（谱峰 m/z - 母离子 m/z）
            # 扩展母离子维度：(n_spectra,) → (n_spectra, n_peaks)，与谱峰维度匹配
            precursor_mz_expanded = precursors[:, [0]].expand(-1, spectra.shape[1])  # (n_spectra, n_peaks)
            mz_diff = spectra[:, :, 0] - precursor_mz_expanded  # (n_spectra, n_peaks)：负差值对应 b 离子

            # 4. 新增：编码差值（用与原始 m/z 相同的 FloatEncoder）
            diff_encoded = self.diff_encoder(mz_diff)  # (n_spectra, n_peaks, dim_model)

            # 5. 新增：融合原始峰编码和差值编码
            # 拼接：(n_spectra, n_peaks, dim_model) + (n_spectra, n_peaks, dim_model) → (n_spectra, n_peaks, 2*dim_model)
            fused_peaks = torch.cat([peaks_encoded, diff_encoded], dim=-1)
            # 投影压缩到 dim_model：(n_spectra, n_peaks, dim_model)
            fused_peaks = self.fusion_proj(fused_peaks)

            # 6. 拼接全局谱图特征（与原逻辑一致）
            latent_spectra = self.latent_spectrum.expand(fused_peaks.shape[0], -1, -1)  # (n_spectra, 1, dim_model)
            transformer_input = torch.cat([latent_spectra, fused_peaks], dim=1)  # (n_spectra, 1 + n_peaks, dim_model)

            # 7. Transformer 编码（与原逻辑一致）
            return self.transformer_encoder(transformer_input, src_key_padding_mask=mask), mask

        else:
            zeros = ~spectra.sum(dim=2).bool()
            mask = [
                torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
                zeros,
            ]
            mask = torch.cat(mask, dim=1)
            peaks = self.peak_encoder(spectra)

            # Add the spectrum representation to each input:
            latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

            peaks = torch.cat([latent_spectra, peaks], dim=1)
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask # src_key_padding_mask 参数用于指定输入序列中哪些位置是填充（padding）的，以便在计算注意力时忽略这些位置

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device