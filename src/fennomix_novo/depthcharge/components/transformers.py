import re

import torch

from .. import utils
from ..masses import PeptideMass
from .encoders import FloatEncoder, PeakEncoder, PositionalEncoder

"""Base Transformer models for working with mass spectra and peptides"""
"""添加了peaks_tokenizer，heix编码"""

class SpectrumEncoder(torch.nn.Module):
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
        peaks_tokenizer=True,
        peaks_helix=True,
        peaks_max_int=3000,
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
        # 母离子差值编码器(与原始 m/z 编码器参数一致，保证编码空间统一)
        self.diff_encoder = FloatEncoder(dim_model=dim_model)
        # 特征融合投影层（将“原始峰编码 + 差值编码”拼接后的 2*dim_model 压缩到 dim_model）
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
            x1 = torch.cat([peaks, int_token_embedding], dim=-1)
            x2 = torch.cat([peaks, dec_token_embedding], dim=-1)
            fused_spectra = torch.cat([x1, x2], dim=-1)
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
            precursor_mz_expanded = precursors[:, [0]].expand(
                -1, spectra.shape[1]
            )  # (n_spectra, n_peaks)
            mz_diff = (
                spectra[:, :, 0] - precursor_mz_expanded
            )  # (n_spectra, n_peaks)：负差值对应 b 离子

            # 4. 新增：编码差值（用与原始 m/z 相同的 FloatEncoder）
            diff_encoded = self.diff_encoder(mz_diff)  # (n_spectra, n_peaks, dim_model)

            # 5. 新增：融合原始峰编码和差值编码
            # 拼接：(n_spectra, n_peaks, dim_model) + (n_spectra, n_peaks, dim_model) → (n_spectra, n_peaks, 2*dim_model)
            fused_peaks = torch.cat([peaks_encoded, diff_encoded], dim=-1)
            # 投影压缩到 dim_model：(n_spectra, n_peaks, dim_model)
            fused_peaks = self.fusion_proj(fused_peaks)

            # 6. 拼接全局谱图特征（与原逻辑一致）
            latent_spectra = self.latent_spectrum.expand(
                fused_peaks.shape[0], -1, -1
            )  # (n_spectra, 1, dim_model)
            transformer_input = torch.cat(
                [latent_spectra, fused_peaks], dim=1
            )  # (n_spectra, 1 + n_peaks, dim_model)

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
        return (
            self.transformer_encoder(peaks, src_key_padding_mask=mask),
            mask,
        )  # src_key_padding_mask 参数用于指定输入序列中哪些位置是填充（padding）的，以便在计算注意力时忽略这些位置

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    pos_encoder : bool
        Use positional encodings for the amino acid sequence.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum charge to embed.
    """

    def __init__(
        self,
        dim_model,
        pos_encoder,
        residues,
        max_charge,
    ):
        super().__init__()
        self.reverse = False
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}

        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            dim_model,
            padding_idx=0,
        )

    def tokenize(self, sequence, partial=False):
        """Transform a peptide sequence into tokens

        Parameters
        ----------
        sequence : str
            A peptide sequence.

        Returns
        -------
        torch.Tensor
            The token for each amino acid in the peptide sequence.
        """
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        if self.reverse:
            sequence = list(reversed(sequence))

        if not partial:
            sequence += ["$"]
        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def detokenize(self, tokens):
        """Transform tokens back into a peptide sequence.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_amino_acids,)
            The token for each amino acid in the peptide sequence.

        Returns
        -------
        list of str
            The amino acids in the peptide sequence.
        """
        sequence = [self._idx2aa.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx + 1]

        if self.reverse:
            sequence = list(reversed(sequence))

        return sequence

    @property
    def vocab_size(self):
        """Return the number of amino acids"""
        return len(self._aa2idx)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device


class PeptideEncoder(_PeptideTransformer):
    """A transformer encoder for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
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
    pos_encoder : bool, optional
        Use positional encodings for the amino acid sequence.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        residues="canonical",
        max_charge=5,
    ):
        """Initialize a PeptideEncoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )

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

    def forward(self, sequences, charges):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor of length batch_size
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        charges : torch.Tensor of size (batch_size,)
            The charge state of the peptide

        Returns
        -------
        latent : torch.Tensor of shape (n_sequences, len_sequence, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        sequences = utils.listify(sequences)
        tokens = [self.tokenize(s) for s in sequences]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        encoded = self.aa_encoder(tokens)

        # Encode charges
        charges = self.charge_encoder(charges - 1)[:, None]
        encoded = torch.cat([charges, encoded], dim=1)

        # Create mask
        mask = ~encoded.sum(dim=2).bool()

        # Add positional encodings
        encoded = self.pos_encoder(encoded)

        # Run through the model:
        latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
        return latent, mask


class PeptideDecoder(_PeptideTransformer):
    """A transformer decoder for peptide sequences.

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
    pos_encoder : bool, optional
        Use positional encodings for the amino acid sequence.
    reverse : bool, optional
        Sequence peptides from c-terminus to n-terminus.
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,  # 通过线性变换，先将数据映射到高纬度的空间再映射到低纬度的空间，提取了更深层次的特征
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        reverse=True,
        residues="canonical",
        max_charge=5,
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            residues=residues,
            max_charge=max_charge,
        )
        self.reverse = reverse

        # Additional model components
        self.mass_encoder = FloatEncoder(dim_model)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.final = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)

    def forward(self, sequences, precursors, memory, memory_key_padding_mask):
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        sequences : list of str or list of torch.Tensor
            The partial peptide sequences for which to predict the next
            amino acid. Optionally, these may be the token indices instead
            of a string.
        precursors : torch.Tensor of size (batch_size, 2)
            The measured precursor mass (axis 0) and charge (axis 1) of each
            tandem mass spectrum
        memory : torch.Tensor of shape (batch_size, n_peaks, dim_model)
            The representations from a ``TransformerEncoder``, such as a
           ``SpectrumEncoder``.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of ``memory`` are padding.

        Returns
        -------
        scores : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
            The raw output for the final linear layer. These can be Softmax
            transformed to yield the probability of each amino acid for the
            prediction.
        tokens : torch.Tensor of size (batch_size, len_sequence)
            The input padded tokens.

        """
        # Prepare sequences
        if sequences is not None:
            sequences = utils.listify(sequences)
            tokens = [self.tokenize(s) for s in sequences]
            tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        else:  # predict
            tokens = torch.tensor([[]]).to(self.device)

        # Prepare mass and charge  需要定义一个motif_encoder

        masses = self.mass_encoder(
            precursors[:, None, 0]
        )  # masses.size: 训练 torch.Size([32, 1, 512]) precursors[:, None, 0] torch.Size([32, 1])
        charges = self.charge_encoder(
            precursors[:, 1].int() - 1
        )  # torch.Size([1024, 512]) 训练 torch.Size([32, 512])

        precursors = masses + charges[:, None, :]  # torch.Size([1024, 1, 512]) ori
        # precursors = torch.cat([masses,charges[:,None,:]],dim=1) #需要删除

        # Feed through model: 如果 sequences 为 None，目标输入 tgt 为前体的编码表示。 否则，将前体的编码表示和氨基酸编码后的 tokens 拼接在一起，形成目标输入 tgt。
        if sequences is None:
            tgt = precursors
        else:
            tgt = torch.cat(
                [precursors, self.aa_encoder(tokens)], dim=1
            )  # ori tokens.size:torch.Size([1024, n]),self.aa_encoder(tokens).Size([1024, n, 512]) n为已推得氨基酸个数
        tgt_key_padding_mask = tgt.sum(axis=2) == 0  # 生成填充掩码
        tgt = self.pos_encoder(tgt)  # 位置编码 seq == none：torch.Size([1024, 1, 512])
        tgt_mask = generate_tgt_mask(tgt.shape[1]).to(
            self.device
        )  # 生成目标输入的掩码 tgt_mask，用于防止模型在解码时看到未来的信息。
        preds = self.transformer_decoder(
            tgt=tgt,  # the sequence to the decoder (required). motif需要编码进去
            memory=memory,  #  the sequence from the last layer of the encoder (required)
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )  # preds torch.Size([1024, 1, 512])

        return self.final(
            preds
        ), tokens  # 线性变换，将解码得到的512维度特征向量映射到29维（氨基酸种类）


def generate_tgt_mask(sz):
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    # return torch.triu(  # 上三角矩阵（未来位置设为True）
    #     torch.ones(sz, sz), diagonal=1
    # ).bool()
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)
