import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import neptune
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from fennomix_novo.data_set import hdf_dataloader
from fennomix_novo.depthcharge.components.encoders import FloatEncoder, PositionalEncoder
from fennomix_novo.depthcharge.components.transformers import SpectrumEncoder
from fennomix_novo.scoring import pGlyco_scoring_1206


def get_default_device():
    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"


@dataclass
class Modelconfig:
    train_scratch: bool = True
    peaks_tokenizer: bool = False  # False when using casanovo encoder weight
    "weighted loss parameters"
    use_score_weight: bool = True
    score_mean: float = 2.16
    score_std: float = 0.75
    weight_min: float = 0.5
    weight_max: float = 1.5

    random_seed: int = 454
    n_peaks: int = 150
    min_mz: float = 50.0
    max_mz: int = 2500
    min_intensity: float = 0.01
    remove_precursor_tol: float = 2.0  # Unused
    max_charge: int = 10
    top_k: int = 10  # Unused, for dp_decoder
    top_k_output: int = 10
    precursor_mass_tol: int = 50  # ppm
    min_length: int = 8
    max_length: int = 14
    dim_model: int = 512
    n_head: int = 8
    dim_feedforward: int = 1024
    n_layers: int = 9
    dropout: float = 0.1
    dim_intensity: int | None = None
    residues: dict[str, float] = field(
        default_factory=lambda: {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "C+57.021": 160.030649,
            "L": 113.084064,
            "I": 113.084064,
            "C": 103.009649,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "W": 186.079313,
            "M+15.995": 147.035400,
            "N+0.984": 115.026943,
            "Q+0.984": 129.042594,
            "+42.011": 42.010565,
            "Y+183.035": 346.099,
            "K+183.035": 311.130,
            "-18.011": -18.011,
            "-17.027": -17.027,
            "C+119.004": 222.014,
        }
    )
    n_log: int = 1  # Unused
    tb_summarywriter: str | None = None
    train_label_smoothing: float = 0.01
    model_save_path: str = "/home/chenzx/project/grade0/denovo_sequencing_immunopeptides/trained_model_weight/FeNNetNovo_Task0_baseline.ckpt"
    warmup_iters: int = 100_000
    cosine_schedule_period_iters: int = 600_000
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    train_batch_size: int = 64
    eval_batch_size: int = 1024
    max_epochs: int = 20
    num_sanity_val_steps: int = 0  # Unused Lightning 正式训练前预检
    save_top_k: int = 5  # Unused
    val_check_interval: int = 2500  # Unused
    calculate_precision: bool = False  # Unused: no evaluation mode
    devices: bool | None = None


@dataclass
class Config:
    pipeline: str = "train"
    seed: int = 454
    device, device_str = get_default_device()
    config: Modelconfig = Modelconfig()


def seeding(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("seeding done!!!")


class PeptideMass:
    canonical = {
        "G": 57.021464,
        "A": 71.037114,
        "S": 87.032028,
        "P": 97.052764,
        "V": 99.068414,
        "T": 101.047670,
        "C+57.021": 160.030649,
        "L": 113.084064,
        "I": 113.084064,
        "C": 103.009649,
        "N": 114.042927,
        "D": 115.026943,
        "Q": 128.058578,
        "K": 128.094963,
        "E": 129.042593,
        "M": 131.040485,
        "H": 137.058912,
        "F": 147.068414,
        "R": 156.101111,
        "Y": 163.063329,
        "W": 186.079313,
        "M+15.995": 147.035400,
        "N+0.984": 115.026943,
        "Q+0.984": 129.042594,
        "+42.011": 42.010565,
        "Y+183.035": 346.099,
        "K+183.035": 311.130,
        "-18.011": -18.011,
        "-17.027": -17.027,
        "C+119.004": 222.014,
        "$": 0,
    }
    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues="canonical"):
        if residues == "canonical":
            self.masses = self.canonical
        elif residues == "massivekb":
            self.masses = self.canonical
            self.masses.update(self.massivekb)
        else:
            self.masses = residues

    def __len__(self):
        return len(self.masses)

    def mass(self, seq, charge=None):
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z])", seq)

        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass


"to test: decoder 迭代细化 Iterative Non-Autoregressive, INAR"


class PeptideDecoder(torch.nn.Module):
    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        n_layers=9,
        drop_out=0.0,
        pos_encoder=True,
        residues="canonical",
        max_charge=5,
        max_length=14,
        num_classes=20,
    ):
        super().__init__()
        self.max_length = max_length
        self.num_classes = num_classes
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]  # 不添加<pad>
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}
        self.vocab_size = len(self._amino_acids)
        self.stop_token = self._aa2idx["$"]  # 先定义，decoder时需使用
        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.mass_encoder = FloatEncoder(dim_model)

        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=drop_out,
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(layer, num_layers=n_layers)

        self.final = torch.nn.Linear(dim_model, num_classes + 1)  # 仍然保留+1(<pad>)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, precursors, memory, memory_key_padding_mask=None):
        charges = self.charge_encoder(precursors[:, 1].int() - 1)

        # precursors = masses + charges[:, None, :]
        precursors = charges[:, None, :]  # 去掉mass信息
        tgt = precursors.repeat(1, self.max_length + 1, 1)
        tgt_key_padding_mask = tgt.sum(axis=2) == 0
        tgt = self.pos_encoder(tgt)

        dec_out = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        dec_out = dec_out.transpose(0, 1)
        logits = self.final(dec_out)
        return logits


class dp_decoder:
    def __init__(
        self, residues="canonical", top_k=10, top_k_output=10, eos_idx=31
    ):  # eos_idx: len(amino acids dict),
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]  # 不添加<pad>
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}
        self.top_k = top_k
        self.top_k_output = top_k_output
        self.eos_idx = eos_idx

    def join_no_stoptoken(self, x):
        return "".join([aa for aa in x if aa != "$"])

    def translate_top_k_seqs(self, batch_top_k_seqs):
        vec_map = np.vectorize(self._idx2aa.get)
        seq_array = vec_map(batch_top_k_seqs)
        B, K, _ = seq_array.shape
        aa_seqs = np.empty((B, K), dtype=object)
        for i in range(B):
            for j in range(K):
                aa_seqs[i, j] = self.join_no_stoptoken(seq_array[i, j])
        return aa_seqs

    # @numba.njit
    def find_batch_top_k_seqs(self, batch_aa_prob_table: np.array):
        batch_top_k_scores = np.zeros(
            (batch_aa_prob_table.shape[0], self.top_k_output), dtype=batch_aa_prob_table.dtype
        )  # 每个batch的top_k_output scores
        batch_top_k_seqs = np.zeros(
            (batch_aa_prob_table.shape[0], self.top_k, batch_aa_prob_table.shape[1]), dtype=np.int8
        )  # size：(B, K, L)array

        for i_batch in range(batch_aa_prob_table.shape[0]):
            batch_top_k_seqs[i_batch, :], batch_top_k_scores[i_batch, :] = self.find_top_k_seqs(
                batch_aa_prob_table[i_batch]
            )
        return batch_top_k_seqs, batch_top_k_scores

    # @numba.njit
    def find_top_k_seqs(self, aa_prob_table: np.ndarray):  # size:[L,V]
        top_k_scores = np.zeros(self.top_k, dtype=aa_prob_table.dtype)  # size: K
        top_k_seqs = np.zeros((self.top_k, aa_prob_table.shape[0]), dtype=np.int8)  # size: K,L

        tmp_k_seqs = top_k_seqs.copy()

        top_k_aa_prob_idxes = np.argsort(aa_prob_table[0])[::-1][
            : self.top_k
        ]  # topk aa index in first step
        # step 0
        top_k_scores[:] = aa_prob_table[0, top_k_aa_prob_idxes]
        top_k_seqs[:, 0] = top_k_aa_prob_idxes

        for i_aa in range(1, aa_prob_table.shape[0]):
            top_k_aa_prob_idxes = np.argsort(aa_prob_table[i_aa])[::-1][: self.top_k]
            scores = (
                top_k_scores.reshape(-1, 1)
                @ aa_prob_table[i_aa, top_k_aa_prob_idxes].reshape(1, -1)
            ).reshape(-1)  # top_k[L-1] @ top_k[L]
            top_k_idxes = np.argsort(scores)[::-1][: self.top_k]
            for i, k in enumerate(top_k_idxes):
                rank_i, cur = k // self.top_k, k % self.top_k
                tmp_k_seqs[i, :i_aa] = top_k_seqs[rank_i, :i_aa]
                tmp_k_seqs[i, i_aa] = top_k_aa_prob_idxes[cur]
            top_k_scores[:] = scores[top_k_idxes]
            top_k_seqs[:, : i_aa + 1] = tmp_k_seqs[:, : i_aa + 1]

        return top_k_seqs, top_k_scores

    def find_top_sequence_fenn(self, logits):
        logits[..., 0] = -1e9
        torch.log_softmax(logits, dim=-1).cpu()
        logits_softmax = F.softmax(logits, dim=-1)
        batch_aa_prob_table = logits_softmax.cpu().numpy()
        batch_top_k_seqs, batch_top_k_scores = self.find_batch_top_k_seqs(batch_aa_prob_table)
        final_top_k_seqs = self.translate_top_k_seqs(batch_top_k_seqs)

        # 提取原始logits
        batch_top_k_seqs_cuda = torch.tensor(batch_top_k_seqs, device=logits.device).long()

        # 使用gather函数提取对应位置的logits
        logits_expanded = logits.unsqueeze(1).expand(
            -1, batch_top_k_seqs_cuda.size(1), -1, -1
        )  # [1024, 10, 15, 32]
        indices_expanded = batch_top_k_seqs_cuda.unsqueeze(-1)  # [1024, 10, 15, 1]
        gathered_logits = torch.gather(
            logits_expanded, dim=3, index=indices_expanded
        )  # [1024, 10, 15, 1]
        gathered_logits = gathered_logits.squeeze(-1)  # [1024, 10, 15]

        # 返回每个字符的logits，而不是总和
        return final_top_k_seqs, gathered_logits, batch_top_k_scores


def batch_truncate_after_eos(truth: torch.Tensor, eos_token: int, pad_token: int):
    B, L = truth.size()
    device = truth.device
    eos_mask = truth == eos_token
    eos_exists = eos_mask.any(dim=1)
    eos_pos = torch.full((B,), L - 1, dtype=torch.long, device=device)
    eos_pos[eos_exists] = eos_mask[eos_exists].float().argmax(dim=1)
    truncated_padded = torch.full((B, 15), pad_token, dtype=truth.dtype, device=device)
    for i in range(B):
        end_idx = eos_pos[i].item() + 1
        truncated_padded[i, :end_idx] = truth[i, :end_idx]
    return truncated_padded


def pep_recall_evaluate(pred, truth):
    max_values, max_indices = torch.max(pred, dim=2)
    truncated_pred = batch_truncate_after_eos(max_indices, 31, 0)
    matches = truncated_pred == truth
    num_exact_matches = torch.sum(matches.all(dim=1))
    pep_top1_recall = num_exact_matches.item() / len(pred)
    return pep_top1_recall


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = 0,
        train_label_smoothing: float = 0.01,
        use_score_weight: bool = True,
        score_mean: float = None,
        score_std: float = None,
        weight_min: float = 0.5,
        weight_max: float = 1.5,
        eps: float = 1e-8,
    ):
        """
        Args:
            ignore_index: pad id.
            label_smoothing: label smoothing param passed to F.cross_entropy.
            use_score_weight: whether to enable score-based weighting.
            score_mean, score_std: 全局（dataset-level）score均值与标准差。若 use_score_weight=True，**必须提供**。
            weight_min, weight_max: 最终权重映射区间 [weight_min, weight_max]。
            eps: 防止除零。
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.train_label_smoothing = train_label_smoothing
        self.use_score_weight = use_score_weight
        if self.use_score_weight and (score_mean is None or score_std is None):
            raise ValueError(
                "use_score_weight=True 时必须提供 score_mean 和 score_std（全局统计）"
            )
        self.score_mean = float(score_mean) if score_mean is not None else None
        self.score_std = float(score_std) if score_std is not None else None
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.eps = float(eps)

    def forward(self, pred, truth, score=None, batch_size: int = None):
        """
        pred: (B*L, V)  after reshape
        truth: (B*L,)
        score: (B,)   per-sample score
        """
        loss_per_token = F.cross_entropy(
            pred,
            truth,
            ignore_index=self.ignore_index,
            label_smoothing=self.train_label_smoothing,
            reduction="none",
        )
        if self.use_score_weight and score is not None:
            # z-score归一化
            device = pred.device
            B = batch_size
            L = truth.numel() // B
            loss_per_token = loss_per_token.view(B, L)
            truth_2d = truth.view(B, L)
            mask = (truth_2d != self.ignore_index).float()
            valid_counts = mask.sum(dim=1).clamp(min=1.0)

            # 每个样本的平均loss
            sample_loss = loss_per_token.sum(dim=1) / valid_counts

            # 全局 z-score + sigmod映射
            z = torch.tensor((score - self.score_mean) / (self.score_std + self.eps))
            s = torch.sigmoid(z)
            weights = self.weight_min + s * (self.weight_max - self.weight_min)  # (B,)

            loss = (sample_loss * weights.to(device)).mean()
        else:
            loss = loss_per_token.mean()

        return loss


class Spec2pep(torch.nn.Module):
    def __init__(
        self,
        dim_model=512,
        n_head=8,
        dim_feedforward=1024,
        n_layers=9,
        dropout=0.0,
        dim_intensity=None,
        max_length=14,
        residues="canonical",
        max_charge=5,
        precursor_mass_tol=50,
        min_length=8,
        train_label_smoothing=0.01,
        warmup_iters=100_000,
        cosine_schedule_period_iters=600_000,
        top_k=10,
        top_k_output=10,
        use_score_weight=True,
        score_mean: float = 2.16,
        score_std: float = 0.75,
        weight_min: float = 0.5,
        weight_max: float = 1.5,
    ):
        super().__init__()
        self.residues = residues
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
        )

        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            drop_out=dropout,
            residues=residues,
            max_charge=max_charge,
            max_length=max_length,
            num_classes=len(PeptideMass(residues=residues).masses) + 1,
        )

        self.softmax = torch.nn.Softmax(2)

        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.weighted_celoss = WeightedCrossEntropyLoss(
            ignore_index=0,
            train_label_smoothing=train_label_smoothing,
            use_score_weight=use_score_weight,
            score_mean=score_mean,
            score_std=score_std,
            weight_min=weight_min,
            weight_max=weight_max,
        )
        self.use_score_weight = use_score_weight
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        self.max_length = max_length
        self.precursor_mass_tol = precursor_mass_tol
        self.min_len = min_length
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]  # 不添加<pad>
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}
        self.top_k = top_k
        self.top_k_output = top_k_output
        self.stop_token = self._aa2idx["$"]  # 暂未定义
        self.dp_decoder = dp_decoder(
            residues=residues,
            top_k=self.top_k,
            eos_idx=self.stop_token,
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

        if not partial:
            sequence += ["$"]
        tokens = [self.decoder._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.decoder.device)
        return tokens

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: list[str],
        raw_name: list[str],
        spec_idx: list[str],
        batch_score: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memories, mem_masks = self.encoder(
            spectra, peaks_tokenizer=False, precursors=precursors, peaks_helix=False
        )  # mz tokenizer开关
        "⭐可修改为固定长度，目前为不固定(tgt已经固定为14)"
        # memories = memories[:, :14, :]
        # mem_masks = mem_masks[:, :14]
        logits = self.decoder(
            precursors=precursors, memory=memories, memory_key_padding_mask=mem_masks
        )

        tokens = [self.tokenize(s) for s in sequences]  # list of Tensors
        tokens = [
            F.pad(token, (0, self.max_length + 1 - len(token)), value=0) for token in tokens
        ]  # padding至15：14 + '$'
        tokens = torch.stack(tokens)
        return (logits, tokens.to(logits.device), raw_name, spec_idx, batch_score)

    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor, spec_idx: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memories, mem_masks = self.encoder(
            spectra, peaks_tokenizer=False, precursors=precursors, peaks_helix=False
        )  # mz tokenizer开关
        "⭐可修改为固定长度，目前为不固定(tgt已经固定为14)"
        # memories = memories[:, :14, :]
        # mem_masks = mem_masks[:, :14]
        logits = self.decoder(
            precursors=precursors, memory=memories, memory_key_padding_mask=mem_masks
        )
        return logits, spec_idx

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[str]],
    ) -> torch.Tensor:
        pred, spec_idx = self.forward(*batch)  # pred: L,B,V+1 truth: B,L
        pred = pred.permute(1, 0, 2)
        dp_predict_seq, dp_predict_logits, _ = self.dp_decoder.find_top_sequence_fenn(logits=pred)
        return dp_predict_seq, dp_predict_logits, spec_idx  # 序列、原始logits、spec_idx

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[str]],
        mode: str = "train",
    ) -> torch.Tensor:
        pred, truth, raw_name, spec_idx, batch_score = self._forward_step(
            *batch
        )  # pred: L,B,V+1 truth: B,L
        pred = pred.permute(1, 0, 2)
        B = pred.shape[0]
        peptide_recall = pep_recall_evaluate(pred, truth)
        vocab_size = self.decoder.num_classes
        # batch_size = pred.shape[0]

        pred = pred.reshape(-1, vocab_size + 1)
        truth = truth.flatten()
        if mode == "train" and self.use_score_weight:
            loss = self.weighted_celoss(pred, truth, score=batch_score, batch_size=B)
        elif mode == "train" and not self.use_score_weight:
            loss = self.celoss(pred, truth)
        else:
            loss = self.val_celoss(pred, truth)

        return loss, peptide_recall


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters))
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor


def load_model_weight(new_model: Spec2pep, pretrained_path: str) -> None:
    pretrained_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if "state_dict" in pretrained_state:
        pretrained_state = pretrained_state["state_dict"]

    state = {}
    for key, value in pretrained_state.items():
        state[key] = value
    missing_keys, unexpected_keys = new_model.load_state_dict(state, strict=False)
    print(f"Loaded encoder weights from {pretrained_path}")
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    if "config" in pretrained_state:
        pretrained_dim = pretrained_state["config"]["dim_model"]
        if pretrained_dim != new_model.encoder.dim_model:
            raise ValueError(
                f"Pretrained model dimension ({pretrained_dim}) "
                f"does not match current model ({new_model.encoder.dim_model})"
            )
    return new_model


def load_encoder_weigth(new_model: Spec2pep, pretrained_path: str) -> None:
    pretrained_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if "state_dict" in pretrained_state:
        pretrained_state = pretrained_state["state_dict"]

    encoder_state = {}
    for key, value in pretrained_state.items():
        if key.startswith("encoder."):
            encoder_state[key] = value
    missing_keys, unexpected_keys = new_model.load_state_dict(encoder_state, strict=False)
    # print(f"Loaded encoder weights from {pretrained_path}")
    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")

    if "config" in pretrained_state:
        pretrained_dim = pretrained_state["config"]["dim_model"]
        if pretrained_dim != new_model.encoder.dim_model:
            raise ValueError(
                f"Pretrained model dimension ({pretrained_dim}) "
                f"does not match current model ({new_model.encoder.dim_model})"
            )
    return new_model


class ModelRunner:
    def __init__(
        self,
        config: Modelconfig,  # 变量：类的实例
        model_filename: None,
    ) -> None:
        self.config = config
        self.model_filename = model_filename

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tmp_dir = None
        self.trainer = None
        self.model = None
        self.loaders = None
        self.writer = None

        self.save_top_k = config.save_top_k
        self.model_save_path = Path(self.config.model_save_path)
        self.min_valid_losses = 5.0
        self.max_recall = 0.0

    def __enter__(self):
        """Enter the context manager"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.tmp_dir.cleanup()
        self.tmp_dir = None
        if self.writer is not None:
            self.writer.save()

    def train(
        self,
        train_folder: str,
        val_folder: str,
    ) -> None:
        self.initialize_model()
        run = neptune.init_run(
            project="FennOmix/FeNNetNovo",
            tags=["Task0_baseline"],
            dependencies="infer",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NjJjYjBjNy1kYTgxLTQ0NmEtYjc2Yy1kZmQyY2FiOGVhYjEifQ==",
            monitoring_namespace="monitoring",
            mode="offline",
        )

        self.loaders = hdf_dataloader.DeNovoDataModule(
            train_folder=train_folder,
            val_folder=val_folder,
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size,
            n_peaks=self.config.n_peaks,
            min_mz=self.config.min_mz,
            max_mz=self.config.max_mz,
            min_intensity=self.config.min_intensity,
            remove_precursor_tol=self.config.remove_precursor_tol,
        )
        self.loaders.setup()

        train_loader = self.loaders.get_train_loader()
        valid_loader = self.loaders.get_val_loader()

        num_epochs = self.config.max_epochs
        print("model_save_path:", self.model_save_path)
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            self.model.train()
            train_losses = []
            epoch_loss = 0
            progress_bar = tqdm(train_loader)
            for _step, batch in enumerate(progress_bar):
                batch = tuple(
                    x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                )  # spectra, precursors, peptides, raw_names, spec_idx, score
                loss, peptide_recall = self.model.training_step(batch, mode="train")
                run["train/step_loss"].log(loss)
                run["train/step_pep_top1_recall"].log(peptide_recall)
                run["train/lr"].log(self.optimizer.param_groups[0]["lr"])
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                train_losses.append(loss.item())
                epoch_loss += loss.item()
                progress_bar.set_postfix(
                    train_loss=loss.item(), train_pep_top1_recall=peptide_recall
                )

            epoch_loss /= len(train_loader)
            val_loss, top1_val_peptide_recall = self.validate(valid_loader)

            if top1_val_peptide_recall > self.max_recall:
                self.max_recall = top1_val_peptide_recall
                torch.save(self.model.state_dict(), self.model_save_path)
            print(
                f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, top1_val_peptide_recall = {top1_val_peptide_recall:.2f}, Val Loss = {val_loss:.4f}"
            )
            run["train/epoch_loss"].log(epoch_loss)
            run["val/epoch_loss"].log(val_loss)
            run["val/epoch_pep_top1_recall"].log(top1_val_peptide_recall)

    def validate(self, valid_loader):
        self.model.eval()
        losses = []
        recalls = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = tuple(
                    x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                )
                loss, top1_peptide_recall = self.model.training_step(batch, mode="val")
                losses.append(loss.item())
                recalls.append(top1_peptide_recall)
        return sum(losses) / len(losses), sum(recalls) / len(recalls)

    def predict(self, predict_folder: str, out_put_folder: str):
        folder_path = Path(predict_folder)
        test_files = folder_path.glob("*.hdf5")
        self.initialize_model(mode="predict")
        for test_file_path in test_files:
            "需要删除"
            scored_df_csv_path = out_put_folder + "/" + test_file_path.stem + "with_score.csv"
            scored_filtered_df_csv_path = out_put_folder + "/" + test_file_path.stem + "result.csv"
            file_path = Path(scored_df_csv_path)
            if file_path.exists():
                continue
            print("Process:", test_file_path)
            self.loaders = hdf_dataloader.DeNovoDataModule(
                test_path=test_file_path,
                eval_batch_size=self.config.eval_batch_size,
                n_peaks=self.config.n_peaks,
                min_mz=self.config.min_mz,
                max_mz=self.config.max_mz,
                min_intensity=self.config.min_intensity,
                remove_precursor_tol=self.config.remove_precursor_tol,
                annotated=False,
            )
            self.loaders.setup()
            predict_result = []
            predict_loader = self.loaders.get_test_loader()
            self.model.eval()
            with torch.no_grad():
                for batch in tqdm(predict_loader):
                    batch = tuple(
                        x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                    )
                    batch_result, batch_ori_logits, spec_idx = self.model.predict_step(batch)

                    spec_idx_2d = spec_idx.reshape(-1, 1)  # 形状 (1024, 1)
                    np.apply_along_axis(
                        lambda x: ",".join(
                            [f"{value:.2f}" for value in x]
                        ),  # 格式化每个元素，保留两位小数
                        axis=2,  # 沿第三维操作
                        arr=batch_ori_logits.cpu().numpy(),  # 先将张量转换为 NumPy 数组
                    )  # 不保留logits
                    merged = np.hstack([spec_idx_2d, batch_result])
                    predict_result.append(merged)  # 将当前batch结果添加到列表

            if predict_result:
                all_merged = np.vstack(predict_result)

                # 定义列名（根据实际情况调整，这里只是示例）
                columns = [
                    "spec_idx",
                    "seq1",
                    "seq2",
                    "seq3",
                    "seq4",
                    "seq5",
                    "seq6",
                    "seq7",
                    "seq8",
                    "seq9",
                    "seq10",
                ]
                df = pd.DataFrame(all_merged, columns=columns)
                "在这里打分: input： df， test_file_path: hdf"
                scored_df, filtered_df = pGlyco_scoring_1206.score_sequence(df, test_file_path)
                output_csv_path = out_put_folder + "/" + test_file_path.stem + "ori.csv"
                df.to_csv(output_csv_path, index=False)
                scored_df.to_csv(scored_df_csv_path, index=False)
                filtered_df.to_csv(scored_filtered_df_csv_path, index=False)

    def initialize_model(self, mode=None):
        self.model = Spec2pep(
            dim_model=self.config.dim_model,
            n_head=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            dim_intensity=self.config.dim_intensity,
            max_length=self.config.max_length,
            residues=self.config.residues,
            max_charge=self.config.max_charge,
            precursor_mass_tol=self.config.precursor_mass_tol,
            min_length=self.config.min_length,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=self.config.warmup_iters,
            cosine_schedule_period_iters=self.config.cosine_schedule_period_iters,
            top_k=self.config.top_k,
            top_k_output=self.config.top_k_output,
            use_score_weight=self.config.use_score_weight,
            score_mean=self.config.score_mean,
            score_std=self.config.score_std,
            weight_min=self.config.weight_min,
            weight_max=self.config.weight_max,
        ).to(self.device)

        if mode == "predict":
            self.model = load_model_weight(self.model, self.model_filename)
        else:
            if not self.config.train_scratch:
                self.model = load_encoder_weigth(self.model, self.model_filename)
                self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
                self.lr_scheduler = CosineWarmupScheduler(
                    self.optimizer,
                    self.config.warmup_iters,
                    self.config.cosine_schedule_period_iters,
                )
            else:
                print("Training from scratch...")
                self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
                self.lr_scheduler = CosineWarmupScheduler(
                    self.optimizer,
                    self.config.warmup_iters,
                    self.config.cosine_schedule_period_iters,
                )


def train(
    train_folder: str,
    val_folder: str,
    model: str | None,
) -> None:
    mconfig = Config()
    config = mconfig.config
    with ModelRunner(config, model) as runner:
        print("Training model from:")
        print(f"  {train_folder}")

        print("Validating on:")
        print(f"  {val_folder}")
        runner.train(train_folder, val_folder)
    print("Training Done")


def predict(
    predict_folder: str,
    model: str | None,
    out_put_folder=str,
) -> None:
    mconfig = Config()
    config = mconfig.config
    with ModelRunner(config, model) as runner:
        print("Predicting model from:")
        print(f" {predict_folder}")
        runner.predict(predict_folder, out_put_folder)
    print("Predicting Done")


ori_ne_weight = "C:/czx/Project/Grade0/denovo_sequencing_immunopeptides/trained_model_weight/for_bruker_zheyi_data/FeNNetNovo_dev4_oriencoder_0830_MgfData_based_for_bruker_finetune_150_token.ckpt"
"训练好的model，该版本模型可直接使用, tokenizer关闭"
model_path = "C:/czx/Project/Grade0/denovo_sequencing_immunopeptides/trained_model_weight/for_bruker_zheyi_data/SOTA/FeNNetNovo_Sampler+score2loss+all_mod_no_mass_val_recall_0.83_batch14trained.ckpt"
test_data_path = r"C:/czx/Project/Grade0\denovo_sequencing_immunopeptides/trained_model_weight/test_data/YG480_MSP2401203_A549-30-R_with_seq.mgf"

train_folder = (
    "C:/czx/Project/Grade0/denovo_sequencing_immunopeptides/Data/zheyi_data/batch14/train_dataset"
)
val_folder = (
    "C:/czx/Project/Grade0/denovo_sequencing_immunopeptides/Data/zheyi_data/batch14/val_dataset"
)
predict_folder = (
    r"C:\czx\Project\Grade0\immunopeptides_dataset\zheyi_search_via_IEAtlas_db_test\hdf_denovo"
)
predict_output_folder = (
    r"C:\czx\Project\Grade0\immunopeptides_dataset\zheyi_search_via_IEAtlas_db_test\hdf_denovo\test"
)


if __name__ == "__main__":
    predict(predict_folder=predict_folder, out_put_folder=predict_output_folder, model=model_path)
    # train(train_folder=train_folder,val_folder=val_folder,model=ori_ne_weight)
