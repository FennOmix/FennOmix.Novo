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
from fennomix_novo.decoders.dp_decoder import DP_Decoder
from fennomix_novo.decoders.peptide_decoder import PeptideNARDecoder
from fennomix_novo.encoders.spectrum_encoder import SpectrumEncoder
from fennomix_novo.scoring import pGlyco_score


def get_default_device():
    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"


@dataclass
class Modelconfig:
    train_scratch: bool = True
    use_weighted_sample: bool = True  # useful for long tail data
    use_weighted_score: bool = True  # useful for low quality data
    score_mean: float = 2.16
    score_std: float = 0.75
    weight_min: float = 0.5
    weight_max: float = 1.5

    random_seed: int = 454
    n_peaks: int = 150
    min_mz: float = 50.0
    max_mz: int = 2500
    min_intensity: float = 0.01
    remove_precursor_tol: float = 2.0
    max_charge: int = 10
    top_k: int = 10
    top_k_output: int = 10
    precursor_mass_tol: int = 2  # Da
    precursor_mass_ppm: int = 20  # ppm
    fragment_mass_ppm: int = 20  # ppm
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
            "C+119.004": 222.014,
        }
    )
    train_label_smoothing: float = 0.01
    model_save_path: str = ""
    warmup_iters: int = 100_000
    cosine_schedule_period_iters: int = 600_000
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    train_batch_size: int = 64
    eval_batch_size: int = 100
    max_epochs: int = 20
    devices: bool | None = None


@dataclass
class Config:
    device, device_str = get_default_device()
    config: Modelconfig = field(default_factory=Modelconfig)


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


config = Config()
mconfig = config.config
seeding(mconfig.random_seed)


class PeptideMass:
    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues: dict[str, float]):
        if not isinstance(residues, dict):
            raise TypeError("residues must be a dict[str, float]")

        self.masses = residues

    def __len__(self):
        return len(self.masses)

    def _split_sequence(self, seq: str):
        return re.split(r"(?<=.)(?=[A-Z])", seq)

    def mass(self, seq, charge: int | None = None):
        if isinstance(seq, str):
            seq = self._split_sequence(seq)

        try:
            calc_mass = sum(self.masses[aa] for aa in seq) + self.h2o
        except KeyError as e:
            raise KeyError(f"Unknown residue detected: {e.args[0]}") from e

        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass


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
    truncated_pred = batch_truncate_after_eos(max_indices, 24, 0)
    matches = truncated_pred == truth
    num_exact_matches = torch.sum(matches.all(dim=1))
    pep_top1_recall = num_exact_matches.item() / len(pred)
    return pep_top1_recall


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = 0,
        train_label_smoothing: float = 0.01,
        use_weighted_score: bool = True,
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
            use_weighted_score: whether to enable score-based weighting.
            score_mean, score_std: 全局（dataset-level）score均值与标准差。若 use_weighted_score=True，**必须提供**。
            weight_min, weight_max: 最终权重映射区间 [weight_min, weight_max]。
            eps: 防止除零。
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.train_label_smoothing = train_label_smoothing
        self.use_weighted_score = use_weighted_score
        if self.use_weighted_score and (score_mean is None or score_std is None):
            raise ValueError(
                "use_weighted_score=True 时必须提供 score_mean 和 score_std（全局统计）"
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
        if self.use_weighted_score and score is not None:
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
            z = (torch.tensor(score, device=device) - self.score_mean) / (self.score_std + self.eps)
            s = torch.sigmoid(z)
            weights = self.weight_min + s * (self.weight_max - self.weight_min)  # (B,)

            loss = (sample_loss * weights).mean()
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
        max_charge=10,
        min_length=8,
        train_label_smoothing=0.01,
        warmup_iters=100_000,
        cosine_schedule_period_iters=600_000,
        top_k=10,
        top_k_output=10,
        use_weighted_score=True,
        score_mean: float = 2.16,
        score_std: float = 0.75,
        weight_min: float = 0.5,
        weight_max: float = 1.5,
        precursor_mass_tol: int = 2,
        precursor_mass_ppm: int = 20,  # ppm
    ):
        super().__init__()
        self.residues = residues
        self.top_k = top_k
        self.top_k_output = top_k_output
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
        )
        self.decoder = PeptideNARDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            drop_out=dropout,
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
            use_weighted_score=use_weighted_score,
            score_mean=score_mean,
            score_std=score_std,
            weight_min=weight_min,
            weight_max=weight_max,
        )

        self.use_weighted_score = use_weighted_score
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        self.min_length = min_length
        self.max_length = max_length
        self.precursor_mass_tol = precursor_mass_tol
        self.precursor_mass_ppm = precursor_mass_ppm
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        self._aa2idx = {aa: i + 1 for i, aa in enumerate(self._amino_acids)}
        self._idx2aa = {i: aa for aa, i in self._aa2idx.items()}
        self.stop_token = self._aa2idx["$"]
        self.dp_decoder = DP_Decoder(
            residues=residues,
            top_k=self.top_k,
            top_k_output=self.top_k_output,
            eos_idx=self.stop_token,
            min_length=self.min_length,
            max_length=self.max_length,
            mass_tol=self.precursor_mass_tol,
            ppm=self.precursor_mass_ppm,
        )

    def calculate_dp_top10_recall(self, pred, truth_sequences, precursors):  # noqa: C901
        """
        输入：
            pred: (B, L, V)
            truth_sequences: list[str]
            precursors: (B, ?)
        """
        if len(truth_sequences) == 0:
            return 0.0
        precursor_masses_np = precursors[:, 0].detach().cpu().numpy()

        dp_predict_seq, _, _, valid_mask = self.dp_decoder.find_top_sequence(
            logits=pred, precursors_masses=precursor_masses_np
        )
        correct = 0
        total = 0
        for i in range(len(truth_sequences)):
            truth_seq = truth_sequences[i]
            if not isinstance(truth_seq, str) or len(truth_seq.strip()) == 0:
                continue
            if valid_mask[i] == 0:
                total += 1
                continue
            truth_clean = truth_seq.strip().replace("I", "L")
            if "$" in truth_clean:
                truth_clean = truth_clean.split("$")[0]
            pred_list = dp_predict_seq[i]
            pred_clean_list = []
            for seq in pred_list:
                if not seq:
                    continue
                seq_clean = seq.replace("I", "L").strip()
                if "$" in seq_clean:
                    seq_clean = seq_clean.split("$")[0]
                if seq_clean:
                    pred_clean_list.append(seq_clean)
            if truth_clean in pred_clean_list:
                correct += 1
        return correct / total if total > 0 else 0.0

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

        tokens = [self._aa2idx[aa] for aa in sequence]
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
        memories, mem_masks = self.encoder(spectra, precursors=precursors)
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
        memories, mem_masks = self.encoder(spectra, precursors=precursors)
        logits = self.decoder(
            precursors=precursors, memory=memories, memory_key_padding_mask=mem_masks
        )
        return logits, spec_idx, precursors

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[str]],
        *args,
    ):
        pred, scan_num, precursors = self.forward(*batch)
        precursor_masses_np = precursors[:, 0].cpu().numpy()

        dp_predict_seq, _, dp_predict_scores, valid_mask = self.dp_decoder.find_top_sequence(
            logits=pred, precursors_masses=precursor_masses_np
        )
        final_output = []
        for i in range(len(scan_num)):
            current_scan = scan_num[i]
            seq_list = dp_predict_seq[i]
            score_list = dp_predict_scores[i]
            if len(seq_list) == 0:
                continue
            for rank_idx, (seq, score) in enumerate(
                zip(seq_list, score_list, strict=False), start=1
            ):
                final_output.append([current_scan, seq, round(float(score), 4), rank_idx])

        return final_output

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[str]],
        mode: str = "train",
    ) -> torch.Tensor:
        pred, truth, raw_name, spec_idx, batch_score = self._forward_step(*batch)
        peptide_recall = pep_recall_evaluate(pred, truth)

        "recall"
        dp_top10_recall = 0.0
        if mode in ["eval", "val"]:
            truth_sequences = []
            for seq in batch[2]:  # batch[2]是peptide序列
                if isinstance(seq, str):
                    # 统一预处理：去空、去终止符
                    clean_seq = seq.strip().rstrip("$")
                    truth_sequences.append(clean_seq)
                else:
                    truth_sequences.append("")

            dp_top10_recall = self.calculate_dp_top10_recall(pred, truth_sequences, batch[1])
        "loss"
        B = pred.shape[0]
        vocab_size = self.decoder.num_classes
        # batch_size = pred.shape[0]

        pred = pred.reshape(-1, vocab_size + 1)
        truth = truth.flatten()
        if mode == "train" and self.use_weighted_score:
            loss = self.weighted_celoss(pred, truth, score=batch_score, batch_size=B)
        elif mode == "train" and not self.use_weighted_score:
            loss = self.celoss(pred, truth)
        else:
            loss = self.val_celoss(pred, truth)

        if mode == "train":
            return loss, peptide_recall  # training: return 2 metrics
        else:
            return loss, peptide_recall, dp_top10_recall  # eval, val: return 3 metrics


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


def load_encoder_weigth(new_model: Spec2pep, pretrained_path: str) -> None:
    pretrained_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if "state_dict" in pretrained_state:
        pretrained_state = pretrained_state["state_dict"]

    encoder_state = {}
    for key, value in pretrained_state.items():
        if key.startswith("encoder."):
            encoder_state[key] = value

    if "config" in pretrained_state:
        pretrained_dim = pretrained_state["config"]["dim_model"]
        if pretrained_dim != new_model.encoder.dim_model:
            raise ValueError(
                f"Pretrained model dimension ({pretrained_dim}) "
                f"does not match current model ({new_model.encoder.dim_model})"
            )
    return new_model


def load_model_weight(new_model: Spec2pep, pretrained_path: str) -> None:  # noqa: C901
    pretrained_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    if "state_dict" in pretrained_state:
        pretrained_state = pretrained_state["state_dict"]

    state = {}

    for key, value in pretrained_state.items():
        if key.startswith("encoder.diff_encoder.") or key.startswith("encoder.fusion_proj."):
            continue
        # mapping old tokenizer encoder keys to new encoder keys
        if key == "encoder.int_mz_embedding.weight":
            new_key = "encoder.tokenizer_encoder.int_mz_embedding.weight"

        elif key == "encoder.dec_mz_embedding.weight":
            new_key = "encoder.tokenizer_encoder.dec_mz_embedding.weight"

        elif key == "encoder.input_proj.weight":
            new_key = "encoder.tokenizer_encoder.input_proj.weight"

        elif key == "encoder.input_proj.bias":
            new_key = "encoder.tokenizer_encoder.input_proj.bias"

        elif key == "encoder.token_lookup":
            new_key = "encoder.tokenizer_encoder.token_lookup"

        else:
            new_key = key

        if new_key in state:
            raise ValueError(f"Key collision detected: {new_key}")

        state[new_key] = value

    missing_keys, unexpected_keys = new_model.load_state_dict(state, strict=False)

    print(f"Loaded encoder weights from {pretrained_path}")

    critical_keys = [
        "encoder.tokenizer_encoder.int_mz_embedding.weight",
        "encoder.tokenizer_encoder.dec_mz_embedding.weight",
        "encoder.tokenizer_encoder.input_proj.weight",
    ]

    missing_critical = [k for k in missing_keys if k in critical_keys]

    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")

    if missing_critical:
        raise RuntimeError(f"Critical weights NOT loaded: {missing_critical}")

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
        config: Modelconfig,
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
            api_token=os.getenv("NEPTUNE_API_TOKEN"),
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
            random_state=self.config.random_seed,
            min_intensity=self.config.min_intensity,
            remove_precursor_tol=self.config.remove_precursor_tol,
            weighted_sample=self.config.use_weighted_sample,
        )
        self.loaders.setup()

        train_loader = self.loaders.get_train_loader()  # train
        valid_loader = self.loaders.get_val_loader()  # val
        train_eval_loader = self.loaders.get_train_eval_loader()  # eval

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
            "train_eval_dataset"
            self.model.eval()
            train_eval_loss, train_eval_recall, train_eval_dp_recall = 0, 0, 0
            with torch.no_grad():
                for batch in train_eval_loader:
                    batch = tuple(
                        x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                    )
                    loss, recall, dp_recall = self.model.training_step(batch, mode="val")
                    train_eval_loss += loss.item()
                    train_eval_recall += recall
                    train_eval_dp_recall += dp_recall if dp_recall is not None else 0

            train_eval_loss /= len(train_eval_loader)
            train_eval_recall /= len(train_eval_loader)
            train_eval_dp_recall /= len(train_eval_loader)

            val_loss, top1_val_peptide_recall, val_dp_recall = self.validate(valid_loader)

            if top1_val_peptide_recall > self.max_recall:
                self.max_recall = top1_val_peptide_recall
                model_save_path = (
                    str(self.model_save_path).replace(".ckpt", "")
                    + "_val1_recall_"
                    + f"{top1_val_peptide_recall:.3f}"
                    + ".ckpt"
                )
                torch.save(self.model.state_dict(), model_save_path)
            print(
                f"Epoch {epoch + 1}: Train Loss = {epoch_loss:.4f}, "
                f"Train Eval Loss = {train_eval_loss:.4f}, Train Eval Top1 Recall = {train_eval_recall:.3f}, Train Eval Top10 Recall = {train_eval_dp_recall:.3f}, "
                f"Val Loss = {val_loss:.4f}, Val Top1 Recall = {top1_val_peptide_recall:.3f}, Val Top10 Recall = {val_dp_recall:.3f}"
            )
            run["train/epoch_loss"].log(epoch_loss)
            run["val/epoch_loss"].log(val_loss)
            run["val/epoch_pep_top1_recall"].log(top1_val_peptide_recall)
            run["train_eval_loss"].log(train_eval_loss)
            run["train_eval_recall"].log(train_eval_recall)
            run["train_eval/dp_top10_recall"].log(train_eval_dp_recall)
            run["val/dp_top10_recall"].log(val_dp_recall)

    def validate(self, valid_loader):
        self.model.eval()
        losses = []
        recalls = []
        dp_recalls = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = tuple(
                    x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch
                )
                loss, top1_recall, dp_recall = self.model.training_step(batch, mode="val")

                losses.append(float(loss.item()) if not torch.isnan(loss) else 0.0)
                recalls.append(float(top1_recall) if top1_recall is not None else 0.0)
                dp_recalls.append(
                    float(dp_recall) if dp_recall is not None and not np.isnan(dp_recall) else 0.0
                )

        avg_loss = np.mean(losses) if losses else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_dp_recall = np.mean(dp_recalls) if dp_recalls else 0.0

        return avg_loss, avg_recall, avg_dp_recall

    def predict(self, predict_folder: str, out_put_folder: str):
        folder_path = Path(predict_folder)
        test_files = folder_path.glob("*.hdf5")
        self.initialize_model(mode="predict")
        for test_file_path in test_files:
            score_top_1_output_csv_path = out_put_folder + "/" + test_file_path.stem + "_result.csv"
            file_path = Path(score_top_1_output_csv_path)
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
                    # self.model.predict_step(batch)
                    predict_batch_table = self.model.predict_step(batch)
                    predict_result.extend(predict_batch_table)
            "测试速度时 跳过"
            if predict_result:
                all_merged = np.vstack(predict_result)
                columns = ["spec_idx", "modified_sequence", "nar_dp_score", "nar_dp_top"]
                df = pd.DataFrame(all_merged, columns=columns)
                scored_top1_df, filtered_scored_top1_df = pGlyco_score.score_sequence(
                    df, test_file_path
                )
                score_top_1_output_csv_path = (
                    out_put_folder + "/" + test_file_path.stem + "_result.csv"
                )
                filtered_scored_top1_df.to_csv(score_top_1_output_csv_path, index=False)

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
            use_weighted_score=self.config.use_weighted_score,
            score_mean=self.config.score_mean,
            score_std=self.config.score_std,
            weight_min=self.config.weight_min,
            weight_max=self.config.weight_max,
            precursor_mass_ppm=self.config.precursor_mass_ppm,
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


# predict(predict_folder=r'X:\chenzx\raw_Data\HLA\HLA_v1_2_all_data\HLA_v1_2_unseen\hdf_by_alpharaw\predict_result_v2_simple_mod_max_recall_0301\test_dp\hdf_test', out_put_folder= r'X:\chenzx\raw_Data\HLA\HLA_v1_2_all_data\HLA_v1_2_unseen\hdf_by_alpharaw\predict_result_v2_simple_mod_max_recall_0301\test_dp\hdf_test', model=r'X:\chenzx\raw_Data\HLA\HLA_v2_all_data\trained_weights\FeNNetNovo_HLA_v2_SOTA_simple_mod_500psm_max_recall_0301.ckpt')
