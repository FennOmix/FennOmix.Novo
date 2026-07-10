import re

import torch
import torch.nn.functional as F

from foxnovo.constants import STOP_TOKEN
from foxnovo.decoders.dp_decoder import DP_Decoder
from foxnovo.decoders.peptide_decoder import PeptideNARDecoder
from foxnovo.encoders.spectrum_encoder import SpectrumEncoder

from .losses import WeightedCrossEntropyLoss
from .utils import PeptideMass, pep_recall_evaluate


class FoxNovoNARModel(torch.nn.Module):
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
        self._amino_acids = list(self._peptide_mass.masses.keys()) + [STOP_TOKEN]
        self._aa2idx = {aa: i + 1 for i, aa in enumerate(self._amino_acids)}
        self._idx2aa = {i: aa for aa, i in self._aa2idx.items()}
        self.stop_token = self._aa2idx[STOP_TOKEN]
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
            total += 1
            if valid_mask[i] == 0:
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
            return sequence
        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

        if not partial:
            sequence += [STOP_TOKEN]

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

        tokens = [self.tokenize(s) for s in sequences]
        tokens = [
            F.pad(token, (0, self.max_length + 1 - len(token)), value=0) for token in tokens
        ]
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
        peptide_recall = pep_recall_evaluate(pred, truth, self.stop_token)

        "recall"
        dp_top10_recall = 0.0
        if mode in ["eval", "val"]:
            truth_sequences = []
            for seq in batch[2]:
                if isinstance(seq, str):
                    clean_seq = seq.strip().rstrip("$")
                    truth_sequences.append(clean_seq)
                else:
                    truth_sequences.append("")

            dp_top10_recall = self.calculate_dp_top10_recall(pred, truth_sequences, batch[1])
        "loss"
        B = pred.shape[0]
        vocab_size = self.decoder.num_classes

        pred = pred.reshape(-1, vocab_size + 1)
        truth = truth.flatten()
        if mode == "train" and self.use_weighted_score:
            loss = self.weighted_celoss(pred, truth, score=batch_score, batch_size=B)
        elif mode == "train" and not self.use_weighted_score:
            loss = self.celoss(pred, truth)
        else:
            loss = self.val_celoss(pred, truth)

        if mode == "train":
            return loss, peptide_recall
        else:
            return loss, peptide_recall, dp_top10_recall
