import re

import torch

from foxnovo.constants import HYDROGEN_MASS, OXYGEN_MASS, PROTON_MASS


class PeptideMass:
    # Constants
    hydrogen = HYDROGEN_MASS
    oxygen = OXYGEN_MASS
    h2o = 2 * hydrogen + oxygen
    proton = PROTON_MASS

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


def pep_recall_evaluate(pred, truth, eos_token):
    max_values, max_indices = torch.max(pred, dim=2)
    truncated_pred = batch_truncate_after_eos(max_indices, eos_token, 0)
    matches = truncated_pred == truth
    num_exact_matches = torch.sum(matches.all(dim=1))
    pep_top1_recall = num_exact_matches.item() / len(pred)
    return pep_top1_recall


def worker_predict_step(input_queue, output_queue, model, device):
    with torch.inference_mode():
        while True:
            batch = input_queue.get()
            if batch is None:
                break

            batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)

            # 这里 model 100% 不是 None！
            result = model.predict_step(batch)
            output_queue.put(result)
