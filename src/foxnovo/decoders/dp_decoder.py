import numpy as np
import torch.nn.functional as F
from numba import njit

from foxnovo.constants import WATER_MASS
from foxnovo.model.utils import PeptideMass

WATER_MW = WATER_MASS


@njit(nogil=True, cache=True)
def find_top_k_seqs_core(  # noqa: C901
    aa_prob_table,
    log_probs,
    idx2mass,
    precursor_mass,
    mass_tol,
    ppm,
    top_k,
    top_k_output,
    eos_idx,
    target_lengths,
):
    L, V = aa_prob_table.shape
    num_targets = len(target_lengths)
    pool_seqs = np.zeros((num_targets * top_k, L), dtype=np.int8)
    pool_scores = np.full(num_targets * top_k, -1e9, dtype=np.float32)
    pool_ptr = 0

    top_k_idx = np.argsort(log_probs[0])[::-1][:top_k]
    curr_seqs = np.zeros((top_k, L), dtype=np.int8)
    curr_seqs[:, 0] = top_k_idx
    cum_log_probs = log_probs[0, top_k_idx].astype(np.float32)
    cum_mass = np.zeros(top_k, dtype=np.float32)

    for i in range(top_k):
        tid = curr_seqs[i, 0]
        cum_mass[i] = idx2mass[tid]

    for i_aa in range(1, L):
        is_target_len = False
        for tl in target_lengths:
            if i_aa == tl:
                is_target_len = True
                break

        if is_target_len:
            log_eos = log_probs[i_aa, eos_idx]
            for k in range(top_k):
                total_mass = cum_mass[k] + WATER_MW
                mass_diff = total_mass - precursor_mass
                delta_tol = round(mass_diff)

                if abs(delta_tol) > mass_tol:
                    continue

                theo_mass = precursor_mass + delta_tol
                current_ppm = abs(total_mass - theo_mass) / theo_mass * 1e6
                if current_ppm > ppm:
                    continue

                norm_score = (cum_log_probs[k] + log_eos) / (i_aa + 1)
                pool_seqs[pool_ptr] = curr_seqs[k]
                pool_seqs[pool_ptr, i_aa] = eos_idx
                pool_scores[pool_ptr] = norm_score
                pool_ptr += 1

        next_log_p = log_probs[i_aa].copy()
        next_log_p[eos_idx] = -1e9
        best_v_idx = np.argsort(next_log_p)[::-1][:top_k]

        combined = (cum_log_probs.reshape(-1, 1) + next_log_p[best_v_idx]).reshape(-1)
        top_idxes = np.argsort(combined)[::-1][:top_k]

        new_seqs = np.zeros((top_k, L), dtype=np.int8)
        new_cum_log = np.zeros(top_k, dtype=np.float32)
        new_cum_mass = np.zeros(top_k, dtype=np.float32)

        for i in range(top_k):
            idx = top_idxes[i]
            r, c = idx // top_k, idx % top_k
            tid = best_v_idx[c]

            new_seqs[i] = curr_seqs[r]
            new_seqs[i, i_aa] = tid
            new_cum_log[i] = combined[idx]
            new_cum_mass[i] = cum_mass[r] + idx2mass[tid]

        curr_seqs = new_seqs
        cum_log_probs = new_cum_log
        cum_mass = new_cum_mass

    valid_num = pool_ptr
    if valid_num == 0:
        return np.empty((0, L), dtype=np.int8), np.empty(0, dtype=np.float32), 0

    final_idx = np.argsort(pool_scores[:valid_num])[::-1][:top_k_output]
    return pool_seqs[final_idx], pool_scores[final_idx], valid_num


@njit(nogil=True, cache=True)
def batch_decode_jit(
    batch_aa_prob_table,
    log_probs_table,
    precursor_mass_array,
    idx2mass,
    mass_tol,
    ppm,
    top_k,
    top_k_output,
    eos_idx,
    target_lengths,
):
    B, L, V = batch_aa_prob_table.shape
    batch_seqs = []
    batch_scores = []
    valid_mask = np.zeros(B, dtype=np.int8)

    for i in range(B):
        seqs, scores, cnt = find_top_k_seqs_core(
            batch_aa_prob_table[i],
            log_probs_table[i],
            idx2mass,
            precursor_mass_array[i],
            mass_tol,
            ppm,
            top_k,
            top_k_output,
            eos_idx,
            target_lengths,
        )
        valid_mask[i] = 1 if cnt > 0 else 0
        batch_seqs.append(seqs)
        batch_scores.append(scores)

    return batch_seqs, batch_scores, valid_mask


class DP_Decoder:
    def __init__(
        self,
        residues="canonical",
        top_k=10,
        top_k_output=10,
        eos_idx=24,
        min_length=8,
        max_length=14,
        mass_tol=2,
        ppm=20,
    ):
        self._peptide_mass = PeptideMass(residues=residues)
        self._amino_acids = list(self._peptide_mass.masses.keys()) + ["$"]
        self._idx2aa = np.array([""] + self._amino_acids)
        self._idx2mass = np.zeros(len(self._idx2aa), dtype=np.float32)
        for i, aa in enumerate(self._idx2aa):
            if aa in self._peptide_mass.masses:
                self._idx2mass[i] = self._peptide_mass.masses[aa]

        self.top_k = top_k
        self.top_k_output = top_k_output
        self.eos_idx = eos_idx
        self.target_lengths = np.arange(min_length, max_length + 1)
        self.mass_tol = mass_tol
        self.ppm = ppm

    def translate_top_k_seqs(self, batch_seqs_list):
        result = []
        for seqs in batch_seqs_list:
            translated = []
            for s in seqs:
                end = np.where(s == self.eos_idx)[0][0] if self.eos_idx in s else len(s)
                seq_str = "".join(self._idx2aa[s[:end]])
                translated.append(seq_str)
            result.append(translated)
        return result

    def find_batch_top_k_seqs(self, batch_aa_prob_table, precursor_mass_array):
        log_probs_table = np.log(batch_aa_prob_table + 1e-10)
        return batch_decode_jit(
            batch_aa_prob_table,
            log_probs_table,
            precursor_mass_array,
            self._idx2mass,
            self.mass_tol,
            self.ppm,
            self.top_k,
            self.top_k_output,
            self.eos_idx,
            self.target_lengths,
        )

    def find_top_sequence(self, logits, precursors_masses):
        """ori_name: find_top_sequence_fenn"""
        logits = logits.clone()
        logits[..., 0] = -1e9
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        seqs_list, scores_list, valid_mask = self.find_batch_top_k_seqs(probs, precursors_masses)
        final_seqs = self.translate_top_k_seqs(seqs_list)
        return final_seqs, seqs_list, scores_list, valid_mask
