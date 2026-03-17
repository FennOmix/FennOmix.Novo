import numba
import numpy as np
import torch

from fennomix_novo.depthcharge.masses import PeptideMass  # 后期单独处理

from .base_decoder import DecoderTail


@numba.njit
def find_top_k_seqs(top_k, aa_prob_table: np.ndarray):  # size:[L,V]
    top_k_scores = np.zeros(top_k, dtype=aa_prob_table.dtype)  # size: K
    top_k_seqs = np.zeros((top_k, aa_prob_table.shape[0]), dtype=np.int8)  # size: K,L

    tmp_k_seqs = top_k_seqs.copy()

    top_k_aa_prob_idxes = np.argsort(aa_prob_table[0])[::-1][:top_k]  # topk aa index in first step
    # step 0
    top_k_scores[:] = aa_prob_table[0, top_k_aa_prob_idxes]
    top_k_seqs[:, 0] = top_k_aa_prob_idxes

    for i_aa in range(1, aa_prob_table.shape[0]):
        top_k_aa_prob_idxes = np.argsort(aa_prob_table[i_aa])[::-1][:top_k]
        scores = (
            top_k_scores.reshape(-1, 1) @ aa_prob_table[i_aa, top_k_aa_prob_idxes].reshape(1, -1)
        ).reshape(-1)  # top_k[L-1] @ top_k[L]
        top_k_idxes = np.argsort(scores)[::-1][:top_k]
        for i, k in enumerate(top_k_idxes):
            rank_i, cur = k // top_k, k % top_k
            tmp_k_seqs[i, :i_aa] = top_k_seqs[rank_i, :i_aa]
            tmp_k_seqs[i, i_aa] = top_k_aa_prob_idxes[cur]
        top_k_scores[:] = scores[top_k_idxes]
        top_k_seqs[:, : i_aa + 1] = tmp_k_seqs[:, : i_aa + 1]

    return top_k_seqs, top_k_scores


class DPDecoderTail(DecoderTail):
    """
    动态规划解码器尾
    只负责从概率矩阵中提取 Top-K 序列
    """

    def __init__(
        self,
        residues: str = "canonical",
        top_k: int = 10,
        top_k_output: int = 10,
    ):
        """
        Args:
            residues: 氨基酸残基字典
            top_k: 搜索时保留的序列数
            top_k_output: 输出的序列数
        """
        self._init_vocabulary(residues)
        self.top_k = top_k
        self.top_k_output = top_k_output

    def _init_vocabulary(self, residues: str):
        """初始化氨基酸映射"""
        peptide_mass = PeptideMass(residues=residues)
        amino_acids = list(peptide_mass.masses.keys()) + ["$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(amino_acids)}
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}
        self.stop_token = self._aa2idx["$"]

    def join_no_stoptoken(self, x):
        result = []
        for aa in x:
            if aa == "$":
                break
            result.append(aa)
        return "".join(result)

    def translate_top_k_seqs(self, batch_top_k_seqs: np.ndarray) -> np.ndarray:
        """将索引序列转换为氨基酸序列"""
        vec_map = np.vectorize(self._idx2aa.get)
        seq_array = vec_map(batch_top_k_seqs)
        B, K, _ = seq_array.shape
        aa_seqs = np.empty((B, K), dtype=object)

        for i in range(B):
            for j in range(K):
                aa_seqs[i, j] = self.join_no_stop_token(seq_array[i, j])

        return aa_seqs

    def find_batch_top_k_seqs(
        self, batch_aa_prob_table: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量寻找 top-k 序列"""
        B, L, V = batch_aa_prob_table.shape
        batch_top_k_scores = np.zeros((B, self.top_k_output), dtype=batch_aa_prob_table.dtype)
        batch_top_k_seqs = np.zeros((B, self.top_k, L), dtype=np.int8)

        for i_batch in range(B):
            batch_top_k_seqs[i_batch, :], batch_top_k_scores[i_batch, :] = find_top_k_seqs(
                top_k=self.top_k, aa_prob_table=batch_aa_prob_table[i_batch]
            )

        return batch_top_k_seqs, batch_top_k_scores

    def decode(
        self,
        logits: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """
        从 logits 解码 top-k 序列

        Args:
            logits: [B, L, V] logits

        Returns:
            final_top_k_seqs: [B, top_k_output] 肽段序列
            gathered_logits: [B, top_k, L] 对应的 logits
            batch_top_k_scores: [B, top_k_output] 分数
        """
        "屏蔽PAD Token"
        logits_masked = logits.clone()
        logits_masked[..., 0] = -1e9
        batch_aa_prob_table = torch.softmax(logits_masked, dim=-1).cpu().numpy()
        "动态规划寻找top-k"
        batch_top_k_seqs, batch_top_k_scores = self.find_batch_top_k_seqs(batch_aa_prob_table)
        final_top_k_seqs = self.translate_top_k_seqs(batch_top_k_seqs)
        "提取对应logits"
        batch_top_k_seqs_cuda = torch.tensor(batch_top_k_seqs, device=logits.device).long()
        logits_expanded = logits.unsqueeze(1).expand(
            -1, batch_top_k_seqs_cuda.size(1), -1, -1
        )  # [B, K, L, V]
        indices_expanded = batch_top_k_seqs_cuda.unsqueeze(-1)  # [B, K, L, 1]
        gathered_logits = torch.gather(
            logits_expanded, dim=3, index=indices_expanded
        )  # [B, K, L, 1]
        gathered_logits = gathered_logits.squeeze(-1)  # [B, K, L]

        return final_top_k_seqs, gathered_logits, batch_top_k_scores
