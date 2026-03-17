"""未使用"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class DecoderHead(ABC):
    """
    概率生成层
    encoder > decoder
    """

    @abstractmethod
    def forward(
        self,
        memory: torch.Tensor,
        precursors: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        生成概率分布（logits）

        Returns:
            logits: [B, L, V] 或 [L, B, V]
        """
        pass


class DecoderTail(ABC):
    """
    序列搜索层
    从概率分布中提取topK序列
    """

    @abstractmethod
    def decode(self, logits: torch.Tensor) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """
        从 logits 解码序列

        Args:
            logits: [B, L, V] 概率或原始 logits

        Returns:
            sequences: [B, K] 肽段序列
            gathered_logits: [B, K, L] 对应的 logits
            scores: [B, K] 分数
        """
        pass
