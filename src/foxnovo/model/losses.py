import torch
import torch.nn.functional as F
from torch import nn


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
            score_mean, score_std: dataset-level mean score and std, when use_weighted_score=True.
            weight_min, weight_max: [weight_min, weight_max].
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
            device = pred.device
            B = batch_size
            L = truth.numel() // B
            loss_per_token = loss_per_token.view(B, L)
            truth_2d = truth.view(B, L)
            mask = (truth_2d != self.ignore_index).float()
            valid_counts = mask.sum(dim=1).clamp(min=1.0)

            sample_loss = loss_per_token.sum(dim=1) / valid_counts

            z = (torch.tensor(score, device=device) - self.score_mean) / (self.score_std + self.eps)
            s = torch.sigmoid(z)
            weights = self.weight_min + s * (self.weight_max - self.weight_min)

            loss = (sample_loss * weights).mean()
        else:
            loss = loss_per_token.mean()

        return loss
