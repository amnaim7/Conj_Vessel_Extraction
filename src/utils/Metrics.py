import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
)
from torchmetrics.segmentation import DiceScore

from src.utils.config import (
    ACCURACY, DICE_SCORE, JACCARD_INDEX,
    PRECISION, RECALL, SPECIFICITY,
    F1_SCORE, MCC,
)


class Metrics:
    """
    Lightweight metrics wrapper for binary segmentation.
    Computes and prints metrics only — no history storage.
    """

    def __init__(self, device: torch.device, threshold: float = 0.5):
        self.device = device
        self.threshold = threshold

        self.metrics = MetricCollection(
            {
                ACCURACY: BinaryAccuracy(),
                DICE_SCORE: DiceScore(num_classes=2),
                JACCARD_INDEX: BinaryJaccardIndex(),
                PRECISION: BinaryPrecision(),
                RECALL: BinaryRecall(),
                SPECIFICITY: BinarySpecificity(),
                F1_SCORE: BinaryF1Score(),
                MCC: BinaryMatthewsCorrCoef(),
            }
        ).to(device)

    def reset(self) -> None:
        self.metrics.reset()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        preds, target_int = self._prepare(logits, target)
        return self.metrics(preds, target_int)

    def compute(self) -> dict:
        return self.metrics.compute()

    def _prepare(self, logits: torch.Tensor, target: torch.Tensor):
        target = target.int().to(self.device)

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).int()

        return preds, target
