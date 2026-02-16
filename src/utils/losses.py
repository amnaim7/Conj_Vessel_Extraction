import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore


class DiceBCELoss(nn.Module):
    """
    BCEWithLogits + (1 - Dice), where Dice is computed on thresholded sigmoid outputs.
    Assumes binary segmentation.
    """
    def __init__(self, device: torch.device, threshold: float = 0.5) -> None:
        super().__init__()
        self.device = device
        self.threshold = threshold
        self.dice = DiceScore(num_classes=2).to(device=self.device)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target expected to be 0/1 mask with same shape as logits
        bce = F.binary_cross_entropy_with_logits(logits, target)

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).to(dtype=torch.int32)

        # DiceScore expects integer targets for num_classes=2
        dice_score = self.dice(preds, target.to(dtype=torch.int32))
        dice_loss = 1.0 - dice_score

        return bce + dice_loss
