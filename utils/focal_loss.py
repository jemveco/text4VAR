import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification implementation in PyTorch"""
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate Cross-Entropy Loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        # Compute probability of true class
        pt = torch.exp(-ce_loss)
        # Apply focal adjustment
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

