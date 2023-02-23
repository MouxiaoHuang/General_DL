import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss.
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """
    def __init__(self,
                 gamma=2,
                 reduction=None,
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ce = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, pred, target):
        """forward"""
        logp = self.ce(pred, target)
        pt = torch.exp(-logp)
        loss = (1 - pt) ** self.gamma * logp
        return loss.mean() * self.loss_weight