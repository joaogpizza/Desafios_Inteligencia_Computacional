""" Implementação de uma combinação de Focal Loss com Dice Loss """

import torch
import torch.nn as nn

from src.losses.dice_loss import DiceLoss
from src.losses.focal_loss import FocalLoss

class FocalDiceLoss(nn.Module):
    """
    Combinação de Focal Loss e Dice Loss.
    Excelente para dados altamente desbalanceados.
    """
    def __init__(
            self, 
            alpha: float = 0.25, 
            gamma: float = 2.0, 
            smooth: float = 1.0, 
            weight_focal: float = 0.5, 
            weight_dice: float = 0.5
        ) -> None:
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=smooth)
        
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
        ) -> torch.Tensor:
        loss_focal = self.focal(pred, target)
        loss_dice = self.dice(pred, target)
        
        return (self.weight_focal * loss_focal) + (self.weight_dice * loss_dice)
