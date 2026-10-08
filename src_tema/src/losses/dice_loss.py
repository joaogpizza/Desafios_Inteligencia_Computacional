""" Implementação da Dice Loss """

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Função de perda por Dice.
    """
    def __init__(self, smooth: float = 1.0) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
        ) -> torch.Tensor:
        # transforma logits em probabilidades [0, 1]
        probavilidades_pred = torch.sigmoid(pred)
        
        probavilidades_pred = probavilidades_pred.view(probavilidades_pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # cálculo do dice
        intersecao = (probavilidades_pred * target).sum(dim=1)
        dice = (2.0 * intersecao + self.smooth) / (probavilidades_pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        
        return 1.0 - dice.mean()