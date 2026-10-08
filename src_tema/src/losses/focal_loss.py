""" Implementação da Focal Loss """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FocalLoss(nn.Module):
    """
    Função de perda Focal (Focal Loss).
    Ideal para lidar com o desbalanceamento de classes, dando mais foco em exemplos difíceis.
    """
    def __init__(self, alpha: Optional[float] = 0.25, gamma: float = 2.0, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
        ) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Probabilidade da classe correta (p_t)
        # O exp(-BCE) é um truque matemático no PyTorch para obter p_t diretamente
        pt = torch.exp(-bce_loss)
        
        # Calcula o componente focal: (1 - p_t)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        # Aplica o fator de balanceamento alpha, se fornecido
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
            
        # Aplica o método de redução escolhido
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss