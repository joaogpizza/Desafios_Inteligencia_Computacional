""" Factory das funções de loss """

import torch.nn as nn

from src.losses.focal_dice_loss import FocalDiceLoss
from src.losses.dice_loss import DiceLoss
from src.losses.focal_loss import FocalLoss

def pegar_loss(nome: str, **kwargs) -> nn.Module:
    """
    Funções de loss possíveis:

    - "FOCAL": Focal loss implementada em src/losses/focal_loss

    - "DICE": Dice loss implementada em src/losses/dice_loss

    - "FOCALDICE": Focal + Dice loss implementada em
    src/losses/focal_dice_loss

    Levanta ValueError caso o nome informado seja diferente de
    um listado.
    """
    nome_upper = nome.upper()
    
    if nome_upper == "FOCAL":
        return FocalLoss(**kwargs)
    elif nome_upper == "DICE":
        return DiceLoss(**kwargs)
    elif nome_upper == "FOCALDICE":
        return FocalDiceLoss(**kwargs)
    else:
        raise ValueError("Função de loss desconhecida")
