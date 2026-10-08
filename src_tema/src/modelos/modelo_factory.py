""" Factory de modelos """

import torch.nn as nn

from src.modelos.attention_unet import AttentionUNet

def pegar_modelo(nome: str) -> nn.Module:
    """
    Modelos possíveis:

    - "ATUNET": Attention U-Net definida em src/modelos/attention_unet.py

    Levanta ValueError caso o nome informado seja diferente de
    um listado.
    """
    if nome.upper() == "ATUNET":
        return AttentionUNet()
    else:
        raise ValueError("Arquitetura desconhecida")