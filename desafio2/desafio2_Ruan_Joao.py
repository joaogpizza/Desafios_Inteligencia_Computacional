import torch
import torch.nn as nn
import math

def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x) * (1/math.sqrt(2))

@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                 fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    if camada == 1:
        fator_tabela = 1.0 
    else:
        fator_tabela = 2.5
    std = math.sqrt(fator_tabela / fan_in)
    nn.init.normal_(W, mean=0.0, std=std)
    nn.init.zeros_(b)
