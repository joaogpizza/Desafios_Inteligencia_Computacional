import torch
import torch.nn as nn

def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.selu(x)

@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                 fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    
    #lam = 1.0507009873554805
    #fator_1a_camada = 0.5          
    std = 1.0 / fan_in ** 0.5

    #if camada == 1:
    #   std *= fator_1a_camada

    nn.init.normal_(W, mean=0.0, std=std)
    nn.init.zeros_(b)
