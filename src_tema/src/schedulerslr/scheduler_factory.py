import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Optimizer

from src.schedulerslr.constante import ConstanteLR
from src.schedulerslr.cosseno_warmup import CossenoComWarmup
from configs.basicas import NUM_EPOCAS

def pegar_scheduler(nome: str, optimizer: Optimizer, **kwargs):
    """
    Cria e retorna o scheduler desejado.
    
    Nomes possíveis:
    - "CONSTANTE"
    - "COSSENO"
    - "WARMUP_COSSENO"
    - "REDUCE_ON_PLATEAU"

    Parâmetros universais obrigatórios:
    - nome: String com o nome do scheduler.
    - optimizer: O otimizador instanciado.
    
    Parâmetros opcionais via kwargs:
    - eta_min (float): para COSSENO e WARMUP_COSSENO (lr mínimo, padrão: 0)
    - mode (str): para REDUCE_ON_PLATEAU ('min' ou 'max', padrão: 'min')
    - factor (float): para REDUCE_ON_PLATEAU (fator de redução, padrão: 0.1)
    - patience (int): para REDUCE_ON_PLATEAU (épocas de espera, padrão: 10)
    """
    nome_upper = nome.upper()
    
    if nome_upper == "CONSTANTE":
        return ConstanteLR(optimizer)
        
    elif nome_upper == "COSSENO":
        T_max = NUM_EPOCAS
        eta_min = kwargs.get("eta_min", 0.0)
        
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
    elif nome_upper == "WARMUP_COSSENO":
        T_max = NUM_EPOCAS
        T_warmup = T_max * 0.1
        eta_min = kwargs.get("eta_min", 0.0)
        
        return CossenoComWarmup(optimizer, T_warmup=T_warmup, T_max=T_max, eta_min=eta_min)
        
    elif nome_upper == "REDUCE_ON_PLATEAU":
        mode = kwargs.get("mode", "min")
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 5)
        
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        
    else:
        raise ValueError(f"Scheduler desconhecido: '{nome}'")