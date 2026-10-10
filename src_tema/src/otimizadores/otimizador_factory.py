import torch.optim as optim
from src.otimizadores.muon import MuonAdamW

def pegar_otimizador(nome: str, params, lr: float, **kwargs) -> optim.Optimizer:
    """
    Cria e retorna o otimizador desejado.
    
    Parâmetros universais obrigatórios:
    - nome: String com o nome do otimizador.
    - params: modelo.parameters()
    - lr: Taxa de aprendizado base.
    
    Parâmetros opcionais via kwargs:
    - momentum (float): para SGD (padrão: 0.9)
    - nesterov (bool): para SGD (padrão: False)
    - weight_decay (float): para AdamW/SGD/RMSprop (padrão: 0.0)
    - betas (tuple): para Adam/AdamW
    """
    nome_upper = nome.upper()
    
    if nome_upper == "SGD":
        momentum = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", False)
        weight_decay = kwargs.get("weight_decay", 0.0)
        
        return optim.SGD(
            params, 
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov, 
            weight_decay=weight_decay
        )
        
    elif nome_upper == "ADAM":
        betas = kwargs.get("betas", (0.9, 0.999))
        eps = kwargs.get("eps", 1e-8)
        
        return optim.Adam(params, lr=lr, betas=betas, eps=eps)
        
    elif nome_upper == "ADAMW":
        weight_decay = kwargs.get("weight_decay", 0.01)
        betas = kwargs.get("betas", (0.9, 0.999))
        
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
        
    elif nome_upper == "RMSPROP":
        alpha = kwargs.get("alpha", 0.99)
        momentum = kwargs.get("momentum", 0.0)
        
        return optim.RMSprop(params, lr=lr, alpha=alpha, momentum=momentum)
        
    elif nome_upper == "MUON":
        momentum = kwargs.get("momentum", 0.95)
        nesterov = kwargs.get("nesterov", True)
        adamw_lr = kwargs.get("adamw_lr", lr)  # se não passar lr específico para AdamW, usa a lr geral
        
        return MuonAdamW(
            params, 
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov, 
            adamw_lr=adamw_lr
        )
        
    else:
        raise ValueError(f"Otimizador desconhecido: '{nome}'")