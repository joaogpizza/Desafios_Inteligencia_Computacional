import math

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class CossenoComWarmup(_LRScheduler):
    """
    Cosseno Annealing com Warmup inicial.
    """
    def __init__(self, optimizer: Optimizer, T_warmup: int, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        self.T_warmup = T_warmup
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            # fase de Warmup, lr aumenta linearmente de 0 até base_lr
            return [base_lr * (self.last_epoch + 1) / self.T_warmup for base_lr in self.base_lrs]
        else:
            # fase Cosseno
            curr_epoch = self.last_epoch - self.T_warmup
            T_cos = self.T_max - self.T_warmup
            
            if curr_epoch >= T_cos:
                 return [self.eta_min for _ in self.base_lrs]
                 
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * curr_epoch / T_cos)) / 2
                    for base_lr in self.base_lrs]