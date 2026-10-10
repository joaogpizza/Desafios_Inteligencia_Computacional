from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer

class ConstanteLR(_LRScheduler):
    """
    Mantém a taxa de aprendizado constante.
    """
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]
