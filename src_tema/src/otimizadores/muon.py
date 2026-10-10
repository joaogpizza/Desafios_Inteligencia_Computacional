import torch
from torch.optim import Optimizer, AdamW

class MuonAdamW(Optimizer):
    """
    Implementação do otimizador Muon combinado com AdamW.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, 
                 adamw_lr=3e-4, adamw_betas=(0.90, 0.95), weight_decay=0.01):
        
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr=adamw_lr, adamw_betas=adamw_betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        self.muon_params = []
        self.adamw_params = []
        
        # separa os parâmetros com base na dimensionalidade
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim >= 2:
                    self.muon_params.append(p)
                else:
                    self.adamw_params.append(p)
                    
        self.adamw = AdamW(self.adamw_params, lr=adamw_lr, betas=adamw_betas, weight_decay=weight_decay)
        
        self.adamw_lr_ratio = adamw_lr / lr if lr > 0 else 1.0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            current_lr = group['lr']
            for adam_group in self.adamw.param_groups:
                adam_group['lr'] = current_lr * self.adamw_lr_ratio
            
        # passo do AdamW para parâmetros 1D
        if len(self.adamw_params) > 0:
            self.adamw.step()
        
        # passo do Muon para parâmetros >= 2D
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p not in self.muon_params or p.grad is None:
                    continue
                    
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                if nesterov:
                    g = grad + momentum * buf
                else:
                    g = buf
                    
                g_ortho = self._zeropower_via_newtonschulz5(g, steps=ns_steps)
                
                p.data.add_(g_ortho, alpha=-lr)
                
        return loss

    def _zeropower_via_newtonschulz5(self, G, steps=5):
        """Método auxiliar de iteração Newton-Schulz."""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.clone()
        transpose = X.size(0) > X.size(1)
        if transpose:
            X = X.T
        
        X = X / (X.norm() + 1e-7)
        
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
            
        if transpose:
            X = X.T
        return X