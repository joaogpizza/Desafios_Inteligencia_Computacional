import torch
import os

def salvar_checkpoint(modelo, otimizador, epoca, caminho='best_model.pt'):
    """Salva o estado do modelo e do otimizador."""
    print(f"=> Salvando novo melhor checkpoint em: {caminho}")
    checkpoint = {
        'epoca': epoca,
        'model_state_dict': modelo.state_dict(),
        'optimizer_state_dict': otimizador.state_dict(),
    }
    torch.save(checkpoint, caminho)

def carregar_checkpoint(modelo, otimizador=None, caminho='best_model.pt'):
    """Carrega os pesos para o modelo e, opcionalmente, para o otimizador."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Checkpoint não encontrado em: {caminho}")
        
    print(f"=> Carregando checkpoint de: {caminho}")
    checkpoint = torch.load(caminho)
    
    modelo.load_state_dict(checkpoint['model_state_dict'])
    
    if otimizador is not None and 'optimizer_state_dict' in checkpoint:
        otimizador.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint.get('epoca', 0)