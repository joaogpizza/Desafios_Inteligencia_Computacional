from src.early_stopping.padrao import EarlyStoppingPadrao
from src.early_stopping.metrica import EarlyStoppingMetrica
from src.early_stopping.gl import EarlyStoppingGL
from src.early_stopping.pq import EarlyStoppingPQ

def pegar_early_stopper(nome: str, **kwargs):
    """
    Factory para retornar a instância correta do Early Stopper.
    
    Args:
        nome (str): "PADRAO", "METRICA", "GL", ou "PQ".
        **kwargs: Parâmetros específicos repassados para o construtor da classe.
    
    Returns:
        Instância configurada do Early Stopper desejado.
    """
    nome_formatado = nome.upper().strip()
    
    if nome_formatado == "PADRAO":
        return EarlyStoppingPadrao(**kwargs)
        
    elif nome_formatado == "METRICA":
        return EarlyStoppingMetrica(**kwargs)
        
    elif nome_formatado == "GL":
        return EarlyStoppingGL(**kwargs)
        
    elif nome_formatado == "PQ":
        return EarlyStoppingPQ(**kwargs)
        
    else:
        raise ValueError(f"Early Stopper '{nome}' não reconhecido. Opções válidas: 'PADRAO', 'METRICA', 'GL', 'PQ'.")