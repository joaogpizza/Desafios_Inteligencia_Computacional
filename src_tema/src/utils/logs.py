""" Módulo para lidar com o log do treinamento """

import os
import json

from src.utils.paths import PATH_LOGS


def adicionar_log(
    nome: str, 
    passo: int, 
    perda: float, 
    grad_norm: float, 
    param_norm: float, 
    lr: float, 
    razoes_atualizacao: dict
) -> None:
    """
    Adiciona uma nova entrada no log do modelo a cada k passos.
    O arquivo é salvo no formato JSON Lines (.jsonl) para lidar facilmente
    com dicionários dinâmicos de atualização das camadas (rho).
    
    Args:
        nome (str): O nome do modelo/experimento (para nomear o arquivo).
        passo (int): O passo de atualização global atual.
        perda (float): A loss calculada naquele passo.
        grad_norm (float): A norma L2 dos gradientes antes do clipping.
        param_norm (float): A norma L2 dos parâmetros.
        lr (float): A taxa de aprendizado efetiva no momento.
        razoes_atualizacao (dict): Dicionário com a razão de atualização (rho) por camada monitorada.
    """
    os.makedirs(PATH_LOGS, exist_ok=True)
    caminho_arquivo = os.path.join(PATH_LOGS, f"{nome}.jsonl")
    
    registro = {
        "passo": passo,
        "perda": perda,
        "grad": grad_norm,
        "theta": param_norm,
        "lr": lr,
    }
    # espalha as razões de atualização de cada camada prefixadas com 'rho/'
    registro.update({f"rho/{k}": v for k, v in razoes_atualizacao.items()})
    
    with open(caminho_arquivo, 'a', encoding='utf-8') as f:
        f.write(json.dumps(registro) + '\n')