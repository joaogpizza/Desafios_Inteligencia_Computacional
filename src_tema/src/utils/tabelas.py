""" Módulo para lidar com o salvamento dos resultados em formato tabular (CSV) """

import os
import pandas as pd

from src.utils.paths import (
    PATH_RESULTS_TABELAS_TABELA_COMPLETA,
    PATH_TABELA_EXPERIMENTOS
)


def salvar_resultado_completo(
    nome_modelo: str, num_epocas: int, lr: float, early_stopper: str, 
    otimizador: str, scheduler: str, loss: str, usou_aug: bool, 
    melhor_epoca: int, melhor_mdice: float, metricas_avaliacao: dict
) -> None:
    """
    Salva uma linha na tabela completa com os hyperparâmetros e todas 
    as métricas extraídas do avaliador.
    """
    dados = {
        "nome_modelo": nome_modelo,
        "numero_epocas": num_epocas,
        "learning_rate": lr,
        "early_stopper": early_stopper,
        "otimizador": otimizador,
        "scheduler": scheduler,
        "loss": loss,
        "usou_aumento_dados": usou_aug,
        "melhor_epoca": melhor_epoca,
        "melhor_mdice_validacao": melhor_mdice
    }
    
    for classe_metrica, valores in metricas_avaliacao.items():
        if isinstance(valores, dict):
            for metrica, valor in valores.items():
                dados[f"{classe_metrica}_{metrica}"] = valor
        else:
            dados[classe_metrica] = valores

    df_novo = pd.DataFrame([dados])
    
    if os.path.exists(PATH_RESULTS_TABELAS_TABELA_COMPLETA):
        df_existente = pd.read_csv(PATH_RESULTS_TABELAS_TABELA_COMPLETA)
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    os.makedirs(os.path.dirname(PATH_RESULTS_TABELAS_TABELA_COMPLETA), exist_ok=True)
    df_final.to_csv(PATH_RESULTS_TABELAS_TABELA_COMPLETA, index=False)


def salvar_tabela_experimentos(
    nome_modelo: str, num_epocas: int, lr: float, early_stopper: str, 
    otimizador: str, scheduler: str, loss: str, usou_aug: bool, 
    melhor_mdice: float, divergiu: bool, tempo_treino: float
) -> None:
    """
    Cria a entrada formatada requerida no arquivo 'experimentos.csv'.
    """
    config_str = (
        f"Mod: {nome_modelo} | Ep: {num_epocas} | LR: {lr} | ES: {early_stopper} | "
        f"Opt: {otimizador} | Sched: {scheduler} | Loss: {loss} | Aug: {usou_aug}"
    )
    
    dados = {
        "configuracao": config_str,
        "melhor_mdice_validacao": melhor_mdice,
        "divergiu": divergiu,
        "tempo_treino_segundos": tempo_treino
    }
    
    df_novo = pd.DataFrame([dados])
    
    if os.path.exists(PATH_TABELA_EXPERIMENTOS):
        df_existente = pd.read_csv(PATH_TABELA_EXPERIMENTOS)
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    os.makedirs(os.path.dirname(PATH_TABELA_EXPERIMENTOS), exist_ok=True)
    df_final.to_csv(PATH_TABELA_EXPERIMENTOS, index=False)