"""
Funções auxiliares para o fluxo de treinamento do Desafio 3.
"""
import os
import json
import numpy as np
import pandas as pd

from src.utils.paths import (
    PATH_RESULTS_TABELAS_TABELA_COMPLETA,
    PATH_TABELA_EXPERIMENTOS,
    PATH_LOGS,
    PATH_RAIZ
)

PATH_LOG_DESAFIO = os.path.join(PATH_RAIZ, "src", "desafios", "log_desafio_3.txt")

def formatar_nome_d3(arq: str, loss: str, opt: str, sched: str, es: str, lr: float, aug: bool, seed: int) -> str:
    """Gera o nome do modelo idêntico ao novo padrão do rodar_experimentos."""
    return f"{arq}_{loss}_{opt}_{sched}_{es}_lr{lr}_aug{aug}_seed{seed}"

def experimento_ja_executado(nome_modelo: str) -> bool:
    """Verifica na tabela completa se o experimento já ocorreu e obteve resultados."""
    if not os.path.exists(PATH_RESULTS_TABELAS_TABELA_COMPLETA) or not os.path.exists(PATH_TABELA_EXPERIMENTOS):
        return False
    try:
        df_completa = pd.read_csv(PATH_RESULTS_TABELAS_TABELA_COMPLETA)
        na_completa = nome_modelo in df_completa["nome_modelo"].values

        df_exp = pd.read_csv(PATH_TABELA_EXPERIMENTOS)
        padrao_busca = f"Mod: {nome_modelo} |"
        na_exp = df_exp["configuracao"].str.contains(padrao_busca, regex=False).any()

        return bool(na_completa or na_exp)
    except pd.errors.EmptyDataError:
        return False

def escrever_log_etapa(etapa: int, resultados: str) -> None:
    """Escreve o log de conclusão de etapa no arquivo txt."""
    os.makedirs(os.path.dirname(PATH_LOG_DESAFIO), exist_ok=True)
    with open(PATH_LOG_DESAFIO, "a", encoding="utf-8") as f:
        f.write(f"Etapa {etapa} terminada, resultados:\n{resultados}\n{'-'*50}\n")
    print(f"\n[LOG] Etapa {etapa} registrada no log com sucesso.")

def analisar_estabilidade(nome_modelo: str, mdice_base: float) -> float:
    """
    Lê o JSONL do modelo e calcula um score combinado.
    Score = mDice - Punição por variância nos gradientes - Punição por derivação (drift) do theta.
    """
    caminho_jsonl = os.path.join(PATH_LOGS, f"{nome_modelo}.jsonl")
    if not os.path.exists(caminho_jsonl):
        return mdice_base
        
    grads, thetas = [], []
    with open(caminho_jsonl, "r", encoding="utf-8") as f:
        for linha in f:
            try:
                dados = json.loads(linha)
                if "grad" in dados and "theta" in dados:
                    grads.append(dados["grad"])
                    thetas.append(dados["theta"])
            except json.JSONDecodeError:
                continue

    if not grads:
        return mdice_base

    std_grad = np.std(grads)
    variacao_theta = (max(thetas) - min(thetas)) / (np.mean(thetas) + 1e-8)
    
    score = mdice_base - (0.01 * std_grad) - (0.05 * variacao_theta)
    return float(score)

def get_melhores_etapa(padrao_busca: str) -> pd.DataFrame:
    """Filtra resultados no CSV completo que contenham um padrão de string."""
    if not os.path.exists(PATH_RESULTS_TABELAS_TABELA_COMPLETA):
        return pd.DataFrame()
    df = pd.read_csv(PATH_RESULTS_TABELAS_TABELA_COMPLETA)
    return df[df['nome_modelo'].str.contains(padrao_busca)]