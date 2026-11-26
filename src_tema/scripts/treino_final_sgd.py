"""
Treino final (dataset completo, 100 épocas) das 9 configurações:

    UKAN + FOCALDICE + SGD (lr=0.1) + early stopper PADRAO + augmentation
    schedulers: COSSENO, CONSTANTE, REDUCE_ON_PLATEAU
    seeds:      42, 125, 2026

Local: src_tema/scripts/treino_final_sgd.py
Execução (a partir de src_tema):

    python scripts/treino_final_sgd.py
    python -m scripts.treino_final_sgd

Opções:
    --num-workers N   workers dos DataLoaders (padrão: 4)
    --dry-run         só mostra o status de cada configuração, sem treinar nem mover nada

Comportamento:
- O número de épocas vem de configs/basicas.NUM_EPOCAS (precisa ser 100). Ele também
  define o T_max do scheduler COSSENO, por isso não é repassado como argumento.
- É retomável: configuração já concluída com 100 épocas (ou que divergiu com 100 épocas)
  é pulada. Se o script cair no meio, basta rodar de novo.
- Os nomes dos modelos são os mesmos das rodadas antigas (25 épocas, 10% dos dados).
  Se ainda existir alguma linha antiga (outro número de épocas) em tabela_completa.csv
  ou experimentos.csv para essas configurações, o script ABORTA antes de treinar, para
  não misturar resultados. Mova esses resultados (tabelas, checkpoints, logs) para o
  backup e rode de novo.
- Log .jsonl de configuração ainda não concluída (sobra de uma execução interrompida,
  por exemplo) é movido para desafio3/logs_incompletos/, porque o log é aberto em modo
  append e misturaria passos de execuções diferentes. Nada é apagado.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Garante que `configs` e `src` (que ficam em src_tema) possam ser importados,
# independente de como o script for chamado.
PATH_SRC_TEMA = Path(__file__).resolve().parents[1]
if str(PATH_SRC_TEMA) not in sys.path:
    sys.path.insert(0, str(PATH_SRC_TEMA))

import pandas as pd  # noqa: E402

from configs.basicas import NUM_EPOCAS  # noqa: E402
from src.desafios.desafio_3_utils import formatar_nome_d3  # noqa: E402
from src.utils.paths import (  # noqa: E402
    PATH_LOGS,
    PATH_RESULTS_TABELAS_TABELA_COMPLETA,
    PATH_TABELA_EXPERIMENTOS,
)
from src.utils.rodar_experimentos import rodar_experimento_completo  # noqa: E402


# ---------------------------------------------------------------------------
# Configuração do experimento
# ---------------------------------------------------------------------------
ARQUITETURA = "UKAN"
LOSS = "FOCALDICE"
OTIMIZADOR = "SGD"
EARLY_STOPPER = "PADRAO"
LR = 0.1
AUMENTO_DADOS = True
SCHEDULERS = ["COSSENO", "CONSTANTE", "REDUCE_ON_PLATEAU"]
SEEDS = [42, 125, 2026]

EPOCAS_ESPERADAS = 100
FRACAO_DADOS = 0.25
NUM_WORKERS_PADRAO = 4

CONCLUIDA = "CONCLUIDA"
DIVERGIU = "DIVERGIU"
PENDENTE = "PENDENTE"
OBSOLETA = "OBSOLETA"


def montar_configuracoes() -> list[tuple[str, int, str]]:
    """Retorna (scheduler, seed, nome_modelo) na ordem de execução."""
    return [
        (
            sched,
            seed,
            formatar_nome_d3(
                ARQUITETURA, LOSS, OTIMIZADOR, sched, EARLY_STOPPER, LR, AUMENTO_DADOS, seed
            ),
        )
        for sched in SCHEDULERS
        for seed in SEEDS
    ]


# ---------------------------------------------------------------------------
# Consulta às tabelas de resultados
# ---------------------------------------------------------------------------
def _ler_csv(caminho: str) -> pd.DataFrame:
    if not os.path.exists(caminho):
        return pd.DataFrame()
    try:
        return pd.read_csv(caminho)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def consultar_registros(
    nome: str, df_completa: pd.DataFrame, df_exp: pd.DataFrame
) -> tuple[set[int], list[tuple[int, bool]]]:
    """
    Procura o modelo nas duas tabelas.

    Retorna:
        - conjunto de números de épocas registrados em tabela_completa.csv;
        - lista de (épocas, divergiu) registrados em experimentos.csv.
    """
    epocas_completa: set[int] = set()
    if not df_completa.empty and {"nome_modelo", "numero_epocas"} <= set(df_completa.columns):
        linhas = df_completa[df_completa["nome_modelo"] == nome]
        epocas_completa = {int(e) for e in linhas["numero_epocas"]}

    registros_exp: list[tuple[int, bool]] = []
    if not df_exp.empty and {"configuracao", "divergiu"} <= set(df_exp.columns):
        marcador = f"Mod: {nome} |"  # o " |" final evita casar seed42 com seed425, etc.
        mascara = df_exp["configuracao"].astype(str).str.contains(marcador, regex=False)
        for _, linha in df_exp[mascara].iterrows():
            achado = re.search(r"Ep: (\d+)", str(linha["configuracao"]))
            if achado:
                divergiu = str(linha["divergiu"]).strip().lower() == "true"
                registros_exp.append((int(achado.group(1)), divergiu))
    return epocas_completa, registros_exp


def avaliar_status(
    nome: str, df_completa: pd.DataFrame, df_exp: pd.DataFrame
) -> tuple[str, str]:
    """Retorna (status, detalhe) da configuração em relação a EPOCAS_ESPERADAS."""
    epocas_completa, registros_exp = consultar_registros(nome, df_completa, df_exp)

    outras_epocas = (epocas_completa | {e for e, _ in registros_exp}) - {EPOCAS_ESPERADAS}
    if outras_epocas:
        lista = ", ".join(str(e) for e in sorted(outras_epocas))
        return OBSOLETA, f"registro(s) antigo(s) com {lista} épocas"

    if EPOCAS_ESPERADAS in epocas_completa:
        return CONCLUIDA, ""
    if any(e == EPOCAS_ESPERADAS and div for e, div in registros_exp):
        return DIVERGIU, "divergiu em execução anterior (linha em experimentos.csv)"
    return PENDENTE, ""


# ---------------------------------------------------------------------------
# Logs .jsonl
# ---------------------------------------------------------------------------
def caminho_log(nome: str) -> Path:
    return Path(PATH_LOGS) / f"{nome}.jsonl"


def mover_log_incompleto(nome: str) -> Path | None:
    """Move o .jsonl de uma configuração não concluída para logs_incompletos/."""
    origem = caminho_log(nome)
    if not origem.exists():
        return None
    pasta_destino = Path(PATH_LOGS).parent / "logs_incompletos"
    pasta_destino.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = pasta_destino / f"{nome}__{carimbo}.jsonl"
    shutil.move(str(origem), str(destino))
    return destino


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def liberar_memoria() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def formatar_duracao(segundos: float) -> str:
    h, resto = divmod(int(segundos), 3600)
    m, s = divmod(resto, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def imprimir_plano(configs: list[tuple[str, int, str]], status: dict[str, tuple[str, str]]) -> None:
    print(f"Plano ({len(configs)} configurações, {EPOCAS_ESPERADAS} épocas, "
          f"fracao_dados={FRACAO_DADOS}):")
    for i, (_, _, nome) in enumerate(configs, 1):
        st, detalhe = status[nome]
        extra = f" ({detalhe})" if detalhe else ""
        print(f"  [{i}/{len(configs)}] {st:<9} {nome}{extra}")
    print()


def imprimir_resumo(configs: list[tuple[str, int, str]]) -> None:
    """Mostra os resultados de teste das configurações concluídas."""
    df = _ler_csv(PATH_RESULTS_TABELAS_TABELA_COMPLETA)
    if df.empty or "nome_modelo" not in df.columns:
        print("[RESUMO] Nenhum resultado encontrado em tabela_completa.csv.")
        return

    nomes = [nome for _, _, nome in configs]
    df = df[df["nome_modelo"].isin(nomes) & (df["numero_epocas"] == EPOCAS_ESPERADAS)]
    if df.empty:
        print("[RESUMO] Nenhuma configuração concluída com "
              f"{EPOCAS_ESPERADAS} épocas.")
        return

    colunas = ["nome_modelo", "melhor_epoca", "melhor_mdice_validacao", "mdice", "miou"]
    colunas = [c for c in colunas if c in df.columns]
    print("\n[RESUMO] Resultados (mdice/miou = conjunto de teste):")
    print(df[colunas].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if {"scheduler", "mdice"} <= set(df.columns):
        agregado = df.groupby("scheduler")["mdice"].agg(["mean", "std", "count"])
        print("\n[RESUMO] mDice de teste por scheduler (média, desvio amostral, nº de seeds):")
        print(agregado.to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS_PADRAO,
                        help=f"workers dos DataLoaders (padrão: {NUM_WORKERS_PADRAO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="só mostra o status das configurações, sem treinar nem mover nada")
    args = parser.parse_args()

    if NUM_EPOCAS != EPOCAS_ESPERADAS:
        print(f"[ERRO] configs/basicas.NUM_EPOCAS = {NUM_EPOCAS}, mas este script "
              f"espera {EPOCAS_ESPERADAS}. Ajuste NUM_EPOCAS e rode de novo.")
        return 1

    configs = montar_configuracoes()
    df_completa = _ler_csv(PATH_RESULTS_TABELAS_TABELA_COMPLETA)
    df_exp = _ler_csv(PATH_TABELA_EXPERIMENTOS)
    status = {nome: avaliar_status(nome, df_completa, df_exp) for _, _, nome in configs}

    imprimir_plano(configs, status)

    obsoletas = [nome for nome, (st, _) in status.items() if st == OBSOLETA]
    if obsoletas:
        print("[ERRO] Ainda há resultados antigos (outro número de épocas) para "
              "configurações deste script:")
        for nome in obsoletas:
            print(f"  - {nome}: {status[nome][1]}")
        print("\nMova esses resultados (linhas das duas tabelas, checkpoints .pt e logs "
              ".jsonl) para o backup e rode o script de novo. Nada foi treinado nem alterado.")
        return 2

    pendentes = [nome for _, _, nome in configs if status[nome][0] == PENDENTE]
    logs_incompletos = [nome for nome in pendentes if caminho_log(nome).exists()]

    if args.dry_run:
        for nome in logs_incompletos:
            print(f"[DRY-RUN] Log existente seria movido para logs_incompletos/: {nome}.jsonl")
        print("[DRY-RUN] Nada foi treinado nem movido.")
        return 0

    for nome in logs_incompletos:
        destino = mover_log_incompleto(nome)
        print(f"[INFO] Log de execução incompleta movido: {nome}.jsonl -> {destino}")

    resultados: dict[str, str] = {}
    falhas: list[str] = []
    inicio_total = time.time()

    for i, (sched, seed, nome) in enumerate(configs, 1):
        print(f"\n{'#' * 70}\n# [{i}/{len(configs)}] {nome}\n{'#' * 70}")

        st, detalhe = status[nome]
        if st in (CONCLUIDA, DIVERGIU):
            print(f"[CACHE] Já executado com {EPOCAS_ESPERADAS} épocas ({st.lower()}); pulando.")
            resultados[nome] = st
            continue

        inicio = time.time()
        try:
            metricas = rodar_experimento_completo(
                nome_arquitetura=ARQUITETURA,
                nome_loss=LOSS,
                nome_otimizador=OTIMIZADOR,
                nome_scheduler=sched,
                nome_early_stopper=EARLY_STOPPER,
                lr=LR,
                aplicar_aug=AUMENTO_DADOS,
                seed=seed,
                fracao_dados=FRACAO_DADOS,
                num_workers=args.num_workers,
            )
        except Exception:  # uma falha não derruba as configurações restantes
            traceback.print_exc()
            print(f"[FALHA] {nome} (após {formatar_duracao(time.time() - inicio)}).")
            falhas.append(nome)
            resultados[nome] = "FALHOU"
        else:
            if metricas:
                resultados[nome] = "OK"
                print(f"[OK] {nome} em {formatar_duracao(time.time() - inicio)}.")
            else:
                resultados[nome] = "DIVERGIU"
                print(f"[AVISO] {nome} divergiu ({formatar_duracao(time.time() - inicio)}).")
        finally:
            liberar_memoria()

    print(f"\n{'=' * 70}\nTempo total: {formatar_duracao(time.time() - inicio_total)}")
    for nome, res in resultados.items():
        print(f"  {res:<9} {nome}")

    imprimir_resumo(configs)

    if falhas:
        print(f"\n[ATENÇÃO] {len(falhas)} configuração(ões) falharam. "
              "Rode o script de novo para retomar apenas as que faltam.")
        return 1
    print("\n[OK] Todas as configurações foram processadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())