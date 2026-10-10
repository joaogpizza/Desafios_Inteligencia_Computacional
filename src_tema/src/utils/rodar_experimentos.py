"""
Módulo para automatizar e orquestrar o fluxo completo de experimentos de segmentação.
"""

import os
import torch

from configs.basicas import NUM_EPOCAS, TAXA_APRENDIZADO, DEVICE
from src.utils.seed import fixar_seed
from src.utils.paths import PATH_RESULTS_MODELOS
from src.utils.checkpoints import carregar_checkpoint
from src.utils.tabelas import salvar_resultado_completo, salvar_tabela_experimentos
from src.data.dataloader import criar_dataloaders
from src.treino.treinador import treinar_modelo
from src.avaliacao.avaliador import avaliar_modelo
from src.modelos.modelo_factory import pegar_modelo


def rodar_experimento_completo(
    nome_arquitetura: str,
    nome_loss: str,
    nome_otimizador: str,
    nome_scheduler: str,
    nome_early_stopper: str,
    lr: float = TAXA_APRENDIZADO,
    num_epocas: int = NUM_EPOCAS,
    aplicar_aug: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    fracao_dados: float = 1.0,
    **kwargs
) -> dict:
    """
    Roda o pipeline completo de treinamento e avaliação.
    
    Args:
        nome_arquitetura (str): Arquitetura (ex: "UNET", "ATUNET").
        nome_loss (str): Função de custo (ex: "DICE", "FOCAL").
        nome_otimizador (str): Otimizador (ex: "ADAMW", "MUON").
        nome_scheduler (str): Agendador de Learning Rate (ex: "COSSENO").
        nome_early_stopper (str): Critério de parada (ex: "PADRAO", "PQ").
        lr (float, optional): Taxa de aprendizado. Default = TAXA_APRENDIZADO.
        num_epocas (int, optional): Número de épocas. Default = NUM_EPOCAS.
        aplicar_aug (bool, optional): Se aplica Data Augmentation. Default = False.
        num_workers (int, optional): Workers para o DataLoader. Default = 4.
        seed (int, optional): Semente aleatória. Default = 42.
        **kwargs: Argumentos extras repassados para a loss, otimizador, etc.
        
    Returns:
        dict: Métricas do modelo avaliado no conjunto de teste.
    """
    nome_modelo = f"{nome_arquitetura}_{nome_loss}_{nome_otimizador}_{nome_scheduler}_{nome_early_stopper}_lr{lr}_aug{aplicar_aug}"
    print(f"{'='*60}")
    print(f"Iniciando Experimento: {nome_modelo}")
    print(f"{'='*60}")

    fixar_seed(seed)
    print(f"[INFO] Seed fixada em {seed}.")

    print("[INFO] Criando DataLoaders (Treino, Validação, Teste)...")
    dl_treino, dl_val, dl_teste = criar_dataloaders(
        num_workers=num_workers, 
        aplicar_aug=aplicar_aug,
        fracao_dados=fracao_dados
    )

    print("[INFO] Iniciando rotina de treinamento...")
    melhor_mdice, melhor_epoca, divergiu, tempo_treino = treinar_modelo(
        nome_modelo=nome_modelo,
        num_epocas=num_epocas,
        lr=lr,
        nome_arquitetura=nome_arquitetura,
        nome_loss=nome_loss,
        nome_otimizador=nome_otimizador,
        nome_scheduler=nome_scheduler,
        nome_early_stopper=nome_early_stopper,
        dataloader_treino=dl_treino,
        dataloader_val=dl_val,
        **kwargs
    )

    print("\n[INFO] Carregando o melhor checkpoint para avaliação no conjunto de teste...")
    modelo = pegar_modelo(nome_arquitetura).to(DEVICE)
    caminho_checkpoint = os.path.join(PATH_RESULTS_MODELOS, f"{nome_modelo}.pt")
    
    carregar_checkpoint(modelo, caminho=caminho_checkpoint)

    print("[INFO] Avaliando o modelo (Teste)...")
    metricas_avaliacao = avaliar_modelo(modelo, dl_teste)
    print(f"[RESULTADOS TESTE] mDice: {metricas_avaliacao['mdice']:.4f} | mIoU: {metricas_avaliacao['miou']:.4f}")

    print("\n[INFO] Salvando resultados nas tabelas (completa e de experimentos)...")
    salvar_resultado_completo(
        nome_modelo=nome_modelo,
        num_epocas=num_epocas,
        lr=lr,
        early_stopper=nome_early_stopper,
        otimizador=nome_otimizador,
        scheduler=nome_scheduler,
        loss=nome_loss,
        usou_aug=aplicar_aug,
        melhor_epoca=melhor_epoca,
        melhor_mdice=melhor_mdice,
        metricas_avaliacao=metricas_avaliacao
    )

    salvar_tabela_experimentos(
        nome_modelo=nome_modelo,
        num_epocas=num_epocas,
        lr=lr,
        early_stopper=nome_early_stopper,
        otimizador=nome_otimizador,
        scheduler=nome_scheduler,
        loss=nome_loss,
        usou_aug=aplicar_aug,
        melhor_mdice=melhor_mdice,
        divergiu=divergiu,
        tempo_treino=tempo_treino
    )

    print(f"\n[SUCESSO] Experimento '{nome_modelo}' finalizado!")
    return metricas_avaliacao
