"""
Factory para criar os dataloaders de treino, validação e teste
"""

from typing import Tuple

from torch.utils.data import DataLoader

from src.utils.paths import (
    PATH_PATCHES_IMGS_TRAIN, PATH_PATCHES_IMGS_VAL, PATH_PATCHES_IMGS_TEST,
    PATH_PATCHES_GT_TRAIN, PATH_PATCHES_GT_VAL, PATH_PATCHES_GT_TEST
)
from src.data.datasets import HistologiaDataset
from configs.basicas import TAM_BATCH

def criar_dataloaders(
    num_workers: int, 
    aplicar_aug: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria os dataloaders de treino, validação e teste do dataset.

    O retorno é na seguinte ordem: dataloader de treino, dataloader de
    validação e dataloader de teste.
    """

    dataset_treino = HistologiaDataset(
        diretorio_imagens=PATH_PATCHES_IMGS_TRAIN,
        diretorio_mascaras=PATH_PATCHES_GT_TRAIN,
        extensao_imagem=".png",
        aplicar_aug=aplicar_aug
    )

    dataset_val = HistologiaDataset(
        diretorio_imagens=PATH_PATCHES_IMGS_VAL,
        diretorio_mascaras=PATH_PATCHES_GT_VAL,
        extensao_imagem=".png",
        aplicar_aug=False
    )

    dataset_teste = HistologiaDataset(
        diretorio_imagens=PATH_PATCHES_IMGS_TEST,
        diretorio_mascaras=PATH_PATCHES_GT_TEST,
        extensao_imagem=".png",
        aplicar_aug=False
    )

    dl_treino = DataLoader(
        dataset_treino, 
        batch_size=TAM_BATCH, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True
    )
    
    dl_val = DataLoader(
        dataset_val, 
        batch_size=TAM_BATCH, 
        shuffle=False, 
        num_workers=num_workers, 
        drop_last=False
    )
    
    dl_teste = DataLoader(
        dataset_teste, 
        batch_size=TAM_BATCH, 
        shuffle=False, 
        num_workers=num_workers, 
        drop_last=False
    )

    return dl_treino, dl_val, dl_teste