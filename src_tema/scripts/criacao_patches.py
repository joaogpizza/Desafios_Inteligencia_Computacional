import os
import sys
import cv2
from tqdm import tqdm

from src.utils.paths import (
    PATH_RAIZ,
    PATH_RAW_TRAIN_IMG,
    PATH_RAW_TRAIN_MASK,
    PATH_RAW_VAL_IMG,
    PATH_RAW_VAL_MASK,
    PATH_RAW_TEST_IMG,
    PATH_RAW_TEST_MASK,
    PATH_PATCHES_IMGS_TRAIN,
    PATH_PATCHES_IMGS_VAL,
    PATH_PATCHES_IMGS_TEST,
    PATH_PATCHES_GT_TRAIN,
    PATH_PATCHES_GT_VAL,
    PATH_PATCHES_GT_TEST
)

TAMANHO_ORIGINAL = 512
TAMANHO_PATCH = 256
STRIDE = 256

def criar_diretorios_saida():
    """Garante que todas as pastas de saída existam antes do processamento."""
    diretorios_saida = [
        PATH_PATCHES_IMGS_TRAIN, PATH_PATCHES_IMGS_VAL, PATH_PATCHES_IMGS_TEST,
        PATH_PATCHES_GT_TRAIN, PATH_PATCHES_GT_VAL, PATH_PATCHES_GT_TEST
    ]
    for diretorio in diretorios_saida:
        os.makedirs(diretorio, exist_ok=True)


def processar_split(nome_split, dir_in_img, dir_in_mask, dir_out_img, dir_out_mask):
    """
    Processa todas as imagens e máscaras de um split específico gerando patches.
    """
    arquivos_imgs = [f for f in os.listdir(dir_in_img) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    arquivos_masks = [f for f in os.listdir(dir_in_mask) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    dict_masks = {os.path.splitext(f)[0]: f for f in arquivos_masks}

    print(f"\n[{nome_split.upper()}] Processando arquivos...")
    
    for nome_img in tqdm(
        arquivos_imgs,
        desc=f"Gerando patches - {nome_split}",
        unit="img",
        ncols=100,
        ascii=False,
        file=sys.stdout
    ):
        stem_img, _ = os.path.splitext(nome_img)
        
        if stem_img not in dict_masks:
            tqdm.write(f"Aviso: Máscara não encontrada para a imagem {nome_img}. Ignorando.")
            continue
            
        nome_mask = dict_masks[stem_img]
        
        caminho_img = os.path.join(dir_in_img, nome_img)
        caminho_mask = os.path.join(dir_in_mask, nome_mask)
        
        img = cv2.imread(caminho_img, cv2.IMREAD_COLOR) 
        mask = cv2.imread(caminho_mask, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            tqdm.write(f"Erro ao ler imagem ou máscara: {stem_img}. Ignorando.")
            continue
            
        altura, largura = img.shape[:2]
        
        if altura != TAMANHO_ORIGINAL or largura != TAMANHO_ORIGINAL:
            tqdm.write(f"Aviso: Imagem {nome_img} com dimensões ({altura}x{largura}) diferentes de 512x512. Ignorando.")
            continue

        patch_idx = 0
        for y in range(0, altura, STRIDE):
            for x in range(0, largura, STRIDE):
                
                if y + TAMANHO_PATCH > altura or x + TAMANHO_PATCH > largura:
                    continue
                
                patch_img = img[y:y+TAMANHO_PATCH, x:x+TAMANHO_PATCH]
                patch_mask = mask[y:y+TAMANHO_PATCH, x:x+TAMANHO_PATCH]
                
                nome_saida = f"{stem_img}_p{patch_idx:02d}.png"
                caminho_salvar_img = os.path.join(dir_out_img, nome_saida)
                caminho_salvar_mask = os.path.join(dir_out_mask, nome_saida)
                
                cv2.imwrite(caminho_salvar_img, patch_img)
                cv2.imwrite(caminho_salvar_mask, patch_mask)
                
                patch_idx += 1

def main():
    print("Iniciando rotina de fatiamento (Patch Cropping) - WHU Building Dataset")
    print(f"Diretório Raiz Identificado: {PATH_RAIZ}")
    print(f"Tamanho do Patch: {TAMANHO_PATCH}x{TAMANHO_PATCH} | Stride: {STRIDE}\n")
    
    criar_diretorios_saida()
    
    splits_config = [
        ("train", PATH_RAW_TRAIN_IMG, PATH_RAW_TRAIN_MASK, PATH_PATCHES_IMGS_TRAIN, PATH_PATCHES_GT_TRAIN),
        ("val", PATH_RAW_VAL_IMG, PATH_RAW_VAL_MASK, PATH_PATCHES_IMGS_VAL, PATH_PATCHES_GT_VAL),
        ("test", PATH_RAW_TEST_IMG, PATH_RAW_TEST_MASK, PATH_PATCHES_IMGS_TEST, PATH_PATCHES_GT_TEST)
    ]
    
    for split in splits_config:
        processar_split(
            nome_split=split[0],
            dir_in_img=split[1],
            dir_in_mask=split[2],
            dir_out_img=split[3],
            dir_out_mask=split[4]
        )
        
    print("\nProcessamento finalizado com sucesso!")

if __name__ == "__main__":
    main()