"""
Otimizadores possíveis:
- "ADAM"
- "ADAMW"
- "MUON"
- "SGD"
- "RMSPROP"

Agendas possíveis:
- "CONSTANTE"
- "COSSENO"
- "WARMUP_COSSENO"
- "REDUCE_ON_PLATEAU"

Modifique NUM_WORKERS baseado na potência de sua GPU e a sua VRAM.

Certifique-se que os dados foram baixados e o script em
"src_tema/scripts/criacao_patches.py" foi executado.

Observação: O inicio da rotina de treinamento demora mesmo, então seja paciente!
"""

import os
import sys

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_pai = os.path.abspath(os.path.join(diretorio_atual, '..'))
diretorio_src_tema = os.path.join(diretorio_pai, 'src_tema')

if diretorio_pai not in sys.path:
    sys.path.append(diretorio_pai)
if diretorio_src_tema not in sys.path:
    sys.path.append(diretorio_src_tema)

from src_tema.src.utils.rodar_experimentos import rodar_experimento_completo

OTIMIZADOR = "ADAM"
AGENDA = "CONSTANTE"
TAXA_APRENDIZADO = 0.001
SEED = 100
NUM_WORKERS = 0

if __name__ == "__main__":
    rodar_experimento_completo(
        nome_arquitetura="UNET",
        nome_loss="FOCALDICE",
        nome_otimizador=OTIMIZADOR,
        nome_scheduler=AGENDA,
        nome_early_stopper="PADRAO",
        lr=TAXA_APRENDIZADO,
        aplicar_aug=True,
        seed=SEED,
        num_workers=NUM_WORKERS,
        fracao_dados=0.1,
    )