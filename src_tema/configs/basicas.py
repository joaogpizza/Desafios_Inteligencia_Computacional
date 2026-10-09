""" Definição de algumas configurações básicas """

import torch.cuda as cuda

TAM_PATCH = 256
NORMALIZACAO_MEAN = (0.485, 0.456, 0.406)
NORMALIZACAO_STD = (0.229, 0.224, 0.225)
NUM_EPOCAS = 50
LIMIAR = 0.5
DEVICE = "cuda" if cuda.is_available() else "cpu"
TAXA_APRENDIZADO = 1e-3
TAM_BATCH = 16