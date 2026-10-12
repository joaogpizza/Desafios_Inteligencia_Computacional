""" Lida com a orquestração do treinamento e log de métricas da época """

import os
import time
import torch

from src.utils.paths import PATH_RESULTS_MODELOS
from src.utils.checkpoints import salvar_checkpoint
from src.utils.logs import adicionar_log
from src.treino.curvas_treino import plotar_losses, plotar_mdice_val, plotar_taxa_aprendizado
from src.treino.validacao import validar
from configs.basicas import DEVICE

from src.modelos.modelo_factory import pegar_modelo
from src.losses.loss_factory import pegar_loss
from src.otimizadores.otimizador_factory import pegar_otimizador
from src.schedulerslr.scheduler_factory import pegar_scheduler
from src.early_stopping.early_stopping_factory import pegar_early_stopper


def obter_camadas_alvo_monitoramento(modelo: torch.nn.Module) -> list:
    """
    Descobre e retorna, de forma automática, a primeira, uma do meio e a 
    última camada do modelo para monitorar a razão de atualização (rho).
    """
    nomes = [n for n, p in modelo.named_parameters() if p.ndim >= 2]
    if len(nomes) >= 3:
        return [nomes[0], nomes[len(nomes) // 2], nomes[-1]]
    return nomes


def treinar_epoca(
    nome_modelo: str, 
    modelo: torch.nn.Module, 
    dataloader, 
    f_loss, 
    otimizador, 
    scaler,
    camadas_monitoradas: list,
    passo_global: int, 
    k: int = 50
) -> tuple:
    """Treina o modelo por uma única época, gerando os logs requisitados a cada k passos."""
    modelo.train()
    loss_epoca = 0.0

    for inputs, mascaras in dataloader:
        inputs = inputs.to(DEVICE)
        mascaras = mascaras.to(DEVICE)

        otimizador.zero_grad()
        
        with torch.amp.autocast(device_type="cuda"):
            logits = modelo(inputs)
            perda = f_loss(logits, mascaras)
        
        scaler.scale(perda).backward()

        vai_gerar_log = (passo_global + 1) % k == 0

        scaler.unscale_(otimizador)

        if vai_gerar_log:
            g = torch.sqrt(sum((p.grad**2).sum() for p in modelo.parameters() if p.grad is not None))
            th = torch.sqrt(sum((p**2).sum() for p in modelo.parameters()))
            antes = {n: p.detach().clone() for n, p in modelo.named_parameters() if n in camadas_monitoradas}

        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)

        scaler.step(otimizador)
        scaler.update()
        
        passo_global += 1

        if vai_gerar_log:
            rho = {n: ((p - antes[n]).norm() / (antes[n].norm() + 1e-8)).item()
                   for n, p in modelo.named_parameters() if n in camadas_monitoradas}

            lr_atual = otimizador.param_groups[0]["lr"]
            adicionar_log(
                nome=nome_modelo,
                passo=passo_global,
                perda=perda.item(),
                grad_norm=g.item(),
                param_norm=th.item(),
                lr=lr_atual,
                razoes_atualizacao=rho
            )

        loss_epoca += perda.item() * inputs.size(0)

    loss_media = loss_epoca / len(dataloader.dataset)
    return loss_media, passo_global


def treinar_modelo(
    nome_modelo: str,
    num_epocas: int,
    lr: float,
    nome_arquitetura: str,
    nome_loss: str,
    nome_otimizador: str,
    nome_scheduler: str,
    nome_early_stopper: str,
    dataloader_treino,
    dataloader_val,
    **kwargs
) -> tuple:
    """
    Função principal que orquestra todo o processo, invocando instâncias a partir
    das strings recebidas e controlando o fluxo do projeto.
    """
    os.makedirs(PATH_RESULTS_MODELOS, exist_ok=True)
    
    modelo = pegar_modelo(nome_arquitetura).to(DEVICE)
    f_loss = pegar_loss(nome_loss, **kwargs.get('loss_kwargs', {}))
    otimizador = pegar_otimizador(nome_otimizador, modelo.parameters(), lr, **kwargs.get('otimizador_kwargs', {}))
    scheduler = pegar_scheduler(nome_scheduler, otimizador, **kwargs.get('scheduler_kwargs', {}))
    early_stopper = pegar_early_stopper(nome_early_stopper, **kwargs.get('early_stopper_kwargs', {}))

    scaler = torch.amp.GradScaler("cuda")

    camadas_monitoradas = obter_camadas_alvo_monitoramento(modelo)

    historico_loss_treino = []
    historico_loss_val = []
    historico_mdice_val = []
    historico_lr = []

    melhor_mdice = -1.0
    melhor_epoca = 0
    divergiu = False
    passo_global = 0

    inicio_treino = time.time()

    for epoca in range(1, num_epocas + 1):
        historico_lr.append(otimizador.param_groups[0]["lr"])

        loss_treino, passo_global = treinar_epoca(
            nome_modelo=nome_modelo,
            modelo=modelo,
            dataloader=dataloader_treino,
            f_loss=f_loss,
            otimizador=otimizador,
            scaler=scaler,
            camadas_monitoradas=camadas_monitoradas,
            passo_global=passo_global,
            k=50
        )

        if torch.isnan(torch.tensor(loss_treino)) or loss_treino > 1e4:
                    divergiu = True
                    print(f"\n[AVISO] Treinamento do modelo {nome_modelo} divergiu na época {epoca}.")
                    historico_lr.pop()
                    if epoca == 1:
                        caminho_salvar = os.path.join(PATH_RESULTS_MODELOS, f"{nome_modelo}.pt")
                        salvar_checkpoint(modelo, otimizador, epoca, caminho=caminho_salvar)
                    break

        loss_val, mdice_val, _ = validar(modelo, dataloader_val, f_loss)
        
        if nome_scheduler.upper() == "REDUCE_ON_PLATEAU":
            scheduler.step(loss_val)
        else:
            scheduler.step()

        historico_loss_treino.append(loss_treino)
        historico_loss_val.append(loss_val)
        historico_mdice_val.append(mdice_val)

        if mdice_val > melhor_mdice:
            melhor_mdice = mdice_val
            melhor_epoca = epoca
            caminho_salvar = os.path.join(PATH_RESULTS_MODELOS, f"{nome_modelo}.pt")
            salvar_checkpoint(modelo, otimizador, epoca, caminho=caminho_salvar)

        print(f'Final da epoca {epoca}, mdice = {mdice_val}')
        
        _ = early_stopper(perda_val=loss_val, perda_treino=loss_treino, metrica_val=mdice_val)
        if early_stopper.parada_antecipada:
            print(f"\n[INFO] Early Stopping engatilhado de forma precoce na época {epoca}.")
            break

    plotar_losses(nome_modelo, historico_loss_treino, historico_loss_val)
    plotar_mdice_val(nome_modelo, historico_mdice_val)
    plotar_taxa_aprendizado(nome_modelo, historico_lr)
    tempo_treino = time.time() - inicio_treino

    return melhor_mdice, melhor_epoca, divergiu, tempo_treino