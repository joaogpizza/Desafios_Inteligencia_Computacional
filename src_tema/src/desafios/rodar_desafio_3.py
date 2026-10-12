"""
Módulo mestre de execução do Desafio 3.
"""
from src.utils.rodar_experimentos import rodar_experimento_completo
from src.desafios.desafio_3_utils import (
    experimento_ja_executado,
    formatar_nome_d3,
    escrever_log_etapa,
    analisar_estabilidade,
    get_melhores_etapa
)

# Alterar baseado na VRAM e GPU
NUM_WORKERS = 4

def fluxo_mestre():
    print("Iniciando Fluxo Mestre do Desafio 3...")

    print("\n--- ETAPA 1: Baseline (3 seeds) ---")
    seeds_baseline = [42, 125, 2026]
    resultados_e1 = []
    
    for s in seeds_baseline:
        nome_e1 = formatar_nome_d3("UNET", "FOCALDICE", "ADAMW", "COSSENO", "PADRAO", 0.001, True, s)
        if not experimento_ja_executado(nome_e1):
            rodar_experimento_completo(
                nome_arquitetura="UNET", nome_loss="FOCALDICE", nome_otimizador="ADAMW",
                nome_scheduler="COSSENO", nome_early_stopper="PADRAO", lr=0.001,
                aplicar_aug=True, seed=s, fracao_dados=0.1, num_workers=NUM_WORKERS
            )
        else:
            print(f"[CACHE] Experimento já executado: {nome_e1}")
        resultados_e1.append(nome_e1)

    escrever_log_etapa(1, f"Modelos treinados: {', '.join(resultados_e1)}")

    print("\n--- ETAPA 2: Varredura da LR ---")
    otimizadores_disponiveis = ["SGD", "ADAM", "ADAMW", "RMSPROP", "MUON"]
    lrs_teste = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    seed_etapa2 = 42
    
    for opt in otimizadores_disponiveis:
        for lr in lrs_teste:
            nome_e2 = formatar_nome_d3("UNET", "FOCALDICE", opt, "WARMUP_COSSENO", "PADRAO", lr, True, seed_etapa2)
            if not experimento_ja_executado(nome_e2):
                rodar_experimento_completo(
                    nome_arquitetura="UNET", nome_loss="FOCALDICE", nome_otimizador=opt,
                    nome_scheduler="WARMUP_COSSENO", nome_early_stopper="PADRAO", lr=lr,
                    aplicar_aug=True, seed=seed_etapa2, fracao_dados=0.1, num_workers=NUM_WORKERS
                )
            else:
                print(f"[CACHE] Experimento já executado: {nome_e2}")

    escrever_log_etapa(2, "Varredura concluída para SGD, ADAM, ADAMW, RMSPROP, MUON nas LRs 10^-1 a 10^-5.")

    print("\n--- ETAPA 3: Agendas (Schedulers) ---")
    
    df_e2 = get_melhores_etapa("WARMUP_COSSENO") 
    
    avaliacoes_opt = {}
    for index, row in df_e2.iterrows():
        score = analisar_estabilidade(row["nome_modelo"], row.get("melhor_mdice_validacao", 0))
        opt = row["otimizador"]
        if opt not in avaliacoes_opt or score > avaliacoes_opt[opt]["score"]:
            avaliacoes_opt[opt] = {"score": score, "lr": row["learning_rate"], "nome": row["nome_modelo"]}
    
    opts_ordenados = sorted(avaliacoes_opt.items(), key=lambda x: x[1]["score"], reverse=True)
    top_opts = [o[0] for o in opts_ordenados[:2]]
    if "MUON" not in top_opts and "MUON" in avaliacoes_opt:
        top_opts.append("MUON")
        
    schedulers = ["CONSTANTE", "COSSENO", "WARMUP_COSSENO", "REDUCE_ON_PLATEAU"]
    resultados_e3 = []
    
    for opt in top_opts:
        melhor_lr = avaliacoes_opt[opt]["lr"]
        for sched in schedulers:
            nome_e3 = formatar_nome_d3("UNET", "FOCALDICE", opt, sched, "PADRAO", melhor_lr, True, 42)
            if not experimento_ja_executado(nome_e3):
                rodar_experimento_completo(
                    nome_arquitetura="UNET", nome_loss="FOCALDICE", nome_otimizador=opt,
                    nome_scheduler=sched, nome_early_stopper="PADRAO", lr=melhor_lr,
                    aplicar_aug=True, seed=42, fracao_dados=0.1, num_workers=NUM_WORKERS
                )
            else:
                print(f"[CACHE] Experimento já executado: {nome_e3}")
            resultados_e3.append(nome_e3)

    log_e3_txt = (f"Otimizadores escolhidos: {top_opts}.\n"
                  f"Testados com Constant, Cosseno, Warmup Cosseno e Plateau.")
    escrever_log_etapa(3, log_e3_txt)

    print("\n--- ETAPA 4: Melhores Execuções com Seeds Adicionais ---")
    
    scores_e3 = []
    for nome in resultados_e3:
        df_model = get_melhores_etapa(nome)
        if not df_model.empty:
            mdice = df_model.iloc[0]["melhor_mdice_validacao"]
            score = analisar_estabilidade(nome, mdice)
            scores_e3.append({"nome": nome, "score": score, "row": df_model.iloc[0]})
            
    scores_e3 = sorted(scores_e3, key=lambda x: x["score"], reverse=True)
    top_3_execucoes = scores_e3[:3]
    seeds_e4 = [42, 125, 2026]
    
    log_finais = []
    for modelo_dict in top_3_execucoes:
        row = modelo_dict["row"]
        for s in seeds_e4:
            nome_arq_recuperado = row.get("nome_arquitetura", "UNET")
            nome_e4 = formatar_nome_d3(
                nome_arq_recuperado, row["loss"], row["otimizador"], 
                row["scheduler"], row["early_stopper"], row["learning_rate"], 
                row["usou_aumento_dados"], s
            )
            if not experimento_ja_executado(nome_e4):
                rodar_experimento_completo(
                    nome_arquitetura=nome_arq_recuperado, 
                    nome_loss=row["loss"], 
                    nome_otimizador=row["otimizador"],
                    nome_scheduler=row["scheduler"], 
                    nome_early_stopper=row["early_stopper"], 
                    lr=row["learning_rate"],
                    aplicar_aug=row["usou_aumento_dados"], 
                    seed=s,
                    fracao_dados=0.1,
                    num_workers=NUM_WORKERS
                )
            else:
                print(f"[CACHE] Experimento já executado: {nome_e4}")
            log_finais.append(nome_e4)
            
    escrever_log_etapa(4, f"Top 3 configurações treinadas nas seeds 42, 125 e 2026.\nModelos Gerados:\n" + "\n".join(log_finais))
    print("\n[OK] Fluxo de desafios completado integralmente.")
