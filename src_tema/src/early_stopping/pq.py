class EarlyStoppingPQ:
    """Avalia a relação entre o Generalization Loss (GL) e o Progresso de Treino (Pk)."""
    def __init__(self, alpha=0.1, k=5):
        self.alpha = alpha  # Tolerância para GL/Pk
        self.k = k          # Janela de épocas para medir progresso de treino
        self.melhor_perda = float('inf')
        self.perdas_treino = []
        self.parada_antecipada = False

    def __call__(self, perda_val, perda_treino, metrica_val=None):
        if perda_treino is None:
            raise ValueError("EarlyStoppingPQ exige o parâmetro 'perda_treino'.")

        self.perdas_treino.append(perda_treino)
        eh_melhor = False
        
        if perda_val < self.melhor_perda:
            self.melhor_perda = perda_val
            eh_melhor = True

        if len(self.perdas_treino) < self.k:
            return eh_melhor

        gl = 100 * ((perda_val / self.melhor_perda) - 1)

        janela = self.perdas_treino[-self.k:]
        pk = 100 * ((sum(janela) / (self.k * min(janela))) - 1)

        if pk < 0.0001: 
            pk = 0.0001
            
        pq = gl / pk

        if pq > self.alpha:
            self.parada_antecipada = True

        return eh_melhor