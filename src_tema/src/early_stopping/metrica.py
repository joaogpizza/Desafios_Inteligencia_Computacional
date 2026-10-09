class EarlyStoppingMetrica:
    """Foca apenas em maximizar a métrica de tarefa (ex: Dice/IoU), ignorando a Loss."""
    def __init__(self, paciencia=10, delta_minimo=0.001):
        self.paciencia = paciencia
        self.delta_minimo = delta_minimo
        self.contador = 0
        self.melhor_metrica = None
        self.parada_antecipada = False

    def __call__(self, perda_val=None, perda_treino=None, metrica_val=None):
        if metrica_val is None:
            raise ValueError("EarlyStoppingMetrica exige o parâmetro 'metrica_val' (ex: Dice).")

        eh_melhor = False

        if self.melhor_metrica is None:
            self.melhor_metrica = metrica_val
            eh_melhor = True
        elif metrica_val < self.melhor_metrica + self.delta_minimo:
            self.contador += 1
            if self.contador >= self.paciencia:
                self.parada_antecipada = True
        else:
            self.melhor_metrica = metrica_val
            eh_melhor = True
            self.contador = 0

        return eh_melhor