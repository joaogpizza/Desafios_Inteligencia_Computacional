class EarlyStoppingPadrao:
    """Para o treinamento baseado em paciência absoluta e delta mínimo na Loss."""
    def __init__(self, paciencia=7, delta_minimo=0.0):
        self.paciencia = paciencia
        self.delta_minimo = delta_minimo
        self.contador = 0
        self.melhor_perda = None
        self.parada_antecipada = False

    def __call__(self, perda_val, perda_treino=None, metrica_val=None):
        eh_melhor = False

        if self.melhor_perda is None:
            self.melhor_perda = perda_val
            eh_melhor = True
        elif perda_val > self.melhor_perda - self.delta_minimo:
            self.contador += 1
            if self.contador >= self.paciencia:
                self.parada_antecipada = True
        else:
            self.melhor_perda = perda_val
            eh_melhor = True
            self.contador = 0

        return eh_melhor