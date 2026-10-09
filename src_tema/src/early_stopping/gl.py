class EarlyStoppingGL:
    """Para se a Perda de Generalização (GL) exceder o limite alpha em porcentagem."""
    def __init__(self, alpha=5.0):
        self.alpha = alpha  # Limite em % (ex: 5% acima da melhor perda)
        self.melhor_perda = float('inf')
        self.parada_antecipada = False

    def __call__(self, perda_val, perda_treino=None, metrica_val=None):
        eh_melhor = False
        
        if perda_val < self.melhor_perda:
            self.melhor_perda = perda_val
            eh_melhor = True
            
        gl = 100 * ((perda_val / self.melhor_perda) - 1)
        
        if gl > self.alpha:
            self.parada_antecipada = True

        return eh_melhor