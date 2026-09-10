"""
Integrantes do grupo:
João Geiger Piza - 12411BCC035
Ruan Pablo Lima Coelho - 12411BCC002
"""
import math
import time
import torch

DIM_MAX = 64
SEMENTES = (0, 1, 2)

# =============================================================================
# >>> SUA SUBMISSÃO — edite apenas esta classe <<<
# =============================================================================
class Submissao:
    DIM_MAX = DIM_MAX

    def fit(self, X: torch.Tensor) -> None:
        X_duplo = X.double()
        self.media = X_duplo.mean(dim=0)
        self.desvio = X_duplo.std(dim=0) + 1e-8

        Xs = self._padronizar(X_duplo)
        d = X.shape[1]

        num_explicitas = 13 if d == 2 else (2 * d + 1)
        num_fourier = self.DIM_MAX - num_explicitas

        sigma_base = self._estimar_sigma_base(Xs)
        self._gerar_pesos_fourier(d, num_fourier, sigma_base)

    def phi(self, X: torch.Tensor) -> torch.Tensor:
        X_duplo = X.double()
        Xs = self._padronizar(X_duplo)

        explicitas = self._extrair_caracteristicas_explicitas(Xs)
        num_fourier = self.DIM_MAX - explicitas.shape[1]

        fourier = self._gerar_caracteristicas_fourier(Xs, num_fourier)

        resultado = torch.cat([explicitas, fourier], dim=1)
        return resultado.float()

    def _padronizar(self, X_duplo: torch.Tensor) -> torch.Tensor:
        """Aplica a padronização z-score com os parâmetros calculados no fit."""
        return (X_duplo - self.media) / self.desvio

    def _estimar_sigma_base(self, Xs: torch.Tensor) -> float:
        """Estima a distância mediana entre os pontos para calibrar a escala gaussiana (sigma)."""
        amostra = Xs[: min(300, len(Xs))]
        distancias_quadradas = torch.cdist(amostra, amostra) ** 2
        valores_positivos = distancias_quadradas[distancias_quadradas > 0]
        mediana = valores_positivos.median().item() if len(valores_positivos) > 0 else 1.0
        return math.sqrt(mediana / 2.0) + 1e-8 # evitar div por 0

    def _gerar_pesos_fourier(self, d: int, num_fourier: int, sigma_base: float) -> None:
        """Gera a matriz W e o vetor b com gerador determinístico em múltiplas escalas."""
        gerador = torch.Generator().manual_seed(2026)
        escalas = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        colunas_por_escala = math.ceil(num_fourier / len(escalas))

        blocos_W = []
        for escala in escalas:
            sigma_atual = sigma_base / escala
            w_escala = torch.randn(d, colunas_por_escala, generator=gerador, dtype=torch.float64) / sigma_atual
            blocos_W.append(w_escala)

        self.matriz_W = torch.cat(blocos_W, dim=1)[:, :num_fourier]
        self.vetor_b = torch.rand(num_fourier, generator=gerador, dtype=torch.float64) * 2 * math.pi

    def _extrair_caracteristicas_explicitas(self, Xs: torch.Tensor) -> torch.Tensor:
        """Extrai termos polinomiais, radiais e harmônicos em coordenadas polares."""
        d = Xs.shape[1]
        raio_quadrado = (Xs ** 2).sum(dim=1, keepdim=True)
        caracteristicas = [Xs, Xs ** 2, raio_quadrado]

        if d == 2:
            produto_cruzado = Xs[:, 0:1] * Xs[:, 1:2]
            angulo_theta = torch.atan2(Xs[:, 1:2], Xs[:, 0:1])
            caracteristicas.extend([
                produto_cruzado,
                angulo_theta,
                torch.sin(angulo_theta),
                torch.cos(angulo_theta),
                torch.sin(2 * angulo_theta),
                torch.cos(2 * angulo_theta),
                torch.sin(3 * angulo_theta),
                torch.cos(3 * angulo_theta)
            ])

        return torch.cat(caracteristicas, dim=1)

    def _gerar_caracteristicas_fourier(self, Xs: torch.Tensor, num_fourier: int) -> torch.Tensor:
        """Aplica a projeção de Fourier Aleatória (RFF) multi-escala."""
        projecao = Xs @ self.matriz_W[:, :num_fourier] + self.vetor_b[:num_fourier]
        return math.sqrt(2.0 / num_fourier) * torch.cos(projecao)

# =============================================================================
# Harness (não edite daqui para baixo)
# =============================================================================
def _luas(n, g):
    t = torch.rand(n // 2, generator=g) * math.pi
    X = torch.cat([torch.stack([torch.cos(t), torch.sin(t)], 1),
                   torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)])
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _circulos(n, g):
    t = torch.rand(n, generator=g) * 2 * math.pi
    r = torch.where(torch.arange(n) < n // 2, 1.0, 0.45)
    X = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    return X + 0.08 * torch.randn(n, 2, generator=g), (torch.arange(n) >= n // 2).float()

def _xor(n, g):
    X = torch.rand(n, 2, generator=g) * 2 - 1
    y = (X[:, 0] * X[:, 1] < 0).float()
    return X + 0.15 * torch.randn(n, 2, generator=g), y

def _espiral(n, g):
    t = torch.sqrt(torch.rand(n // 2, generator=g)) * 3 * math.pi
    a = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 10
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
    return torch.cat([a, -a]) + 0.05 * torch.randn(n, 2, generator=g), y

def _esfera(n, g, d=10):
    X = torch.randn(n, d, generator=g)
    r2 = (X ** 2).sum(1)
    return X, (r2 > r2.median()).float()

TAREFAS = {"xor": lambda g: _xor(600, g), "duas_luas": lambda g: _luas(600, g),
           "circulos": lambda g: _circulos(600, g), "espiral": lambda g: _espiral(800, g),
           "esfera_10d": lambda g: _esfera(800, g)}


@torch.no_grad()
def perceptron_pocket(Z, y, epocas=50, eta=1.0, semente=0):
    """Regra de Rosenblatt (w <- w + eta*y*z nos erros) + pocket: guarda o melhor w."""
    Zb = torch.cat([Z, torch.ones(len(Z), 1)], 1)     # viés embutido
    yb = 2 * y - 1
    w = torch.zeros(Zb.shape[1]); melhor_w, melhor_acc = w.clone(), -1.0
    g = torch.Generator().manual_seed(semente)
    for _ in range(epocas):
        for i in torch.randperm(len(Zb), generator=g).tolist():
            if yb[i] * (Zb[i] @ w) <= 0:
                w += eta * yb[i] * Zb[i]
        acc = ((Zb @ w) * yb > 0).float().mean().item()
        if acc > melhor_acc:
            melhor_acc, melhor_w = acc, w.clone()
    return melhor_w


def acuracia_balanceada(y, yhat):
    return torch.stack([(yhat[y == c] == c).float().mean() for c in y.unique()]).mean().item()


@torch.no_grad()
def rodar(sub, gerador, semente, checar=True):
    g = torch.Generator().manual_seed(semente)
    X, y = gerador(g)
    idx = torch.randperm(len(X), generator=g); ntr = int(0.6 * len(X))
    Xtr, ytr, Xte, yte = X[idx[:ntr]], y[idx[:ntr]], X[idx[ntr:]], y[idx[ntr:]]

    torch.manual_seed(semente)
    sub.fit(Xtr)                                       # nunca recebe ytr
    t0 = time.perf_counter(); Ztr = sub.phi(Xtr); dt = time.perf_counter() - t0
    Zte = sub.phi(Xte)
    if checar:                                         # regras do desafio
        d, dl = Xtr.shape[1], Ztr.shape[1]
        assert Ztr.ndim == 2 and Zte.shape[1] == dl, "phi deve devolver (n, d')"
        assert d < dl <= DIM_MAX, f"exige d < d' <= {DIM_MAX}; recebi d={d}, d'={dl}"
        assert torch.isfinite(Ztr).all() and torch.isfinite(Zte).all(), "NaN/Inf na saída de phi"
        assert torch.allclose(sub.phi(Xtr[:20]), Ztr[:20]), "phi não é determinística"
        assert dt * (10_000 / len(Xtr)) < 2.0, "phi lenta demais (limite: 10^4 pontos em 2 s)"

    w = perceptron_pocket(Ztr, ytr, semente=semente)
    yhat = (torch.cat([Zte, torch.ones(len(Zte), 1)], 1) @ w > 0).float()
    return acuracia_balanceada(yte, yhat)


class _Identidade:                       # baseline (viola d < d', mas é só o ponto zero da escala)
    def fit(self, X): pass
    def phi(self, X): return X

class _RFF:                              # referência: random Fourier features, d' = 64
    def fit(self, X):
        g = torch.Generator().manual_seed(0)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        d2 = torch.cdist(Xs[:300], Xs[:300]) ** 2
        sigma = math.sqrt(d2[d2 > 0].median().item() / 2)
        self.W = torch.randn(X.shape[1], DIM_MAX, generator=g) / sigma
        self.b = torch.rand(DIM_MAX, generator=g) * 2 * math.pi
    def phi(self, X):
        return math.sqrt(2 / DIM_MAX) * torch.cos(((X - self.mu) / self.sd) @ self.W + self.b)


def _mediana(cls, gerador, checar=True):
    vals = [rodar(cls(), gerador, sem, checar) for sem in SEMENTES]
    return float(torch.tensor(vals).median())


def avaliar():
    print(f"{'tarefa':<12}{'baseline':>10}{'referência':>12}{'você':>8}{'s_t':>7}")
    s = []
    for nome, gen in TAREFAS.items():
        b = _mediana(_Identidade, gen, checar=False)
        r = max(_mediana(_RFF, gen, checar=False), b + 1e-3)
        try:
            m = _mediana(Submissao, gen); erro = ""
        except AssertionError as e:
            m, erro = b, f"   <- {e}"
        st = min(max((m - b) / (r - b), 0.0), 1.25); s.append(st)
        print(f"{nome:<12}{b:>10.3f}{r:>12.3f}{m:>8.3f}{st:>7.2f}{erro}")
    S = 100 * (0.7 * sum(s) / len(s) + 0.3 * min(s))
    print(f"\nESCORE S = {S:.1f}   (0 = baseline, 100 = referência, até 125 com bônus)")
    return S


if __name__ == "__main__":
    avaliar()