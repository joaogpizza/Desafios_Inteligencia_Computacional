"""
Implementação da U-KAN (Li et al., 2024, arXiv:2406.02918).

Baseada no código oficial (CUHK-AIM-Group/U-KAN, Seg_UKAN/archs.py) e na
KANLinear do efficient-kan, sem dependência de timm.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    Camada KAN: cada conexão (entrada i -> saída j) é uma função 1D aprendível,
    com a forma w_b * silu(x) + w_s * spline(x).

    A spline é uma combinação linear de (grid_size + ordem_spline) B-splines
    fixas, e os coeficientes dessa combinação são os parâmetros treináveis.

    Os cálculos são feitos sempre em float32 (autocast desligado), pois a
    recursão das B-splines faz divisões e comparações que são frágeis em fp16.
    """
    def __init__(
            self,
            features_entrada: int,
            features_saida: int,
            grid_size: int = 5,
            ordem_spline: int = 3,
            escala_ruido: float = 0.1,
            escala_base: float = 1.0,
            escala_spline: float = 1.0,
            faixa_grid: Tuple[float, float] = (-1.0, 1.0)
        ) -> None:
        super().__init__()
        self.features_entrada = features_entrada
        self.features_saida = features_saida
        self.grid_size = grid_size
        self.ordem_spline = ordem_spline
        self.escala_ruido = escala_ruido
        self.escala_base = escala_base
        self.escala_spline = escala_spline

        passo = (faixa_grid[1] - faixa_grid[0]) / grid_size
        grid = (
            torch.arange(-ordem_spline, grid_size + ordem_spline + 1) * passo
            + faixa_grid[0]
        ).expand(features_entrada, -1).contiguous()
        self.register_buffer("grid", grid)

        self.peso_base = nn.Parameter(torch.empty(features_saida, features_entrada))
        self.peso_spline = nn.Parameter(
            torch.empty(features_saida, features_entrada, grid_size + ordem_spline)
        )
        self.escala_individual_spline = nn.Parameter(
            torch.empty(features_saida, features_entrada)
        )
        self.ativacao_base = nn.SiLU()

        self.resetar_parametros()

    def resetar_parametros(self) -> None:
        nn.init.kaiming_uniform_(self.peso_base, a=math.sqrt(5) * self.escala_base)
        with torch.no_grad():
            ruido = (
                (torch.rand(self.grid_size + 1, self.features_entrada, self.features_saida) - 0.5)
                * self.escala_ruido / self.grid_size
            )
            self.peso_spline.data.copy_(
                self._curva_para_coeficientes(
                    self.grid.T[self.ordem_spline:-self.ordem_spline], ruido
                )
            )
            nn.init.kaiming_uniform_(
                self.escala_individual_spline, a=math.sqrt(5) * self.escala_spline
            )

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula as bases B-spline (recursão de Cox-de Boor).

        Entrada: (N, features_entrada).
        Saída: (N, features_entrada, grid_size + ordem_spline).
        """
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.ordem_spline + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def _curva_para_coeficientes(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Acha os coeficientes das B-splines que passam pelos pontos (x, y) (mínimos quadrados)."""
        A = self._b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solucao = torch.linalg.lstsq(A, B).solution
        return solucao.permute(2, 0, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype_original = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            saida_base = F.linear(self.ativacao_base(x), self.peso_base)
            peso_spline_escalado = self.peso_spline * self.escala_individual_spline.unsqueeze(-1)
            saida_spline = F.linear(
                self._b_splines(x).view(x.size(0), -1),
                peso_spline_escalado.view(self.features_saida, -1)
            )
            saida = saida_base + saida_spline
        return saida.to(dtype_original)


class ConvDepthwiseBnRelu(nn.Module):
    """Convolução depthwise 3x3 + BatchNorm + ReLU, sobre tokens (B, N, C)."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


class CamadaKAN(nn.Module):
    """
    Três camadas KAN em sequência, cada uma seguida de DwConv + BN + ReLU.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.kan1 = KANLinear(dim, dim)
        self.kan2 = KANLinear(dim, dim)
        self.kan3 = KANLinear(dim, dim)

        self.dw1 = ConvDepthwiseBnRelu(dim)
        self.dw2 = ConvDepthwiseBnRelu(dim)
        self.dw3 = ConvDepthwiseBnRelu(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape

        x = self.kan1(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dw1(x, H, W)
        x = self.kan2(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dw2(x, H, W)
        x = self.kan3(x.reshape(B * N, C)).reshape(B, N, C)
        x = self.dw3(x, H, W)
        return x


class BlocoKAN(nn.Module):
    """
    Bloco KAN tokenizado: LayerNorm -> CamadaKAN, com conexão residual.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norma = nn.LayerNorm(dim)
        self.camada = CamadaKAN(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return x + self.camada(self.norma(x), H, W)


class Tokenizacao(nn.Module):
    """
    Tokeniza o mapa de características: convolução 3x3 com stride 2 (reduz a
    resolução pela metade), vira uma sequência de tokens (B, N, C) e aplica
    LayerNorm.
    """
    def __init__(self, canais_entrada: int, dim_embedding: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(canais_entrada, dim_embedding, kernel_size=3, stride=2, padding=1)
        self.norma = nn.LayerNorm(dim_embedding)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norma(x), H, W


class BlocoConvolucional(nn.Module):
    """
    O bloco convolucional do encoder (igual ao das outras arquiteturas).
    """
    def __init__(self, canais_entrada: int, canais_saida: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True),
            nn.Conv2d(canais_saida, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BlocoConvolucionalDecoder(nn.Module):
    """
    O bloco convolucional do decoder da U-KAN: a primeira convolução mantém
    os canais e a segunda os altera.
    """
    def __init__(self, canais_entrada: int, canais_saida: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(canais_entrada, canais_entrada, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_entrada),
            nn.ReLU(inplace=True),
            nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(canais_saida),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _tokens_para_mapa(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, N, C) -> (B, C, H, W)"""
    B = x.shape[0]
    return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


def _mapa_para_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """(B, C, H, W) -> (B, N, C)"""
    _, _, H, W = x.shape
    return x.flatten(2).transpose(1, 2), H, W


def _subir(x: torch.Tensor) -> torch.Tensor:
    """Dobra a resolução (interpolação bilinear) e aplica ReLU."""
    return F.relu(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class UKAN(nn.Module):
    """
    A U-KAN usada.

    Encoder: 3 blocos convolucionais (H/2, H/4, H/8) e 2 estágios com blocos
    KAN tokenizados (H/16 e H/32).
    Decoder: 2 blocos KAN tokenizados (H/16 e H/8) e blocos convolucionais.
    As skip connections são somas, como no código oficial.

    A altura e a largura da entrada devem ser divisíveis por 32.

    dims_embedding = [128, 160, 256] é a configuração "U-KAN" do artigo
    (~6,4 M de parâmetros). [256, 320, 512] é a "U-KAN-L" (~25 M) e é o
    valor padrão do repositório oficial.
    """
    def __init__(
            self,
            canais_entrada: int = 3,
            classes_saida: int = 1,
            dims_embedding: List[int] = [128, 160, 256]
        ) -> None:
        super().__init__()

        d0, d1, d2 = dims_embedding

        self.encoder1 = BlocoConvolucional(canais_entrada, d0 // 8)
        self.encoder2 = BlocoConvolucional(d0 // 8, d0 // 4)
        self.encoder3 = BlocoConvolucional(d0 // 4, d0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.tokenizacao3 = Tokenizacao(d0, d1)
        self.bloco_kan_enc1 = BlocoKAN(d1)
        self.norma_enc1 = nn.LayerNorm(d1)

        self.tokenizacao4 = Tokenizacao(d1, d2)
        self.bloco_kan_enc2 = BlocoKAN(d2)
        self.norma_enc2 = nn.LayerNorm(d2)

        self.decoder1 = BlocoConvolucionalDecoder(d2, d1)
        self.bloco_kan_dec1 = BlocoKAN(d1)
        self.norma_dec1 = nn.LayerNorm(d1)

        self.decoder2 = BlocoConvolucionalDecoder(d1, d0)
        self.bloco_kan_dec2 = BlocoKAN(d0)
        self.norma_dec2 = nn.LayerNorm(d0)

        self.decoder3 = BlocoConvolucionalDecoder(d0, d0 // 4)
        self.decoder4 = BlocoConvolucionalDecoder(d0 // 4, d0 // 8)
        self.decoder5 = BlocoConvolucionalDecoder(d0 // 8, d0 // 8)

        self.saida = nn.Conv2d(d0 // 8, classes_saida, kernel_size=1)

        self._inicializar_pesos_tokens()

    def _inicializar_pesos_tokens(self) -> None:
        """Inicialização do código oficial para as partes tokenizadas."""
        modulos = [
            self.tokenizacao3, self.tokenizacao4,
            self.bloco_kan_enc1, self.bloco_kan_enc2,
            self.bloco_kan_dec1, self.bloco_kan_dec2,
        ]
        for modulo in modulos:
            for m in modulo.modules():
                if isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] % 32 != 0 or x.shape[3] % 32 != 0:
            raise ValueError(
                f"A U-KAN exige altura e largura divisíveis por 32, mas recebeu {tuple(x.shape[2:])}"
            )

        # Encoder convolucional
        t1 = self.maxpool(self.encoder1(x))
        t2 = self.maxpool(self.encoder2(t1))
        t3 = self.maxpool(self.encoder3(t2))

        # Encoder com blocos KAN
        out, H, W = self.tokenizacao3(t3)
        out = self.bloco_kan_enc1(out, H, W)
        t4 = _tokens_para_mapa(self.norma_enc1(out), H, W)

        # Gargalo
        out, H, W = self.tokenizacao4(t4)
        out = self.bloco_kan_enc2(out, H, W)
        out = _tokens_para_mapa(self.norma_enc2(out), H, W)

        # Decoder com blocos KAN
        out = _subir(self.decoder1(out)) + t4
        out, H, W = _mapa_para_tokens(out)
        out = self.bloco_kan_dec1(out, H, W)
        out = _tokens_para_mapa(self.norma_dec1(out), H, W)

        out = _subir(self.decoder2(out)) + t3
        out, H, W = _mapa_para_tokens(out)
        out = self.bloco_kan_dec2(out, H, W)
        out = _tokens_para_mapa(self.norma_dec2(out), H, W)

        # Decoder convolucional
        out = _subir(self.decoder3(out)) + t2
        out = _subir(self.decoder4(out)) + t1
        out = _subir(self.decoder5(out))

        return self.saida(out)
