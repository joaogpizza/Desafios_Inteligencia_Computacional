import torch
import torch.nn as nn
from typing import List

class BlocoConvolucional(nn.Module):
    """
    O bloco convolucional padrão.
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

class UNet(nn.Module):
    """
    U-Net Padrão.
    """
    def __init__(
            self,
            canais_entrada: int = 3,
            classes_saida: int = 1,
            filtros: List[int] = [48, 96, 192, 384, 768] 
        ) -> None:
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder1 = BlocoConvolucional(canais_entrada, filtros[0])
        self.encoder2 = BlocoConvolucional(filtros[0], filtros[1])
        self.encoder3 = BlocoConvolucional(filtros[1], filtros[2])
        self.encoder4 = BlocoConvolucional(filtros[2], filtros[3])
        
        self.gargalo = BlocoConvolucional(filtros[3], filtros[4])
        
        self.upconv4 = nn.ConvTranspose2d(filtros[4], filtros[3], kernel_size=2, stride=2)
        self.decoder4 = BlocoConvolucional(filtros[4], filtros[3]) 
        
        self.upconv3 = nn.ConvTranspose2d(filtros[3], filtros[2], kernel_size=2, stride=2)
        self.decoder3 = BlocoConvolucional(filtros[3], filtros[2])
        
        self.upconv2 = nn.ConvTranspose2d(filtros[2], filtros[1], kernel_size=2, stride=2)
        self.decoder2 = BlocoConvolucional(filtros[2], filtros[1])
        
        self.upconv1 = nn.ConvTranspose2d(filtros[1], filtros[0], kernel_size=2, stride=2)
        self.decoder1 = BlocoConvolucional(filtros[1], filtros[0])
        
        self.saida = nn.Conv2d(filtros[0], classes_saida, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        e4 = self.encoder4(self.maxpool(e3))
        
        b = self.gargalo(self.maxpool(e4))
        
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)
        
        saida = self.saida(d1)
        return saida
