"""Discriminateur PatchGAN conditionnel et encodeur latent (régression de z, BicycleGAN)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from happycity.models.generator import COND_DIM


class PatchDiscriminator(nn.Module):
    """PatchGAN 70×70 : juge des patchs de (M, R, c, masque d'ajouts).

    Identique pour toutes les variantes. Il voit la condition c, ce qui lui permet de
    pénaliser une sortie incompatible avec (b, a). Normalisation spectrale pour la stabilité.
    """

    def __init__(self, in_channels: int, ndf: int = 32) -> None:
        super().__init__()
        cin = in_channels + COND_DIM + 1

        def block(i: int, o: int, stride: int) -> list[nn.Module]:
            return [spectral_norm(nn.Conv2d(i, o, 4, stride, 1)), nn.LeakyReLU(0.2, inplace=True)]

        self.net = nn.Sequential(
            *block(cin, ndf, 2),
            *block(ndf, ndf * 2, 2),
            *block(ndf * 2, ndf * 4, 2),
            *block(ndf * 4, ndf * 8, 1),
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 1)),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Score par patch ``(B, 1, h, w)``."""
        planes = cond[:, :, None, None].expand(-1, -1, *x.shape[-2:])
        return self.net(torch.cat([x, planes, mask], dim=1))


class LatentEncoder(nn.Module):
    """Prédit z à partir de (M, R, sortie) : la sortie doit « garder la trace » de z.

    Perte de régression latente ``|E(G(x, z)) − z|`` (cLR-GAN / BicycleGAN) : empêche le
    générateur d'ignorer z, donc favorise des propositions différentes.
    """

    def __init__(self, in_channels: int, z_dim: int, nef: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cin = in_channels + 1
        for cout in (nef, nef * 2, nef * 4, nef * 4):
            layers += [nn.Conv2d(cin, cout, 4, 2, 1), nn.InstanceNorm2d(cout)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(nef * 4, z_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ẑ ``(B, z_dim)``."""
        h = self.features(torch.cat([x, mask], dim=1))
        return self.head(h.mean(dim=(2, 3)))
