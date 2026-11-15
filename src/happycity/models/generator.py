"""Générateur conditionnel U-Net : G(M, R, c = (b, a), z) → carte d'ajouts d'espaces verts.

Une seule architecture. Seule la variable étudiée change d'une variante à l'autre :

* **A** (``cond_mode=concat``) : b et a ajoutés comme deux canaux constants en entrée ;
* **B** (``cond_mode=film``) : c encodé par un MLP, puis modulation FiLM de chaque bloc
  (encodeur et décodeur) ;
* **C** = B + attention spatiale guidée par R sur les connexions de saut + **projection**.
  En entraînement, la sortie est multipliée par R (la partie différentiable de
  l'opérateur) ; en inférence, l'opérateur déterministe complet est appliqué
  (:mod:`happycity.projection`).

Le bruit z est injecté de la même façon dans toutes les variantes (``z_injection``) :
AdaIN dans le décodeur par défaut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from happycity.models.blocks import Embedding, Modulation, RGuidedAttention

COND_DIM = 2  # (b, a)
BUDGET_SCALE = 20.0  # b = 5 % → 1


def condition_vector(b_frac: torch.Tensor, a_target: torch.Tensor) -> torch.Tensor:
    """Vecteur de condition c = (b normalisé, a), ``(B, 2)``."""
    return torch.stack([b_frac * BUDGET_SCALE, a_target], dim=1).float()


@dataclass(frozen=True)
class GeneratorSpec:
    """Hyperparamètres du générateur (issus de ``cfg.model``)."""

    in_channels: int  # M + R
    cond_mode: str = "film"  # concat | film | adain
    attention: bool = True
    projection: bool = True
    ngf: int = 32
    n_downs: int = 5
    dropout: float = 0.5
    z_dim: int = 8
    z_injection: str = "adain"  # adain | concat | none


@dataclass
class GeneratorOutput:
    """Sortie du générateur."""

    logits: torch.Tensor  # (B, 1, H, W) avant projection
    prob: torch.Tensor  # (B, 1, H, W) après la projection différentiable (C) ou brute
    attention: list[torch.Tensor]  # cartes d'attention (vide sans attention)


class _Down(nn.Module):
    def __init__(self, cin: int, cout: int, norm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 4, 2, 1, bias=not norm)
        self.norm = nn.InstanceNorm2d(cout) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class _Up(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)
        self.norm = nn.InstanceNorm2d(cout)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class UNetGenerator(nn.Module):
    """U-Net à ``n_downs`` sous-échantillonnages (128 → 4 px pour 5)."""

    def __init__(self, spec: GeneratorSpec) -> None:
        super().__init__()
        self.spec = spec
        n = spec.n_downs
        ch = [spec.ngf * min(2**i, 8) for i in range(n)]  # sorties de l'encodeur
        extra = (COND_DIM if spec.cond_mode == "concat" else 0) + (
            spec.z_dim if spec.z_injection == "concat" else 0
        )
        self.downs = nn.ModuleList(
            [_Down(spec.in_channels + extra, ch[0], norm=False)]
            + [_Down(ch[i - 1], ch[i], norm=i < n - 1) for i in range(1, n)]
        )
        # Décodeur : du goulot vers la résolution 1/2, puis couche de sortie.
        up_out = [ch[i] for i in range(n - 2, -1, -1)]  # n − 1 blocs
        up_in = [ch[-1]] + [2 * c for c in up_out[:-1]]
        self.ups = nn.ModuleList(
            [
                _Up(i, o, spec.dropout if k < 2 else 0.0)  # dropout dans les 2 premiers blocs
                for k, (i, o) in enumerate(zip(up_in, up_out, strict=True))
            ]
        )
        self.out = nn.ConvTranspose2d(2 * up_out[-1], 1, 4, 2, 1)

        self.c_embed = Embedding(COND_DIM) if spec.cond_mode in ("film", "adain") else None
        enc_mod = spec.cond_mode == "film"
        self.c_mod_down = nn.ModuleList([Modulation(64, c) for c in ch[1:]] if enc_mod else [])
        self.c_mod_up = nn.ModuleList(
            [Modulation(64, c) for c in up_out] if self.c_embed is not None else []
        )
        self.z_embed = Embedding(spec.z_dim) if spec.z_injection == "adain" else None
        self.z_mod_up = nn.ModuleList(
            [Modulation(64, c) for c in up_out] if self.z_embed is not None else []
        )
        # Attention : une porte par connexion de saut (encodeur niveau n−2 … 0).
        self.gates = nn.ModuleList(
            [RGuidedAttention(o, o) for o in up_out] if spec.attention else []
        )

    def sample_z(self, batch: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Code latent z ~ N(0, I)."""
        device = next(self.parameters()).device
        if self.spec.z_injection == "none" or self.spec.z_dim == 0:
            return torch.zeros(batch, max(self.spec.z_dim, 1), device=device)
        z = torch.randn(batch, self.spec.z_dim, generator=generator, device="cpu")
        return z.to(device)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, z: torch.Tensor) -> GeneratorOutput:
        """``x`` = (M, R) ``(B, C, H, W)`` ; ``cond`` = c ``(B, 2)`` ; ``z`` ``(B, z_dim)``."""
        spec = self.spec
        region = x[:, -1:]  # R est le dernier canal
        h = x
        size = x.shape[-2:]
        if spec.cond_mode == "concat":
            h = torch.cat([h, cond[:, :, None, None].expand(-1, -1, *size)], dim=1)
        if spec.z_injection == "concat":
            h = torch.cat([h, z[:, :, None, None].expand(-1, -1, *size)], dim=1)
        c_vec = self.c_embed(cond) if self.c_embed is not None else None
        z_vec = self.z_embed(z) if self.z_embed is not None else None

        skips = []
        for i, down in enumerate(self.downs):
            h = down(h)
            if i > 0 and self.c_mod_down and c_vec is not None:
                h = self.c_mod_down[i - 1](h, c_vec)
            h = nn.functional.leaky_relu(h, 0.2)
            skips.append(h)
        skips.pop()  # le goulot n'a pas de connexion de saut

        attention = []
        for k, up in enumerate(self.ups):
            h = up(h)
            if self.c_mod_up and c_vec is not None:
                h = self.c_mod_up[k](h, c_vec)
            if self.z_mod_up and z_vec is not None:
                h = self.z_mod_up[k](h, z_vec)
            h = cast(_Up, up).dropout(torch.relu(h))
            skip = skips.pop()
            if self.gates:
                skip, alpha = self.gates[k](skip, h, region)
                attention.append(alpha)
            h = torch.cat([h, skip], dim=1)
        logits = self.out(h)
        prob = torch.sigmoid(logits)
        if spec.projection:
            prob = prob * region  # partie différentiable de la projection : rien hors de R
        return GeneratorOutput(logits=logits, prob=prob, attention=attention)
