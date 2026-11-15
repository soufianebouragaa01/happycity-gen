"""Briques du générateur : modulation (FiLM / AdaIN) et attention guidée par R."""

from __future__ import annotations

import torch
from torch import nn


class Modulation(nn.Module):
    """Modulation affine par canal, prédite à partir d'un vecteur : ``h · (1 + γ) + β``.

    Appliquée après une normalisation d'instance, c'est FiLM (vecteur = condition c) ou
    AdaIN (vecteur = code latent z). Initialisée à l'identité (γ = β = 0), pour que
    l'entraînement démarre comme un U-Net non modulé.
    """

    def __init__(self, in_dim: int, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 2 * channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Module ``h`` (B, C, H, W) par ``vec`` (B, in_dim)."""
        gamma, beta = self.linear(vec).chunk(2, dim=1)
        return h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class Embedding(nn.Module):
    """Petit MLP qui encode un vecteur (c ou z) avant les modulations."""

    def __init__(self, in_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim)
        )
        self.out_dim = out_dim

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Encode ``vec`` (B, in_dim)."""
        return torch.relu(self.net(vec))


class RGuidedAttention(nn.Module):
    """Porte d'attention spatiale sur une connexion de saut du U-Net, guidée par R.

    ``α = σ(ψ(ReLU(W_s · skip + W_g · gate + W_r · R)))`` puis ``skip · α``. Le décodeur
    (``gate``) et la région modifiable R décident ensemble quelles zones de l'encodeur
    transmettre. Le biais de ψ vaut 1 au départ : la porte est plutôt ouverte.
    """

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int | None = None) -> None:
        super().__init__()
        inter = inter_ch or max(skip_ch // 2, 8)
        self.w_skip = nn.Conv2d(skip_ch, inter, 1, bias=False)
        self.w_gate = nn.Conv2d(gate_ch, inter, 1)
        self.w_r = nn.Conv2d(1, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)
        assert self.psi.bias is not None
        nn.init.constant_(self.psi.bias, 1.0)  # porte plutôt ouverte au départ

    def forward(
        self, skip: torch.Tensor, gate: torch.Tensor, region: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Renvoie la connexion de saut pondérée et la carte d'attention (B, 1, H, W)."""
        height, width = int(skip.shape[-2]), int(skip.shape[-1])
        r = nn.functional.adaptive_avg_pool2d(region, (height, width))
        alpha = torch.sigmoid(
            self.psi(torch.relu(self.w_skip(skip) + self.w_gate(gate) + self.w_r(r)))
        )
        return skip * alpha, alpha
