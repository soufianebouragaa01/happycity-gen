"""Pertes du cGAN : adversariale, reconstruction, violations, condition, diversité."""

from __future__ import annotations

import torch

_P_MAX = 1.0 - 1e-4


def soft_a400(
    prob: torch.Tensor,
    pmap: torch.Tensor,
    patterns: torch.Tensor,
    n_origins: torch.Tensor,
    n_base: torch.Tensor,
) -> torch.Tensor:
    """A400 différentiable d'une carte de probabilités, par lot.

    Une origine non desservie le devient avec la probabilité ``1 − Π(1 − pᵢ)``, le produit
    portant sur les pixels de R qui la desservent. Les pixels partagent des **motifs** de
    desserte précalculés avec des distances réseau (:func:`happycity.metrics.pixel_service`).
    Le calcul se fait en ``float32``, hors autocast.

    Args:
        prob: ``(B, 1, H, W)``.
        pmap: ``(B, H, W)`` indice du motif de chaque pixel (−1 : aucun).
        patterns: ``(B, K, O)`` origines desservies par chaque motif (zéros de remplissage).
        n_origins: ``(B,)`` nombre total d'origines de la tuile.
        n_base: ``(B,)`` origines déjà desservies.

    Returns:
        ``(B,)`` A400 estimé.
    """
    p = prob.float().flatten(1).clamp(0.0, _P_MAX)
    log_q = torch.log1p(-p)
    idx = pmap.flatten(1) + 1  # case 0 : pixels sans motif
    sums = torch.zeros(p.shape[0], patterns.shape[1] + 1, device=p.device)
    sums = sums.scatter_add(1, idx, log_q)[:, 1:]
    new = 1.0 - torch.exp(torch.bmm(sums[:, None, :], patterns.float()).squeeze(1))
    return (n_base + new.sum(dim=1)) / n_origins.clamp(min=1.0)


def reconstruction(prob: torch.Tensor, target: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """L1 dans R (là où les cibles vivent), moyenne par tuile puis par lot."""
    err = ((prob - target).abs() * region).flatten(1).sum(1)
    return (err / region.flatten(1).sum(1).clamp(min=1.0)).mean()


def violation(prob: torch.Tensor, region: torch.Tensor, budget_px: torch.Tensor) -> torch.Tensor:
    """Masse hors de R + dépassement de budget, en fraction du budget."""
    outside = (prob * (1 - region)).flatten(1).sum(1)
    over = torch.relu(prob.flatten(1).sum(1) - budget_px)
    return ((outside + over) / budget_px.clamp(min=1.0)).mean()


MS_EPS = 0.05  # borne la perte à 1 / MS_EPS = 20 quand les sorties sont identiques


def mode_seeking(
    p1: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    """MSGAN : maximise |G(z₁) − G(z₂)| / |z₁ − z₂| (on minimise l'inverse, borné).

    L'écart entre sorties est moyenné **dans R** : les ajouts n'occupent que 1 à 5 % de la
    tuile, et une moyenne sur toute l'image rendrait le rapport minuscule même pour des
    propositions entièrement différentes. ``MS_EPS`` évite une perte infinie au départ,
    quand la modulation par z est encore l'identité.
    """
    diff = ((p1 - p2).abs() * region).flatten(1).sum(1) / region.flatten(1).sum(1).clamp(min=1.0)
    ratio = diff.mean() / (z1 - z2).abs().mean().clamp(min=1e-6)
    return 1.0 / (ratio + MS_EPS)


def lsgan(scores: torch.Tensor, real: bool) -> torch.Tensor:
    """Perte adversariale moindres carrés (LSGAN)."""
    return ((scores - (1.0 if real else 0.0)) ** 2).mean()
