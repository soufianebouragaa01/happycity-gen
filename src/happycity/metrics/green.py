"""Couverture végétale V = A_vert / A_zone et contrôle du budget foncier.

Seuls les espaces verts publics ou explicitement accessibles sont comptés (la carte
``green_fraction`` ne contient que ceux-là). La zone est la tuile entière, identique avant
et après aménagement (même projection métrique, même périmètre).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BudgetExceededError(ValueError):
    """Surface verte ajoutée supérieure au budget b."""


@dataclass(frozen=True)
class GreenCover:
    """Surface verte et couverture V d'une zone."""

    green_m2: float
    zone_m2: float

    @property
    def V(self) -> float:  # noqa: N802 - notation du cahier des charges
        """Couverture végétale V = A_vert / A_zone."""
        return self.green_m2 / self.zone_m2


def green_cover(
    green_fraction: np.ndarray, m_per_px: float, added: np.ndarray | None = None
) -> GreenCover:
    """Surface verte d'une tuile, éventuellement après ajout d'un masque ``added``.

    Args:
        green_fraction: fraction de chaque pixel couverte par un espace vert compté (0–1).
        m_per_px: résolution.
        added: masque booléen (ou fraction) des nouvelles surfaces vertes.

    Un pixel déjà partiellement vert n'est pas compté deux fois : ``max(existant, ajout)``.
    """
    cover = np.clip(np.asarray(green_fraction, dtype=float), 0, 1)
    if added is not None:
        if added.shape != cover.shape:
            raise ValueError(f"masque {added.shape} ≠ tuile {cover.shape}")
        cover = np.maximum(cover, np.clip(added.astype(float), 0, 1))
    pixel_m2 = m_per_px**2
    return GreenCover(green_m2=float(cover.sum()) * pixel_m2, zone_m2=cover.size * pixel_m2)


def added_green_m2(before: GreenCover, after: GreenCover) -> float:
    """ΔA_vert entre deux états d'une même zone."""
    if before.zone_m2 != after.zone_m2:
        raise ValueError("périmètres différents avant/après")
    return after.green_m2 - before.green_m2


def check_budget(delta_m2: float, budget_m2: float, tol_m2: float = 1e-6) -> None:
    """Lève :class:`BudgetExceededError` si ΔA_vert > b."""
    if budget_m2 < 0:
        raise ValueError("budget négatif")
    if delta_m2 > budget_m2 + tol_m2:
        raise BudgetExceededError(f"ΔA_vert = {delta_m2:.1f} m² > b = {budget_m2:.1f} m²")
