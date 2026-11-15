"""Équité spatiale E = 1 − (max_q A400^(q) − min_q A400^(q)) sur des secteurs fixes.

Les secteurs sont une grille ``k × k`` fixe de la tuile (2 × 2 = quadrants de 512 m).
Un secteur n'est retenu que s'il contient au moins ``min_origins`` origines ; s'il reste
moins de deux secteurs, E est indéfini (NaN). E doit toujours être lu AVEC A400 : une
égalité parfaite à zéro accès (E = 1, A400 = 0) n'est pas une réussite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sector_of(rows: np.ndarray, cols: np.ndarray, size_px: int, per_side: int) -> np.ndarray:
    """Indice de secteur (ligne de secteurs × k + colonne) de pixels ``(row, col)``."""
    step = size_px / per_side
    sr = np.clip((np.asarray(rows) // step).astype(int), 0, per_side - 1)
    sc = np.clip((np.asarray(cols) // step).astype(int), 0, per_side - 1)
    return np.asarray(sr * per_side + sc, dtype=np.int64)


@dataclass(frozen=True)
class EquityResult:
    """Équité et détail par secteur."""

    E: float  # NaN si moins de deux secteurs valides
    sector_a400: tuple[float, ...]  # NaN pour les secteurs ignorés
    sector_counts: tuple[int, ...]

    @property
    def n_valid(self) -> int:
        """Nombre de secteurs retenus."""
        return int(np.sum(~np.isnan(self.sector_a400)))


def equity(
    served: np.ndarray, sectors: np.ndarray, n_sectors: int, min_origins: int
) -> EquityResult:
    """Équité entre secteurs à partir de l'accès de chaque origine."""
    served = np.asarray(served, dtype=bool)
    sectors = np.asarray(sectors, dtype=np.int64)
    if served.shape != sectors.shape:
        raise ValueError("served et sectors doivent avoir la même taille")
    counts = np.bincount(sectors, minlength=n_sectors)[:n_sectors]
    hits = np.bincount(sectors, weights=served.astype(float), minlength=n_sectors)[:n_sectors]
    valid = counts >= min_origins
    rates = np.full(n_sectors, np.nan)
    rates[valid] = hits[valid] / counts[valid]
    e = float(1 - (rates[valid].max() - rates[valid].min())) if valid.sum() >= 2 else float("nan")
    return EquityResult(
        E=e, sector_a400=tuple(rates.tolist()), sector_counts=tuple(counts.tolist())
    )
