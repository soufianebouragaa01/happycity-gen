"""Opérateur de projection de validité : déterministe, non appris, sans torch.

Il transforme une carte de probabilités d'ajout en un masque d'ajouts **valide** :

1. **restriction à R** : seuls les pixels des parcelles candidates adjacentes au réseau
   piéton peuvent devenir espace vert. Tout le reste (bâti, réseau, obstacles, espaces
   verts existants) est un élément **non modifiable** et n'est jamais touché ;
2. **budget** : au plus ``budget_px`` pixels, en gardant les plus probables (à égalité,
   l'ordre des pixels décide, pour rester déterministe) ;
3. **adjacence au réseau** : seules les composantes connexes (4-connexité) qui touchent le
   voisinage du réseau piéton (≤ ``adjacency_px`` pixel) sont conservées. Un espace vert
   isolé au milieu d'un îlot n'est pas accessible.

L'étape 3 vient après le budget, car tronquer peut détacher un morceau du réseau. Le
résultat respecte toujours les trois contraintes, et l'opérateur est **idempotent**.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label


@dataclass(frozen=True)
class ProjectionResult:
    """Masque projeté et pixels retirés à chaque étape (diagnostic des violations)."""

    mask: np.ndarray  # (H, W) bool
    proposed_px: int  # pixels proposés (p ≥ seuil), avant projection
    removed_outside_r: int  # hors de R (éléments non modifiables ou parcelles non adjacentes)
    removed_budget: int  # au-delà du budget
    removed_detached: int  # composantes non adjacentes au réseau

    @property
    def n_px(self) -> int:
        """Pixels ajoutés après projection."""
        return int(self.mask.sum())


def network_neighbourhood(walk: np.ndarray, adjacency_px: int = 1) -> np.ndarray:
    """Pixels à ≤ ``adjacency_px`` pixel (8-voisinage) d'un pixel du réseau piéton."""
    near = np.asarray(walk) > 0
    if adjacency_px > 0 and near.any():
        near = binary_dilation(near, structure=np.ones((3, 3), dtype=bool), iterations=adjacency_px)
    return near


def project(
    prob: np.ndarray,
    allowed: np.ndarray,
    walk: np.ndarray,
    budget_px: int,
    *,
    threshold: float = 0.5,
    adjacency_px: int = 1,
) -> ProjectionResult:
    """Projette une carte de probabilités sur l'ensemble des masques valides.

    Args:
        prob: probabilités d'ajout ``(H, W)`` (ou masque binaire).
        allowed: région modifiable R (parcelles candidates adjacentes au réseau).
        walk: canal ``walk_network`` (> 0 sur le réseau piéton).
        budget_px: budget b en pixels.
        threshold: seuil de décision.
        adjacency_px: distance maximale au réseau (pixels) d'un ajout accessible.
    """
    prob = np.asarray(prob, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=bool)
    proposed = prob >= threshold
    sel = proposed & allowed
    removed_outside = int((proposed & ~allowed).sum())
    removed_budget = 0
    n_sel = int(sel.sum())
    if n_sel > budget_px:
        flat = np.flatnonzero(sel)
        order = flat[np.lexsort((flat, -prob.ravel()[flat]))]
        keep = np.zeros(sel.size, dtype=bool)
        keep[order[: max(int(budget_px), 0)]] = True
        sel = keep.reshape(sel.shape)
        removed_budget = n_sel - int(sel.sum())
    labels, _ = label(sel)
    touching = np.unique(labels[sel & network_neighbourhood(walk, adjacency_px)])
    kept = np.isin(labels, touching[touching > 0]) & sel
    return ProjectionResult(
        mask=kept,
        proposed_px=int(proposed.sum()),
        removed_outside_r=removed_outside,
        removed_budget=removed_budget,
        removed_detached=int(sel.sum() - kept.sum()),
    )


def compose_after(
    channels: np.ndarray, channel_names: Sequence[str], added: np.ndarray
) -> np.ndarray:
    """Carte sémantique « après » : l'existant est conservé, les ajouts deviennent verts.

    Seul le canal ``public_green`` change (``max(existant, ajout)``) ; les éléments non
    modifiables restent identiques par construction.
    """
    out = np.array(channels, dtype=np.float32, copy=True)
    k = list(channel_names).index("public_green")
    out[k] = np.maximum(out[k], np.asarray(added, dtype=np.float32))
    return out
