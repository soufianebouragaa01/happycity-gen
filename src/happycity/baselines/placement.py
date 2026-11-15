"""Références non génératives : placement aléatoire valide, glouton, glouton stochastique.

Toutes les méthodes respectent les mêmes contraintes que l'opérateur de projection :

* ajouts uniquement sur des parcelles candidates **adjacentes au réseau piéton** ;
* surface ajoutée ≤ budget b (en pixels) ;
* si une parcelle dépasse le budget restant, on n'en prend qu'un **morceau compact** :
  les pixels les plus proches (distance euclidienne) du pixel de la parcelle le plus
  proche du réseau. Le morceau est ancré sur le réseau, donc accessible, et forme un bloc
  plutôt qu'une bande ou un anneau le long des rues.

L'objectif interne du glouton (origines nouvellement desservies) est **exact** : l'accès
après aménagement est l'union des origines desservies par chaque ajout, puisque A400
utilise la source la plus proche. Les scores rapportés sont néanmoins recalculés par le
module de métriques indépendant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

from happycity.metrics import (
    MetricParams,
    TileAccessContext,
    access_points_for,
    accessibility,
    mask_polygons,
    served_origins,
)


@dataclass
class Placement:
    """Aménagement proposé (masque) et trace de sa construction."""

    method: str
    mask: np.ndarray  # (H, W) bool
    parcels: list[int] = field(default_factory=list)  # identifiants locaux, dans l'ordre
    served: np.ndarray | None = None  # origines desservies après (estimation interne)

    @property
    def n_px(self) -> int:
        """Nombre de pixels ajoutés."""
        return int(self.mask.sum())

    @property
    def a400(self) -> float:
        """A400 après aménagement (estimation interne)."""
        if self.served is None or len(self.served) == 0:
            return float("nan")
        return float(self.served.mean())


class PlacementProblem:
    """Données d'une tuile pour les méthodes de placement (gains mis en cache).

    Args:
        candidate_ids: carte des parcelles (identifiant local, −1 hors parcelle).
        adjacent_ids: parcelles adjacentes au réseau (seules utilisables).
        walk_raster: canal ``walk_network`` (pour l'ordre des pixels d'un morceau).
        ctx: contexte d'évaluation de la tuile.
        params: paramètres des métriques.
    """

    def __init__(
        self,
        candidate_ids: np.ndarray,
        adjacent_ids: list[int],
        walk_raster: np.ndarray,
        ctx: TileAccessContext,
        params: MetricParams,
    ) -> None:
        self.ctx = ctx
        self.params = params
        self.shape = candidate_ids.shape
        dist = distance_transform_edt(walk_raster <= 0)  # 0 sur le réseau
        self._pixels: dict[int, np.ndarray] = {}
        width = candidate_ids.shape[1]
        for cid in adjacent_ids:
            flat = np.flatnonzero(candidate_ids.ravel() == cid)
            if len(flat):
                anchor = flat[np.lexsort((flat, dist.ravel()[flat]))[0]]  # le plus proche du réseau
                rows, cols = np.divmod(flat, width)
                ar, ac = divmod(int(anchor), width)
                spread = np.hypot(rows - ar, cols - ac)
                self._pixels[int(cid)] = flat[np.lexsort((flat, spread))]  # bloc autour de l'ancre
        self.base_served = served_origins(ctx, params)
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    @property
    def parcels(self) -> list[int]:
        """Parcelles utilisables (adjacentes, non vides)."""
        return sorted(self._pixels)

    @property
    def capacity_px(self) -> int:
        """Surface totale aménageable (pixels)."""
        return int(sum(len(p) for p in self._pixels.values()))

    def size(self, cid: int) -> int:
        """Taille d'une parcelle (pixels)."""
        return len(self._pixels[cid])

    def piece(self, cid: int, n_px: int) -> np.ndarray:
        """Masque d'un morceau compact de ``n_px`` pixels ancré sur le réseau."""
        mask = np.zeros(self.shape[0] * self.shape[1], dtype=bool)
        mask[self._pixels[cid][:n_px]] = True
        return mask.reshape(self.shape)

    def served_by(self, cid: int, n_px: int) -> np.ndarray:
        """Origines desservies par l'existant ET un morceau de ``n_px`` pixels de ``cid``."""
        n_px = min(n_px, self.size(cid))
        key = (cid, n_px)
        if key not in self._cache:
            polygons = mask_polygons(self.piece(cid, n_px), self.ctx)
            points = self.ctx.existing.concat(
                access_points_for(polygons, self.ctx.walk, self.params.reach_m)
            )
            res = accessibility(self.ctx.walk, self.ctx.origin_nodes, points, self.params.radius_m)
            self._cache[key] = res.served
        return self._cache[key]


def _empty(problem: PlacementProblem, method: str) -> Placement:
    return Placement(method, np.zeros(problem.shape, dtype=bool), [], problem.base_served.copy())


def random_placement(
    problem: PlacementProblem, budget_px: int, rng: np.random.Generator
) -> Placement:
    """Placement aléatoire VALIDE : parcelles adjacentes en ordre aléatoire, jusqu'au budget."""
    result = _empty(problem, "random")
    assert result.served is not None
    remaining = budget_px
    for cid in rng.permutation(problem.parcels):
        if remaining <= 0:
            break
        take = min(problem.size(int(cid)), remaining)
        result.mask |= problem.piece(int(cid), take)
        result.served |= problem.served_by(int(cid), take)
        result.parcels.append(int(cid))
        remaining -= take
    return result


def _greedy(
    problem: PlacementProblem,
    budget_px: int,
    criterion: str,
    rng: np.random.Generator | None,
    top_k: int,
    method: str,
) -> Placement:
    result = _empty(problem, method)
    assert result.served is not None
    remaining = budget_px
    available = set(problem.parcels)
    while remaining > 0 and available:
        options = []
        for cid in sorted(available):
            take = min(problem.size(cid), remaining)
            gain = int((problem.served_by(cid, take) & ~result.served).sum())
            if gain > 0:
                score = gain / take if criterion == "ratio" else float(gain)
                options.append((score, gain, -take, cid, take))
        if not options:
            break
        options.sort(reverse=True)
        k = min(top_k, len(options)) if rng is not None else 1
        _, _, _, cid, take = options[int(rng.integers(k)) if rng is not None else 0]
        result.mask |= problem.piece(cid, take)
        result.served |= problem.served_by(cid, take)
        result.parcels.append(cid)
        available.discard(cid)
        remaining -= take
    return result


def greedy_placement(
    problem: PlacementProblem,
    budget_px: int,
    rng: np.random.Generator | None = None,
    top_k: int = 1,
) -> Placement:
    """Glouton maximisant A400 sous contrainte de budget.

    * Déterministe (``rng=None``) : meilleur de deux gloutons — au **ratio** (nouvelles
      origines desservies par pixel) et au **gain absolu** — selon l'A400 atteint (à
      égalité, le moins de pixels). Combiner les deux corrige le défaut connu du seul
      ratio sous contrainte de budget (sac à dos).
    * Stochastique (``rng`` et ``top_k > 1``) : glouton au ratio, choix tiré uniformément
      parmi les ``top_k`` meilleurs ajouts de gain strictement positif → plusieurs
      solutions de qualité voisine pour une même carte.

    On s'arrête quand aucun ajout ne dessert de nouvelle origine : le budget peut alors
    rester partiellement inutilisé.
    """
    if rng is not None and top_k > 1:
        return _greedy(problem, budget_px, "ratio", rng, top_k, f"greedy_top{top_k}")
    runs = [_greedy(problem, budget_px, c, None, 1, "greedy") for c in ("ratio", "gain")]
    return max(runs, key=lambda r: (r.a400, -r.n_px))
