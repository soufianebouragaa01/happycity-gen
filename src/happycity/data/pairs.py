"""Paires avant/après (supervision simulée) et évaluation des références non génératives.

Pour chaque tuile exploitable et chaque budget b, ``solutions_per_budget`` aménagements du
**glouton stochastique** (tirage parmi les ``top_k`` meilleurs ajouts) servent de cibles.
La condition ``a`` est l'A400 réellement atteint par la solution (étiquetage a posteriori),
donc toujours atteignable. Ces paires sont une supervision SIMULÉE, pas des projets urbains.

Le découpage train/val/test est celui de la tuile (par quartier, fixé AVANT de créer les
variantes) : aucune variante d'une tuile de test ne peut se retrouver à l'entraînement.
"""

from __future__ import annotations

import zlib
from itertools import combinations
from typing import Any

import numpy as np
from omegaconf import DictConfig

from happycity.baselines import Placement, PlacementProblem, greedy_placement, random_placement
from happycity.data.raster import TileRaster
from happycity.metrics import MetricParams, TileAccessContext, evaluate_tile


def stable_seed(*parts: object) -> int:
    """Graine déterministe à partir d'identifiants (indépendante de l'ordre de traitement)."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def synthetic_split(tile_id: str, val_fraction: float, test_fraction: float = 0.0) -> str:
    """Split d'une carte synthétique (par carte, hachage stable de l'identifiant).

    ``val_synth`` sert au suivi de l'entraînement ; ``test_synth`` est un test synthétique
    tenu à l'écart, rapporté SÉPARÉMENT du test réel (qui reste le test final).
    """
    u = stable_seed(tile_id) % 1000
    if u < val_fraction * 1000:
        return "val_synth"
    if u < (val_fraction + test_fraction) * 1000:
        return "test_synth"
    return "train"


def budget_pixels(fraction: float, data_cfg: DictConfig) -> int:
    """Budget b en pixels (fraction de la surface de la tuile)."""
    return int(fraction * int(data_cfg.tile.size_px) ** 2)


def make_problem(
    tile: TileRaster, ctx: TileAccessContext, params: MetricParams
) -> PlacementProblem:
    """Problème de placement d'une tuile (parcelles adjacentes au réseau seulement)."""
    adjacent = tile.candidates.loc[tile.candidates["adjacent"].astype(bool), "local_id"]
    return PlacementProblem(
        tile.candidate_ids, [int(i) for i in adjacent], tile.channel("walk_network"), ctx, params
    )


def mean_pairwise_iou_distance(masks: list[np.ndarray]) -> float:
    """Diversité : moyenne de 1 − IoU entre propositions (NaN si moins de deux)."""
    if len(masks) < 2:
        return float("nan")
    values = []
    for a, b in combinations(masks, 2):
        union = np.logical_or(a, b).sum()
        values.append(1 - np.logical_and(a, b).sum() / union if union else 0.0)
    return float(np.mean(values))


def _nanmean(values: list[float]) -> float:
    """Moyenne en ignorant les NaN (NaN si toutes les valeurs le sont, sans avertissement)."""
    arr = np.asarray(values, dtype=float)
    return float(arr[~np.isnan(arr)].mean()) if (~np.isnan(arr)).any() else float("nan")


def _scores(prefix: str, ctx: TileAccessContext, placement: Placement, params: MetricParams,
            budget_m2: float) -> dict[str, Any]:  # fmt: skip
    s = evaluate_tile(ctx, placement.mask, params=params, budget_m2=budget_m2)
    return {f"{prefix}A400": s.A400, f"{prefix}V": s.V, f"{prefix}E": s.E,
            f"{prefix}delta_m2": s.delta_green_m2}  # fmt: skip


def tile_pairs(
    tile: TileRaster, ctx: TileAccessContext, split: str, data_cfg: DictConfig, params: MetricParams
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[dict[str, Any]]]:
    """Paires d'une tuile, masques associés et scores des références au même budget.

    Returns:
        ``(lignes de paires, masques, lignes de références)``.
    """
    cfg = data_cfg.pairs
    res2 = float(data_cfg.tile.m_per_px) ** 2
    problem = make_problem(tile, ctx, params)
    before = evaluate_tile(ctx, params=params)
    common = {
        "tile_id": tile.spec.tile_id,
        "source": tile.spec.source,
        "zone": tile.spec.zone,
        "split": split,
        "capacity_px": problem.capacity_px,
        "A400_before": before.A400,
        "V_before": before.V,
        "E_before": before.E,
    }
    pairs, masks, baselines = [], [], []
    for b_idx, fraction in enumerate(cfg.budget_fractions):
        budget_px = budget_pixels(float(fraction), data_cfg)
        budget_m2 = budget_px * res2
        info = {**common, "b_frac": float(fraction), "budget_px": budget_px,
                "saturated": problem.capacity_px < budget_px}  # fmt: skip

        greedy = greedy_placement(problem, budget_px)
        rngs = [
            np.random.default_rng(stable_seed(tile.spec.tile_id, b_idx, "rnd", d))
            for d in range(int(cfg.random_draws))
        ]
        randoms = [random_placement(problem, budget_px, rng) for rng in rngs]
        random_scores = [_scores("", ctx, r, params, budget_m2) for r in randoms]
        keys = list(random_scores[0])
        row: dict[str, Any] = {
            **info,
            **_scores("greedy_", ctx, greedy, params, budget_m2),
            "greedy_px": greedy.n_px,
            "random_px": float(np.mean([r.n_px for r in randoms])),
        }
        for k in keys:
            row[f"random_{k}"] = _nanmean([r[k] for r in random_scores])
        baselines.append(row)

        # Tirages stochastiques ; on ne garde que des solutions de qualité VOISINE :
        # gain d'accès ≥ min_relative_gain × meilleur gain connu (glouton compris).
        draws = [
            greedy_placement(
                problem,
                budget_px,
                np.random.default_rng(stable_seed(cfg.seed, tile.spec.tile_id, b_idx, s)),
                int(cfg.top_k),
            )
            for s in range(int(cfg.attempts))
        ]
        base_served = int(problem.base_served.sum())
        best_gain = max(
            int(d.served.sum()) - base_served for d in [greedy, *draws] if d.served is not None
        )
        threshold = float(cfg.min_relative_gain) * best_gain
        seen: set[bytes] = set()
        solutions: list[Placement] = []
        low_quality = 0
        for sol in draws:
            assert sol.served is not None
            key = np.packbits(sol.mask).tobytes()
            if sol.n_px == 0 or key in seen:
                continue
            if int(sol.served.sum()) - base_served < threshold:
                low_quality += 1
                continue
            seen.add(key)
            if len(solutions) < int(cfg.solutions_per_budget):
                solutions.append(sol)
        info["low_quality_draws"] = low_quality
        diversity = mean_pairwise_iou_distance([s.mask for s in solutions])
        if not solutions:
            pairs.append(
                {**info, "solution": -1, "excluded": "aucun gain d'accès possible", "method": "",
                 "n_px": 0, "n_parcels": 0, "a_target": np.nan, "A400_after": before.A400,
                 "V_after": before.V, "E_after": before.E, "delta_m2": 0.0, "internal_a400": np.nan,
                 "diversity_1_iou": np.nan, "n_unique_solutions": 0}
            )  # fmt: skip
            masks.append(np.zeros_like(tile.candidate_ids, dtype=bool))
            continue
        for s_idx, sol in enumerate(solutions):
            scores = _scores("", ctx, sol, params, budget_m2)
            pairs.append(
                {
                    **info,
                    "solution": s_idx,
                    "excluded": "",
                    "method": sol.method,
                    "n_px": sol.n_px,
                    "n_parcels": len(sol.parcels),
                    "a_target": scores["A400"],
                    "A400_after": scores["A400"],
                    "V_after": scores["V"],
                    "E_after": scores["E"],
                    "delta_m2": scores["delta_m2"],
                    "internal_a400": sol.a400,
                    "diversity_1_iou": diversity,
                    "n_unique_solutions": len(solutions),
                }
            )
            masks.append(sol.mask)
    return pairs, masks, baselines
