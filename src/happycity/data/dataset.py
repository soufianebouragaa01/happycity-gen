"""Jeu de données compact pour l'entraînement (un ``.npz`` par split, sans géométries).

Contenu par tuile :

* ``x`` : entrées du générateur, ``(C, H, W)`` en ``uint8`` (0–255). Les canaux de la
  carte M (``INPUT_CHANNELS``) sont suivis de la région modifiable R ;
* **motifs de desserte** pour la perte de condition différentiable. Pour chaque pixel de R,
  on calcule l'ensemble des origines non desservies qu'il desservirait seul
  (:func:`happycity.metrics.pixel_service`, distances réseau). Les pixels ayant le même
  ensemble partagent un motif : ``pattern_map`` (H, W) donne l'indice du motif (−1 : aucun)
  et ``patterns`` (K, O) les origines de chaque motif.

Contenu par paire : tuile, budget, cible ``a``, masque cible (bits compactés).

L'entraînement (Kaggle) ne lit que ces fichiers : ni geopandas, ni OSMnx, ni graphe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from happycity.data.arrays import tile_patterns, unpack_masks
from happycity.data.raster import TileRaster
from happycity.metrics import MetricParams, TileAccessContext, pixel_service, served_origins

__all__ = ["tile_patterns", "unpack_masks"]  # réexportés pour compatibilité

MAP_CHANNELS = (
    "residential_buildings",
    "other_buildings",
    "walk_network",
    "public_green",
    "obstacles",
    "origins",
    "unserved_origins",
)
INPUT_CHANNELS = (*MAP_CHANNELS, "allowed_region")
COUNT_CHANNELS = ("origins", "unserved_origins")
COUNT_SCALE = 8.0  # log1p(n) / log1p(8) : 8 origines ou plus dans un pixel → 1


def allowed_region(tile: TileRaster) -> np.ndarray:
    """Région modifiable R : pixels des parcelles candidates ADJACENTES au réseau."""
    adjacent = tile.candidates.loc[tile.candidates["adjacent"].astype(bool), "local_id"]
    return np.isin(tile.candidate_ids, np.asarray(adjacent, dtype=np.int32))


def tile_inputs(tile: TileRaster) -> np.ndarray:
    """Entrées normalisées dans [0, 1], ``(len(INPUT_CHANNELS), H, W)`` float32."""
    layers = []
    for name in MAP_CHANNELS:
        ch = np.asarray(tile.channel(name), dtype=np.float32)
        if name in COUNT_CHANNELS:
            ch = np.log1p(np.maximum(ch, 0)) / np.log1p(COUNT_SCALE)
        layers.append(np.clip(ch, 0.0, 1.0))
    layers.append(allowed_region(tile).astype(np.float32))
    return np.stack(layers)


def to_uint8(x: np.ndarray) -> np.ndarray:
    """[0, 1] → 0–255."""
    return np.asarray(np.round(np.clip(x, 0, 1) * 255), dtype=np.uint8)


def service_patterns(
    tile: TileRaster, ctx: TileAccessContext, params: MetricParams
) -> dict[str, np.ndarray]:
    """Motifs de desserte des pixels de R vers les origines non desservies.

    Returns:
        ``pattern_map`` (H, W) int16 (−1 si le pixel ne dessert aucune nouvelle origine),
        ``patterns`` (K, O) bool, ``n_origins`` et ``n_base`` (origines déjà desservies).
    """
    base = served_origins(ctx, params)
    targets = np.flatnonzero(~base & (ctx.origin_nodes >= 0))
    rows, cols = np.nonzero(allowed_region(tile))
    service = pixel_service(ctx, rows, cols, targets, params)
    pattern_map = np.full(tile.candidate_ids.shape, -1, dtype=np.int16)
    useful = service.any(axis=1)
    if useful.any():
        patterns, inverse = np.unique(service[useful], axis=0, return_inverse=True)
        pattern_map[rows[useful], cols[useful]] = inverse.ravel().astype(np.int16)
    else:
        patterns = np.zeros((0, len(targets)), dtype=bool)
    return {
        "pattern_map": pattern_map,
        "patterns": patterns.astype(bool),
        "n_origins": np.array(len(ctx.origin_nodes), dtype=np.int32),
        "n_base": np.array(base.sum(), dtype=np.int32),
    }


def soft_a400(prob: np.ndarray, data: dict[str, np.ndarray]) -> float:
    """A400 « doux » (numpy) d'une carte de probabilités, à partir des motifs.

    Une origine non desservie l'est après ajout avec la probabilité 1 − Π(1 − pᵢ) sur les
    pixels qui la desservent. Pour un masque binaire, c'est l'estimation pixel par pixel.
    """
    n = int(data["n_origins"])
    if n == 0:
        return float("nan")
    patterns = np.asarray(data["patterns"], dtype=np.float64)
    pmap = np.asarray(data["pattern_map"]).ravel()
    p = np.clip(np.asarray(prob, dtype=np.float64).ravel(), 0.0, 1.0 - 1e-6)
    log_q = np.zeros(len(patterns))
    np.add.at(log_q, pmap[pmap >= 0], np.log1p(-p[pmap >= 0]))
    new = 1.0 - np.exp(log_q @ patterns) if len(patterns) else np.zeros(0)
    return float((int(data["n_base"]) + new.sum()) / n)


def pack_split(
    tiles: list[tuple[TileRaster, dict[str, np.ndarray]]],
    pairs: list[dict[Any, Any]],
    masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Assemble les tableaux d'un split (tuiles, motifs, paires)."""
    tile_ids = [t.spec.tile_id for t, _ in tiles]
    index = {t: i for i, t in enumerate(tile_ids)}
    shapes = np.array([p["patterns"].shape for _, p in tiles], dtype=np.int64).reshape(-1, 2)
    flat = [p["patterns"].ravel() for _, p in tiles]
    size = tiles[0][0].candidate_ids.shape if tiles else (0, 0)
    stack = np.stack(masks) if masks else np.zeros((0, *size), dtype=bool)
    return {
        "tile_ids": np.array(tile_ids, dtype=str),
        "x": np.stack([to_uint8(tile_inputs(t)) for t, _ in tiles]),
        "pattern_map": np.stack([p["pattern_map"] for _, p in tiles]),
        "pattern_shape": shapes,
        "pattern_data": np.concatenate(flat) if flat else np.zeros(0, dtype=bool),
        "n_origins": np.array([p["n_origins"] for _, p in tiles], dtype=np.int32),
        "n_base": np.array([p["n_base"] for _, p in tiles], dtype=np.int32),
        "pair_ids": np.array([p["pair_id"] for p in pairs], dtype=str),
        "pair_tile": np.array([index[p["tile_id"]] for p in pairs], dtype=np.int32),
        "b_frac": np.array([p["b_frac"] for p in pairs], dtype=np.float32),
        "budget_px": np.array([p["budget_px"] for p in pairs], dtype=np.int32),
        "a_target": np.array([p["a_target"] for p in pairs], dtype=np.float32),
        "a_before": np.array([p["A400_before"] for p in pairs], dtype=np.float32),
        "n_px": np.array([p["n_px"] for p in pairs], dtype=np.int32),
        "masks": np.packbits(stack, axis=-1),
        "mask_shape": np.array(stack.shape, dtype=np.int64),
        "input_channels": np.array(INPUT_CHANNELS, dtype=str),
    }


def save_split(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    """Écrit un split (``.npz`` compressé, sans objets Python) et son résumé JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, allow_pickle=False, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
