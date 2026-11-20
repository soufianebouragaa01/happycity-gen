"""Évaluation : propositions des modèles et des références, métriques EXACTES.

Deux temps :

1. **génération** (processus principal, GPU si disponible) : pour chaque exécution
   (checkpoint ``best.pt``) et chaque problème, ``eval.n_proposals`` propositions avec des
   codes z identiques pour toutes les exécutions (graine dérivée du problème). On garde la
   sortie **brute** (p ≥ seuil) et la sortie **projetée** ;
2. **mesure** (processus parallèles, par tuile) : métriques exactes du module indépendant
   (:func:`happycity.metrics.evaluate_tile`, Dijkstra réseau) et validité de chaque masque,
   pour les modèles ET pour les références (aléatoire, glouton, glouton ciblé).

Sortie « retenue » d'un modèle : projetée si la variante utilise la projection (C), brute
sinon (A, B, ablation ``no_projection``). La sortie projetée de A et B est aussi mesurée,
pour isoler l'apport de la projection.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from happycity.baselines import greedy_placement, random_placement, targeted_greedy_placement
from happycity.data.access_context import metric_params
from happycity.data.dataset import tile_inputs, to_uint8
from happycity.data.pairs import make_problem, stable_seed
from happycity.data.pipeline import _pool_map, context_dir, tiles_dir
from happycity.data.raster import TileRaster
from happycity.metrics import TileAccessContext, evaluate_tile
from happycity.projection import validity

log = logging.getLogger(__name__)

WALK_CHANNEL = 2


@dataclass(frozen=True)
class RunInfo:
    """Une exécution entraînée (dossier ``experiments/<nom>``)."""

    name: str
    checkpoint: Path
    variant: str
    ablation: str
    seed: int
    projection: bool


def discover_runs(experiments_dir: Path, pattern: str = "*") -> list[RunInfo]:
    """Exécutions disposant d'un ``checkpoints/best.pt``.

    Les exécutions sont rangées par graine (``experiments/seed42/C-full-s42``) ; les dossiers
    placés directement sous ``experiments/`` sont aussi acceptés. ``pattern`` filtre le nom
    de l'exécution (par ex. ``"C-*"``).
    """
    from omegaconf import OmegaConf

    root = Path(experiments_dir)
    dirs = {c.parent.parent for c in root.glob("*/checkpoints/best.pt")}
    dirs |= {c.parent.parent for c in root.glob("seed*/*/checkpoints/best.pt")}
    runs = []
    for d in sorted((d for d in dirs if d.match(pattern)), key=lambda d: d.name):
        ckpt = d / "checkpoints" / "best.pt"
        conf = d / "config.yaml"
        if not ckpt.exists() or not conf.exists():
            continue
        c = OmegaConf.to_container(OmegaConf.load(conf), resolve=False)
        assert isinstance(c, dict)
        runs.append(RunInfo(
            name=d.name, checkpoint=ckpt, variant=str(c["model"]["name"]),
            ablation=str(c["ablation"]["name"]), seed=int(c["train"]["seed"]),
            projection=bool(c["model"]["projection"]),
        ))  # fmt: skip
    return runs


def _inputs(tile: TileRaster) -> np.ndarray:
    """Entrées identiques à l'entraînement (quantifiées sur 8 bits)."""
    return to_uint8(tile_inputs(tile)).astype(np.float32) / 255.0


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask, dtype=bool))


def _unpack(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)


def generate(
    cfg: DictConfig, problems: pd.DataFrame, runs: list[RunInfo], device: str
) -> dict[str, list[dict[str, Any]]]:
    """Propositions de chaque exécution pour chaque problème, regroupées par tuile."""
    from happycity.models.inference import load_checkpoint, propose

    n = int(cfg.eval.n_proposals)
    inputs = {
        f: _inputs(TileRaster.load(tiles_dir(cfg) / f)) for f in problems["pair_file"].unique()
    }
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        nets, state = load_checkpoint(run.checkpoint, device)
        thr = float(state["config"]["model"]["threshold"])
        start = time.time()
        for p in problems.to_dict("records"):
            t0 = time.perf_counter()
            props = propose(nets.generator, inputs[str(p["pair_file"])], float(p["b_frac"]),
                            float(p["a_target"]), int(p["budget_px"]), n=n,
                            seed=stable_seed(cfg.eval.seed, p["problem_id"]),
                            threshold=thr)  # fmt: skip
            ms = (time.perf_counter() - t0) * 1000 / n
            for k, prop in enumerate(props):
                for output, mask in (("brute", prop.raw), ("projetée", prop.projection.mask)):
                    out[str(p["pair_file"])].append({
                        "problem_id": p["problem_id"], "method": run.name, "family": "génératif",
                        "variant": run.variant, "ablation": run.ablation, "seed": run.seed,
                        "output": output, "deployed": (output == "projetée") == run.projection,
                        "proposal": k, "infer_ms": ms, "bits": _pack(mask),
                    })  # fmt: skip
        log.info("%s : %d problèmes en %.0f s", run.name, len(problems), time.time() - start)
    return out


EvalJob = tuple[DictConfig, str, list[dict[Any, Any]], list[dict[str, Any]]]


def _iou_distance(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return float("nan")
    values = []
    for a, b in combinations(masks, 2):
        union = np.logical_or(a, b).sum()
        values.append(1 - np.logical_and(a, b).sum() / union if union else 0.0)
    return float(np.mean(values))


def _evaluate_tile(job: EvalJob) -> list[dict[str, Any]]:
    """Métriques exactes et validité de toutes les propositions d'une tuile."""
    cfg, pair_file, problems, entries = job
    params = metric_params(cfg.data)
    tile = TileRaster.load(tiles_dir(cfg) / pair_file)
    ctx = TileAccessContext.load(context_dir(cfg) / pair_file)
    shape = tile.candidate_ids.shape
    x = _inputs(tile)
    allowed, walk = x[-1] > 0.5, x[WALK_CHANNEL] > 0.5
    placement = make_problem(tile, ctx, params)
    res2 = float(cfg.data.tile.m_per_px) ** 2
    cache: dict[bytes, Any] = {}

    def scores(mask: np.ndarray) -> Any:
        key = _pack(mask).tobytes()
        if key not in cache:
            cache[key] = evaluate_tile(ctx, mask, params=params)
        return cache[key]

    # Références : glouton et aléatoire ne dépendent que du budget ; le glouton ciblé, de a.
    by_budget: dict[int, dict[str, list[np.ndarray]]] = {}
    draws = int(cfg.eval.random_baseline_draws)
    for budget in sorted({int(p["budget_px"]) for p in problems}):
        rngs = [np.random.default_rng(stable_seed(cfg.eval.seed, pair_file, budget, "rnd", d))
                for d in range(draws)]  # fmt: skip
        by_budget[budget] = {
            "glouton": [greedy_placement(placement, budget).mask],
            "aléatoire": [random_placement(placement, budget, r).mask for r in rngs],
        }
    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], np.ndarray]]] = defaultdict(list)
    for p in problems:
        budget = int(p["budget_px"])
        base = {"family": "référence", "variant": "", "ablation": "", "seed": -1,
                "output": "valide", "deployed": True, "infer_ms": np.nan}  # fmt: skip
        targeted = targeted_greedy_placement(placement, budget, float(p["a_target"]))
        masks = {**by_budget[budget], "glouton ciblé": [targeted.mask]}
        for method, ms in masks.items():
            for k, m in enumerate(ms):
                entry = {**base, "problem_id": p["problem_id"], "method": method, "proposal": k}
                groups[(p["problem_id"], method, "valide")].append((entry, m))
    for e in entries:
        groups[(e["problem_id"], e["method"], e["output"])].append((e, _unpack(e["bits"], shape)))

    info = {p["problem_id"]: p for p in problems}
    rows = []
    for (pid, _, _), items in groups.items():
        p = info[pid]
        diversity = _iou_distance([m for _, m in items])
        for e, mask in items:
            s = scores(mask)
            v = validity(mask, allowed, walk, int(p["budget_px"]))
            rows.append({
                **{k: e[k] for k in ("problem_id", "method", "family", "variant", "ablation",
                                     "seed", "output", "deployed", "proposal", "infer_ms")},
                "n_px": int(mask.sum()), "fill": mask.sum() / max(int(p["budget_px"]), 1),
                "A400": s.A400, "V": s.V, "E": s.E, "n_valid_sectors": s.n_valid_sectors,
                "delta_m2": s.delta_green_m2, "budget_m2": int(p["budget_px"]) * res2,
                "outside_r_px": v.outside_r_px, "over_budget_px": v.over_budget_px,
                "detached_px": v.detached_px, "valid": v.valid, "diversity": diversity,
            })  # fmt: skip
    return rows


def evaluate(
    cfg: DictConfig, problems: pd.DataFrame, runs: list[RunInfo], device: str, workers: int
) -> pd.DataFrame:
    """Table complète (une ligne par proposition) jointe aux problèmes."""
    generated = generate(cfg, problems, runs, device) if runs else {}
    jobs: list[EvalJob] = []
    for pair_file, group in problems.groupby("pair_file"):
        jobs.append(
            (cfg, str(pair_file), group.to_dict("records"), generated.get(str(pair_file), []))
        )
    rows = [r for chunk in _pool_map(_evaluate_tile, jobs, workers) for r in chunk]
    table = pd.DataFrame(rows).merge(problems, on="problem_id", how="left", validate="many_to_one")
    table["dA400"] = table["A400"] - table["A400_before"]
    table["abs_err"] = (table["A400"] - table["a_target"]).abs()
    return table
