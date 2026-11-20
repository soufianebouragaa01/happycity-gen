"""Démonstration : un scénario d'aménagement sur une tuile, chiffré avec les métriques exactes.

Ce module est le **seul** point d'entrée de la démonstration. La commande
``happycity demo``, le notebook ``notebooks/demo.ipynb`` et l'interface Streamlit
(``app/streamlit_app.py``) appellent ces mêmes fonctions, sans logique propre.

Un scénario = (tuile, budget b, cible a) :

* ``n`` propositions du modèle génératif (sortie retenue : projetée pour C, brute sinon) ;
* la référence **glouton ciblé** (non générative, reçoit aussi a) ;
* pour chacune : A400, ΔA400, E (toujours avec A400), V, surface ajoutée, validité, |â − a|.

Les indicateurs mesurent l'accès piéton à 400 m et la végétation. Ils ne mesurent pas le
bien-être des habitants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from happycity.baselines import greedy_placement, targeted_greedy_placement
from happycity.data.access_context import metric_params
from happycity.data.pairs import budget_pixels, make_problem
from happycity.data.pipeline import context_dir, tiles_dir
from happycity.data.raster import TileRaster
from happycity.evaluation.runner import WALK_CHANNEL, RunInfo, _inputs, discover_runs
from happycity.metrics import TileAccessContext, TileScores, evaluate_tile
from happycity.projection import Validity, validity

DISCLAIMER = (
    "Scénarios exploratoires : les indicateurs mesurent l'accès piéton à 400 m (distance "
    "réseau) et la végétation, pas le bien-être des habitants. Les propositions ne sont pas "
    "des projets urbains : elles ignorent le foncier, les coûts et la concertation."
)


# --------------------------------------------------------------------------- choix


def available_tiles(cfg: DictConfig, splits: list[str] | None = None) -> pd.DataFrame:
    """Tuiles utilisables pour la démonstration (une ligne par tuile).

    Colonnes : identifiant, fichier, split, source, quartier, A400/V/E avant, capacité
    aménageable (px) et A400 atteint par le glouton pour chaque budget (``greedy_<b>``).
    """
    processed = Path(cfg.paths.processed_dir)
    base = pd.read_parquet(processed / "baselines.parquet")
    pairs = pd.read_parquet(processed / "pairs.parquet", columns=["tile_id", "pair_file"])
    if splits:
        base = base[base["split"].isin(splits)]
    tiles = base.drop_duplicates("tile_id")[
        ["tile_id", "split", "source", "zone", "A400_before", "V_before", "E_before",
         "capacity_px"]
    ].copy()  # fmt: skip
    greedy = base.pivot_table(index="tile_id", columns="b_frac", values="greedy_A400")
    greedy.columns = pd.Index([f"greedy_{float(c):.2f}" for c in greedy.columns])
    files = pairs.drop_duplicates("tile_id").set_index("tile_id")["pair_file"]
    tiles["pair_file"] = tiles["tile_id"].map(files)
    tiles = tiles.join(greedy, on="tile_id")
    order = {"test": 0, "val": 1, "train": 2, "test_synth": 3, "val_synth": 4}
    tiles["_o"] = tiles["split"].map(order).fillna(9)
    return tiles.dropna(subset=["pair_file"]).sort_values(["_o", "tile_id"]).drop(columns="_o")


def available_runs(cfg: DictConfig) -> list[RunInfo]:
    """Modèles entraînés disponibles (``experiments/seed<graine>/<exécution>``)."""
    return discover_runs(Path(cfg.paths.experiments_dir))


def default_run(runs: list[RunInfo]) -> RunInfo | None:
    """Modèle par défaut : C complet, plus petite graine ; sinon le premier disponible."""
    full_c = [r for r in runs if r.variant == "C" and r.ablation in ("full", "none")]
    pool = full_c or runs
    return sorted(pool, key=lambda r: (r.seed, r.name))[0] if pool else None


# --------------------------------------------------------------------------- scénario


@dataclass
class Option:
    """Une proposition (modèle ou référence) et ses indicateurs exacts."""

    name: str
    mask: np.ndarray
    scores: TileScores
    validity: Validity


@dataclass
class Scenario:
    """Résultat d'un scénario sur une tuile."""

    tile: TileRaster
    run: str
    b_frac: float
    budget_px: int
    a_target: float
    before: TileScores
    greedy_max: float  # A400 atteint par le glouton avec ce budget
    options: list[Option] = field(default_factory=list)

    @property
    def reachable(self) -> bool:
        """Cible atteignable par le glouton (estimation par le bas : glouton non optimal)."""
        return self.a_target <= self.greedy_max + 1e-6


@lru_cache(maxsize=8)
def _networks(checkpoint: str, device: str) -> tuple[Any, float]:
    from happycity.models.inference import load_checkpoint

    nets, state = load_checkpoint(Path(checkpoint), device)
    return nets.generator, float(state["config"]["model"]["threshold"])


def run_scenario(
    cfg: DictConfig,
    run: RunInfo,
    pair_file: str,
    b_frac: float,
    a_target: float,
    *,
    n: int = 3,
    seed: int = 0,
    device: str = "cpu",
) -> Scenario:
    """Génère ``n`` propositions et la référence glouton ciblé, chiffrées exactement."""
    from happycity.models.inference import propose

    params = metric_params(cfg.data)
    tile = TileRaster.load(tiles_dir(cfg) / pair_file)
    ctx = TileAccessContext.load(context_dir(cfg) / pair_file)
    budget = budget_pixels(float(b_frac), cfg.data)
    x = _inputs(tile)
    allowed, walk = x[-1] > 0.5, x[WALK_CHANNEL] > 0.5
    problem = make_problem(tile, ctx, params)
    greedy_max = greedy_placement(problem, budget).a400
    scenario = Scenario(tile=tile, run=run.name, b_frac=float(b_frac), budget_px=budget,
                        a_target=float(a_target), before=evaluate_tile(ctx, params=params),
                        greedy_max=float(greedy_max))  # fmt: skip

    generator, thr = _networks(str(run.checkpoint), device)
    proposals = propose(generator, x, float(b_frac), float(a_target), budget, n=n, seed=seed,
                        threshold=thr)  # fmt: skip
    for k, prop in enumerate(proposals, start=1):
        mask = prop.deployed(run.projection)
        scenario.options.append(Option(
            f"{run.name} · proposition {k}", mask, evaluate_tile(ctx, mask, params=params),
            validity(mask, allowed, walk, budget),
        ))  # fmt: skip
    ref = targeted_greedy_placement(problem, budget, float(a_target)).mask
    scenario.options.append(Option("glouton ciblé (référence)", ref,
                                   evaluate_tile(ctx, ref, params=params),
                                   validity(ref, allowed, walk, budget)))  # fmt: skip
    return scenario


def scenario_table(scenario: Scenario) -> pd.DataFrame:
    """Tableau des indicateurs (état initial + chaque option)."""
    res2 = float(scenario.tile.spec.size_m / scenario.tile.candidate_ids.shape[0]) ** 2
    b = scenario.before
    rows: list[dict[str, Any]] = [{
        "option": "état initial", "A400": b.A400, "ΔA400": 0.0, "E": b.E, "V": b.V,
        "surface ajoutée (m²)": 0.0, "valide": True, "|â − a|": abs(b.A400 - scenario.a_target),
    }]  # fmt: skip
    for o in scenario.options:
        rows.append({
            "option": o.name, "A400": o.scores.A400, "ΔA400": o.scores.A400 - b.A400,
            "E": o.scores.E, "V": o.scores.V, "surface ajoutée (m²)": float(o.mask.sum()) * res2,
            "valide": o.validity.valid, "|â − a|": abs(o.scores.A400 - scenario.a_target),
        })  # fmt: skip
    return pd.DataFrame(rows)


def diversity(scenario: Scenario) -> float:
    """1 − IoU moyen entre les propositions du modèle (NaN si moins de deux)."""
    from happycity.data.pairs import mean_pairwise_iou_distance

    masks = [o.mask for o in scenario.options if "proposition" in o.name]
    return mean_pairwise_iou_distance(masks)


def plot_scenario(scenario: Scenario, path: Path | None = None) -> Any:
    """Figure : état initial, région modifiable R, chaque option (ajouts en vert vif)."""
    import matplotlib.pyplot as plt

    from happycity.data.coverage import composite

    base = composite(scenario.tile)
    panels = [("état initial", base)]
    region = base.copy()
    region[_inputs(scenario.tile)[-1] > 0.5] = (0.98, 0.75, 0.15)
    panels.append(("parcelles aménageables R (orange)", region))
    for o in scenario.options:
        img = base.copy()
        img[o.mask] = (0.05, 0.62, 0.25)
        e = "—" if np.isnan(o.scores.E) else f"{o.scores.E:.2f}"
        flag = "" if o.validity.valid else " · NON VALIDE"
        panels.append((f"{o.name}\nA400 {o.scores.A400:.0%} · E {e}{flag}", img))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.9))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    b = scenario.before
    reach = "oui" if scenario.reachable else "non"
    fig.suptitle(
        f"{scenario.tile.spec.tile_id} · budget {scenario.b_frac:.0%} ({scenario.budget_px} px) · "
        f"cible a = {scenario.a_target:.0%} · avant : A400 {b.A400:.0%} · glouton : "
        f"{scenario.greedy_max:.0%} (cible atteignable par le glouton : {reach})",
        fontsize=9.5,
    )
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    return fig
