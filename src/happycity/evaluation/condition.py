"""Test de réponse à la condition : le modèle utilise-t-il vraiment la cible a ?

Une faible erreur |â − a| moyenne ne suffit pas : si les cibles testées sont proches de
l'A400 maximal atteignable, un modèle qui **maximise** l'accès en ignorant a tombe juste.
Ce test demande, pour une même tuile et un même budget, deux cibles :

* **basse** : A400 avant + 25 % de l'écart jusqu'au glouton ;
* **haute** : A400 atteint par le glouton.

La **pente** = (A400 obtenu haut − A400 obtenu bas) / (cible haute − cible basse) vaut 1 pour
un modèle qui suit parfaitement a, 0 pour un modèle qui l'ignore. On ne retient que les cas
où l'écart entre les cibles est d'au moins 0,10. Le glouton ciblé sert de référence : ses
ajouts sont des parcelles entières, ce qui limite sa pente.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from happycity.baselines import targeted_greedy_placement
from happycity.data.access_context import metric_params
from happycity.data.pairs import make_problem, stable_seed
from happycity.data.pipeline import context_dir, tiles_dir
from happycity.data.raster import TileRaster
from happycity.evaluation.runner import RunInfo, _inputs
from happycity.metrics import TileAccessContext, evaluate_tile

log = logging.getLogger(__name__)

LOW_FRACTION = 0.25
MIN_GAP = 0.10
BUDGETS = (0.02, 0.03)


def condition_response(
    cfg: DictConfig, problems: pd.DataFrame, runs: list[RunInfo], split: str, device: str
) -> pd.DataFrame:
    """Une ligne par (exécution, tuile, budget, cible basse/haute) ; A400 exacts."""
    from happycity.models.inference import load_checkpoint, propose

    params = metric_params(cfg.data)
    cases = (
        problems[(problems["split"] == split) & problems["b_frac"].isin(BUDGETS)]
        .drop_duplicates(["tile_id", "b_frac"])
        .loc[lambda d: d["greedy_A400"] - d["A400_before"] >= MIN_GAP]
    )
    n = int(cfg.eval.n_proposals)
    rows: list[dict[str, Any]] = []
    tiles: dict[str, tuple[TileRaster, TileAccessContext, np.ndarray]] = {}
    for c in cases.to_dict("records"):
        if c["pair_file"] not in tiles:
            tile = TileRaster.load(tiles_dir(cfg) / c["pair_file"])
            ctx = TileAccessContext.load(context_dir(cfg) / c["pair_file"])
            tiles[c["pair_file"]] = (tile, ctx, _inputs(tile))
        tile, ctx, _ = tiles[c["pair_file"]]
        low = c["A400_before"] + LOW_FRACTION * (c["greedy_A400"] - c["A400_before"])
        problem = make_problem(tile, ctx, params)
        for name, a in (("basse", low), ("haute", c["greedy_A400"])):
            ref = targeted_greedy_placement(problem, int(c["budget_px"]), float(a))
            score = evaluate_tile(ctx, ref.mask, params=params).A400
            rows.append({"method": "glouton ciblé", "variant": "", "seed": -1,
                         "tile_id": c["tile_id"], "b_frac": c["b_frac"], "cible": name,
                         "a": a, "A400": score})  # fmt: skip
    for run in runs:
        nets, state = load_checkpoint(run.checkpoint, device)
        thr = float(state["config"]["model"]["threshold"])
        for c in cases.to_dict("records"):
            tile, ctx, x = tiles[c["pair_file"]]
            low = c["A400_before"] + LOW_FRACTION * (c["greedy_A400"] - c["A400_before"])
            for name, a in (("basse", low), ("haute", c["greedy_A400"])):
                z_seed = stable_seed(cfg.eval.seed, c["tile_id"], c["b_frac"])  # même z : bas/haut
                props = propose(nets.generator, x, float(c["b_frac"]), float(a),
                                int(c["budget_px"]), n=n, threshold=thr, seed=z_seed)  # fmt: skip
                a400 = np.mean([evaluate_tile(ctx, p.deployed(run.projection), params=params).A400
                                for p in props])  # fmt: skip
                rows.append({"method": run.name, "variant": run.variant, "seed": run.seed,
                             "tile_id": c["tile_id"], "b_frac": c["b_frac"], "cible": name,
                             "a": a, "A400": float(a400)})  # fmt: skip
        log.info("réponse à la condition : %s", run.name)
    return pd.DataFrame(rows)


def slopes(probe: pd.DataFrame) -> pd.DataFrame:
    """Pente et erreurs par (méthode, tuile, budget)."""
    keys = ["method", "variant", "seed", "tile_id", "b_frac"]
    low = probe[probe["cible"] == "basse"].set_index(keys)
    high = probe[probe["cible"] == "haute"].set_index(keys)
    out = pd.DataFrame({
        "pente": (high["A400"] - low["A400"]) / (high["a"] - low["a"]),
        "err_basse": (low["A400"] - low["a"]).abs(),
        "err_haute": (high["A400"] - high["a"]).abs(),
    }).dropna().reset_index()  # fmt: skip
    return out


def summary(probe: pd.DataFrame) -> pd.DataFrame:
    """Tableau : pente moyenne (± écart-type entre graines), part sans réaction, erreurs."""
    s = slopes(probe)
    s["label"] = np.where(s["variant"] == "", s["method"], s["variant"])
    s = s[s["method"].str.contains("-full-") | (s["variant"] == "")]  # variantes complètes
    per_run = s.groupby(["label", "method"]).agg(
        pente=("pente", "mean"), sans_reaction=("pente", lambda p: float((p < 0.1).mean())),
        err_basse=("err_basse", "mean"), err_haute=("err_haute", "mean"), cas=("pente", "size"),
    ).reset_index()  # fmt: skip
    rows = []
    for label, g in per_run.groupby("label", sort=False):

        def fmt(col: str, digits: int = 2, gg: pd.DataFrame = g) -> str:
            m, sd = gg[col].mean(), gg[col].std(ddof=1)
            return f"{m:.{digits}f}" + (f" ± {sd:.{digits}f}" if len(gg) > 1 else "")

        rows.append({"méthode": label, "graines": len(g) if label != "glouton ciblé" else "—",
                     "cas": int(g["cas"].iloc[0]), "pente (idéal 1)": fmt("pente"),
                     "sans réaction (pente < 0,1)": f"{g['sans_reaction'].mean():.0%}",
                     "|â−a| cible basse": fmt("err_basse", 3),
                     "|â−a| cible haute": fmt("err_haute", 3)})  # fmt: skip
    return pd.DataFrame(rows)
