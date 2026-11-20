"""Problèmes d'évaluation : (tuile, budget b, cible a), sur les splits de test.

Deux types de cibles :

* **a posteriori** : a = A400 atteint par une solution cible connue (paire de test). Une
  solution valide existe donc toujours ;
* **inatteignable (+δ)** : a = A400 du glouton + δ (δ ∈ ``eval.unreachable_target_offsets``),
  plafonné à 1. Ces cibles sont choisies pour être hors de portée : elles vérifient que
  le modèle reste valide quand on lui demande l'impossible.

La colonne ``atteignable_glouton`` (oui/non) indique si le glouton atteint a. Le glouton
n'étant **pas optimal**, c'est une estimation par le bas de ce qui est atteignable : une
cible a posteriori issue d'un tirage stochastique peut dépasser le glouton, et une cible
« inatteignable par le glouton » n'est donc pas forcément impossible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

TOL = 1e-6


def build_problems(cfg: DictConfig, splits: list[str]) -> pd.DataFrame:
    """Table des problèmes d'évaluation des splits demandés."""
    processed = Path(cfg.paths.processed_dir)
    pairs = pd.read_parquet(processed / "pairs.parquet")
    baselines = pd.read_parquet(processed / "baselines.parquet")
    files = pairs.drop_duplicates("tile_id").set_index("tile_id")["pair_file"]
    base = baselines[baselines["split"].isin(splits)].copy()
    base["pair_file"] = base["tile_id"].map(files)
    keep = ["tile_id", "pair_file", "source", "zone", "split", "b_frac", "budget_px", "saturated",
            "capacity_px", "A400_before", "V_before", "E_before", "greedy_A400"]  # fmt: skip

    hindsight = pairs[(pairs["excluded"] == "") & pairs["split"].isin(splits)]
    hindsight = hindsight[["pair_id", "tile_id", "b_frac", "a_target"]].merge(
        base[keep], on=["tile_id", "b_frac"], how="left", validate="many_to_one"
    )
    hindsight = hindsight.assign(target_type="a posteriori", problem_id=hindsight["pair_id"])

    unreachable = []
    for offset in cfg.eval.unreachable_target_offsets:
        u = base[keep].copy()
        u["a_target"] = np.minimum(u["greedy_A400"] + float(offset), 1.0)
        u = u[u["a_target"] > u["greedy_A400"] + TOL]  # glouton déjà à 100 % : pas de cible
        u["target_type"] = f"inatteignable (+{float(offset):.2f})"
        u["pair_id"] = ""
        u["problem_id"] = [f"{t}_b{b:.2f}_u{float(offset):.2f}"
                           for t, b in zip(u["tile_id"], u["b_frac"], strict=True)]  # fmt: skip
        unreachable.append(u)
    problems = pd.concat([hindsight, *unreachable], ignore_index=True)
    problems["atteignable_glouton"] = np.where(
        problems["a_target"] <= problems["greedy_A400"] + TOL, "oui", "non"
    )
    cols = ["problem_id", "split", "tile_id", "pair_file", "source", "zone", "b_frac",
            "budget_px", "saturated", "capacity_px", "target_type", "a_target", "greedy_A400",
            "atteignable_glouton", "A400_before", "V_before", "E_before", "pair_id"]  # fmt: skip
    return (
        problems[cols]
        .sort_values(["split", "tile_id", "b_frac", "problem_id"])
        .reset_index(drop=True)
    )
