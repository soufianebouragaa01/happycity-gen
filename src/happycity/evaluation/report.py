"""Tableaux de résultats à partir de la table des propositions (``results.parquet``).

Règles de présentation (cahier des charges) :

* E est toujours rapporté **avec** A400 (une équité élevée peut signifier « tout le monde
  aussi mal desservi ») ;
* |â − a| est rapporté tel quel, séparément pour les cibles atteignables par le glouton,
  non atteignables, et au global ;
* les variantes à plusieurs graines sont résumées par moyenne ± écart-type **entre graines**
  (chaque graine étant d'abord moyennée sur les problèmes) ;
* ces indicateurs mesurent l'accessibilité piétonne à 400 m et la couverture végétale, pas le
  bonheur des habitants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["abs_err", "A400", "dA400", "E", "V", "valid", "diversity", "fill", "infer_ms"]
BASELINE_ORDER = ["aléatoire", "glouton", "glouton ciblé"]
COLUMNS = [  # (colonne, titre, en pourcentage)
    ("A400", "A400 après", True),
    ("dA400", "ΔA400", True),
    ("E", "E après", False),
    ("V", "V après", True),
    ("valid", "valide", True),
    ("diversity", "diversité", False),
]


def label_of(row: pd.Series) -> str:
    """Nom lisible : « A », « B », « C », « C sans attention »… ou nom de la référence."""
    if row["family"] != "génératif":
        return str(row["method"])
    if row["ablation"] in ("full", "none", ""):
        return str(row["variant"])
    names = {"no_attention": "sans attention", "no_diversity": "sans diversité",
             "no_modulation": "sans modulation", "no_projection": "sans projection"}  # fmt: skip
    ablation = str(row["ablation"])
    return f"{row['variant']} – {names.get(ablation, ablation)}"


def problem_level(table: pd.DataFrame) -> pd.DataFrame:
    """Moyenne sur les propositions : une ligne par (problème, méthode, sortie)."""
    t = table.assign(label=table.apply(label_of, axis=1), E_defined=table["E"].notna())
    keys = ["split", "problem_id", "target_type", "atteignable_glouton", "b_frac", "a_target",
            "label",
            "method", "family", "seed", "output", "deployed"]  # fmt: skip
    return t.groupby(keys, dropna=False)[[*METRICS, "E_defined"]].mean().reset_index()


def _per_run(pl: pd.DataFrame) -> pd.DataFrame:
    """Moyenne sur les problèmes, par exécution (une graine = une exécution)."""
    return (
        pl.groupby(["label", "method", "family", "seed"])[[*METRICS, "E_defined"]]
        .mean()
        .reset_index()
    )


def _fmt(mean: float, std: float, n: int, pct: bool = False, digits: int = 3) -> str:
    if np.isnan(mean):
        return "—"
    scale = 100 if pct else 1
    text = f"{mean * scale:.{1 if pct else digits}f}"
    if n > 1 and not np.isnan(std):
        text += f" ± {std * scale:.{1 if pct else digits}f}"
    return text + (" %" if pct else "")


def summary(pl: pd.DataFrame, split: str, deployed_only: bool = True) -> pd.DataFrame:
    """Tableau principal d'un split : une ligne par méthode (graines agrégées)."""
    d = pl[(pl["split"] == split) & (pl["deployed"] if deployed_only else True)]
    parts = {
        "global": d,
        "atteignable": d[d["atteignable_glouton"] == "oui"],
        "non atteignable": d[d["atteignable_glouton"] == "non"],
    }
    runs = {k: _per_run(v) for k, v in parts.items() if len(v)}
    rows = []
    for label, g in runs["global"].groupby("label", sort=False):
        n = len(g)
        row: dict[str, object] = {
            "méthode": label,
            "graines": n if g["family"].iloc[0] == "génératif" else "—",
        }
        for name, r in runs.items():
            sub = r[r["label"] == label]["abs_err"]
            row[f"|â−a| {name}"] = _fmt(sub.mean(), sub.std(ddof=1), len(sub))
        for col, title, pct in COLUMNS:
            row[title] = _fmt(g[col].mean(), g[col].std(ddof=1), n, pct)
        row["E défini"] = f"{g['E_defined'].mean():.0%}"
        ms = g["infer_ms"].mean()
        row["ms / carte"] = "—" if np.isnan(ms) else f"{ms:.1f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(["A", "B", "C"])}
    out["_k"] = [order.get(m, 10 if m in BASELINE_ORDER else 5) for m in out["méthode"]]
    out["_b"] = [BASELINE_ORDER.index(m) if m in BASELINE_ORDER else 0 for m in out["méthode"]]
    return (
        out.sort_values(["_k", "_b", "méthode"]).drop(columns=["_k", "_b"]).reset_index(drop=True)
    )


def violations(table: pd.DataFrame, split: str) -> pd.DataFrame:
    """Sorties brutes des modèles : part des propositions qui violent chaque contrainte."""
    t = table[
        (table["split"] == split) & (table["family"] == "génératif") & (table["output"] == "brute")
    ]
    t = t.assign(label=t.apply(label_of, axis=1))
    g = t.groupby("label")
    return (
        pd.DataFrame(
            {
                "hors de R": g["outside_r_px"].apply(lambda s: (s > 0).mean()),
                "budget dépassé": g["over_budget_px"].apply(lambda s: (s > 0).mean()),
                "morceaux détachés": g["detached_px"].apply(lambda s: (s > 0).mean()),
                "valide": g["valid"].mean(),
                "px hors de R (moy.)": g["outside_r_px"].mean(),
            }
        )
        .reset_index()
        .rename(columns={"label": "méthode"})
    )


def projection_effect(pl: pd.DataFrame, split: str) -> pd.DataFrame:
    """Brut vs projeté pour chaque variante (même z) : validité, |â − a|, A400."""
    d = pl[(pl["split"] == split) & (pl["family"] == "génératif")]
    g = (
        d.groupby(["label", "output"])[["valid", "abs_err", "A400", "fill"]]
        .mean()
        .unstack("output")
    )
    g = pd.DataFrame(g)
    g.columns = pd.Index([f"{c[0]} ({c[1]})" for c in list(g.columns)])
    return g.reset_index().rename(columns={"label": "méthode"})


def by_budget(pl: pd.DataFrame, split: str, labels: list[str]) -> pd.DataFrame:
    """|â − a| et A400 par budget, pour quelques méthodes (sorties retenues)."""
    d = pl[(pl["split"] == split) & pl["deployed"] & pl["label"].isin(labels)]
    g = d.groupby(["b_frac", "label"])[["abs_err", "A400"]].mean().unstack("label")
    g = pd.DataFrame(g)
    g.columns = pd.Index([f"{c[0]} – {c[1]}" for c in list(g.columns)])
    return g.reset_index()
