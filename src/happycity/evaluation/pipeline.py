"""Étape ``happycity evaluate`` : problèmes, propositions, métriques exactes, rapport."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from happycity.data import coverage
from happycity.data.pairs import make_problem, stable_seed
from happycity.data.pipeline import context_dir, tiles_dir
from happycity.data.raster import TileRaster
from happycity.evaluation import condition, figures, report
from happycity.evaluation.problems import build_problems
from happycity.evaluation.runner import RunInfo, _inputs, discover_runs, evaluate
from happycity.metrics import TileAccessContext, evaluate_tile

log = logging.getLogger(__name__)

SPLITS = ["test_synth", "test"]
SPLIT_TITLES = {
    "test_synth": "Test synthétique (rapporté à part)",
    "test": "Test réel : Guéliz (test final)",
}


def evaluation_dir(cfg: DictConfig) -> Path:
    """Dossier des résultats d'évaluation."""
    return Path(cfg.paths.experiments_dir) / "evaluation"


def run_evaluate(
    cfg: DictConfig,
    pattern: str = "*",
    splits: list[str] | None = None,
    workers: int | None = None,
    device: str | None = None,
) -> Path:
    """Évalue toutes les exécutions trouvées et les références ; écrit le rapport."""
    import torch

    splits = splits or SPLITS
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = evaluation_dir(cfg)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = discover_runs(Path(cfg.paths.experiments_dir), pattern)
    problems = build_problems(cfg, splits)
    log.info("%d exécutions, %d problèmes", len(runs), len(problems))
    table = evaluate(cfg, problems, runs, dev, int(workers or cfg.data.raster.workers))
    problems.to_parquet(out / "problems.parquet", index=False)
    table.to_parquet(out / "results.parquet", index=False)
    meta = {"runs": [r.name for r in runs], "device": dev, "n_proposals": int(cfg.eval.n_proposals),
            "random_draws": int(cfg.eval.random_baseline_draws), "splits": splits}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    full = [r for r in runs if r.ablation in ("full", "none")]
    if full and "test_synth" in splits:
        probe = condition.condition_response(cfg, problems, full, "test_synth", dev)
        probe.to_parquet(out / "condition_probe.parquet", index=False)
    write_report(cfg, table, problems, runs, out)
    return out


def write_report(
    cfg: DictConfig, table: pd.DataFrame, problems: pd.DataFrame, runs: list[RunInfo], out: Path
) -> None:
    """Rapport Markdown + tableaux CSV + figures."""
    pl = report.problem_level(table)
    splits = [s for s in SPLITS if s in set(problems["split"])]
    n_prob = (
        problems.groupby(["split", "target_type", "atteignable_glouton"]).size().rename("problèmes")
    )
    parts = [
        "# Évaluation : modèles A/B/C, ablations et références",
        "",
        "Métriques **exactes** : module indépendant, distances réseau (Dijkstra). E est toujours",
        "lu avec A400. Ces indicateurs mesurent l'accès piéton à 400 m et la végétation ; ils ne",
        "mesurent pas le bonheur des habitants.",
        "",
        f"Exécutions : {', '.join(r.name for r in runs) or 'aucune'}. "
        f"{int(cfg.eval.n_proposals)} propositions par problème (mêmes z pour tous les modèles) ; "
        f"aléatoire : {int(cfg.eval.random_baseline_draws)} tirages.",
        "",
        "## Problèmes",
        "",
        coverage.to_markdown(n_prob.reset_index()),
        "",
        "« Atteignable » = atteint par le glouton. Le glouton n'est pas optimal : c'est une",
        "estimation **par le bas** de ce qui est atteignable.",
        "",
    ]
    for split in splits:
        s = report.summary(pl, split)
        s.to_csv(out / f"summary_{split}.csv", index=False)
        v = report.violations(table, split).round(3)
        e = report.projection_effect(pl, split).round(3)
        parts += [f"## {SPLIT_TITLES.get(split, split)}", "", "### Sorties retenues", "",
                  coverage.to_markdown(s), "",
                  "Moyenne ± écart-type **entre graines** quand il y en a plusieurs. « valide » : "
                  "dans R, budget respecté, ajouts adjacents au réseau.", "",
                  "### Violations des sorties brutes (avant projection)", "",
                  coverage.to_markdown(v), "",
                  "### Effet de la projection (mêmes propositions, brutes vs projetées)", "",
                  coverage.to_markdown(e), ""]  # fmt: skip
        labels = [
            lab for lab in ("A", "B", "C", "aléatoire", "glouton ciblé") if lab in set(pl["label"])
        ]
        b = report.by_budget(pl, split, labels).round(3)
        parts += ["### Par budget", "", coverage.to_markdown(b), ""]
    (out / "report.md").write_text("\n".join(parts), encoding="utf-8")

    figures.plot_abs_error(pl, splits, out / "figures" / "abs_error.png")
    labels = [
        lab for lab in ("A", "B", "C", "glouton ciblé", "aléatoire") if lab in set(pl["label"])
    ]
    figures.plot_condition(pl, splits[0], labels, out / "figures" / f"condition_{splits[0]}.png")
    c_runs = [r for r in runs if r.variant == "C" and r.ablation in ("full", "none")]
    if c_runs:
        for split in splits:
            plot_example(cfg, problems, table, sorted(c_runs, key=lambda r: r.seed)[0], split,
                         out / "figures" / f"example_{split}.png")  # fmt: skip


def plot_example(
    cfg: DictConfig, problems: pd.DataFrame, table: pd.DataFrame, run: RunInfo, split: str,
    path: Path,
) -> None:  # fmt: skip
    """Deux propositions de C et le glouton ciblé sur un problème a posteriori représentatif.

    Choix : budget 2 %, la tuile dont l'erreur de C est la plus proche de sa médiane (ni le
    meilleur ni le pire cas).
    """
    from happycity.baselines import targeted_greedy_placement
    from happycity.data.access_context import metric_params
    from happycity.models.inference import load_checkpoint, propose

    d = table[(table["split"] == split) & (table["method"] == run.name) & table["deployed"]
              & (table["target_type"] == "a posteriori") & (table["b_frac"] == 0.02)]  # fmt: skip
    if d.empty:
        return
    err = d.groupby("problem_id")["abs_err"].mean()
    pid = str((err - err.median()).abs().idxmin())
    p = problems[problems["problem_id"] == pid].to_dict("records")[0]
    tile = TileRaster.load(tiles_dir(cfg) / str(p["pair_file"]))
    ctx = TileAccessContext.load(context_dir(cfg) / str(p["pair_file"]))
    params = metric_params(cfg.data)
    nets, state = load_checkpoint(run.checkpoint)
    props = propose(nets.generator, _inputs(tile), float(p["b_frac"]), float(p["a_target"]),
                    int(p["budget_px"]), n=int(cfg.eval.n_proposals),
                    seed=stable_seed(cfg.eval.seed, pid),
                    threshold=float(state["config"]["model"]["threshold"]))  # fmt: skip
    masks = [props[0].projection.mask, props[1].projection.mask]
    greedy = targeted_greedy_placement(make_problem(tile, ctx, params), int(p["budget_px"]),
                                       float(p["a_target"])).mask  # fmt: skip
    titles = []
    for name, m in (
        ("C, proposition 1", masks[0]),
        ("C, proposition 2", masks[1]),
        ("glouton ciblé", greedy),
    ):
        s = evaluate_tile(ctx, m, params=params)
        titles.append(f"{name} : A400 {s.A400:.0%} (cible {p['a_target']:.0%}), E {s.E:.2f}")
    coverage.plot_pairs(tile, [*masks, greedy], titles, path)
    log.info(
        "exemple %s : %s (A400 avant %.0f %%)",
        split,
        pid,
        100 * float(np.asarray(p["A400_before"])),
    )
