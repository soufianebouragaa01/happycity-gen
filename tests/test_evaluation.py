"""Évaluation : problèmes (cibles atteignables / inatteignables), mesure exacte, agrégation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from happycity.evaluation import report
from happycity.evaluation.problems import build_problems
from happycity.evaluation.runner import evaluate


def test_problems_types_and_reachability(processed: DictConfig) -> None:
    problems = build_problems(processed, ["test_synth"])
    hind = problems[problems["target_type"] == "a posteriori"]
    unreach = problems[problems["target_type"] != "a posteriori"]
    assert len(hind)
    assert len(unreach)
    assert (unreach["atteignable_glouton"] == "non").all()
    assert (unreach["a_target"] <= 1.0).all()
    assert (unreach["a_target"] > unreach["greedy_A400"]).all()
    assert problems["problem_id"].is_unique
    flag = np.where(hind["a_target"] <= hind["greedy_A400"] + 1e-6, "oui", "non")
    assert (hind["atteignable_glouton"] == flag).all()


def test_baselines_are_valid_and_measured_exactly(processed: DictConfig) -> None:
    problems = build_problems(processed, ["test_synth"])
    table = evaluate(processed, problems, runs=[], device="cpu", workers=1)
    assert set(table["method"]) == {"aléatoire", "glouton", "glouton ciblé"}
    assert table["valid"].all()  # les références respectent toujours les contraintes
    assert (table["n_px"] <= table["budget_px"]).all()
    assert (table["A400"] >= table["A400_before"] - 1e-9).all()
    np.testing.assert_allclose(table["abs_err"], (table["A400"] - table["a_target"]).abs())
    greedy = table[(table["method"] == "glouton") & (table["target_type"] == "a posteriori")]
    np.testing.assert_allclose(greedy["A400"], greedy["greedy_A400"], atol=1e-9)
    targeted = table[table["method"] == "glouton ciblé"].set_index("problem_id")
    full = table[table["method"] == "glouton"].set_index("problem_id")
    assert (targeted["n_px"] <= full.loc[targeted.index, "n_px"]).all()  # s'arrête plus tôt


def _fake(rows: list[tuple[str, int, str, float]]) -> pd.DataFrame:
    out = []
    for method, seed, reach, err in rows:
        out.append({"split": "test_synth", "problem_id": f"{reach}{seed}", "target_type": "x",
                    "atteignable_glouton": reach, "b_frac": 0.01, "a_target": 0.5,
                    "method": method, "family": "génératif", "variant": method[0],
                    "ablation": "full", "seed": seed, "output": "projetée", "deployed": True,
                    "abs_err": err, "A400": 0.5, "dA400": 0.1, "E": np.nan, "V": 0.1,
                    "valid": True, "diversity": 0.2, "fill": 0.5, "infer_ms": 1.0})  # fmt: skip
    return pd.DataFrame(out)


def test_seeds_are_aggregated_between_runs() -> None:
    table = _fake([("C-full-s42", 42, "oui", 0.10), ("C-full-s42", 42, "non", 0.30),
                   ("C-full-s43", 43, "oui", 0.20), ("C-full-s43", 43, "non", 0.40)])  # fmt: skip
    s = report.summary(report.problem_level(table), "test_synth").set_index("méthode")
    assert s.loc["C", "graines"] == 2
    assert s.loc["C", "|â−a| atteignable"] == "0.150 ± 0.071"
    assert s.loc["C", "|â−a| global"] == "0.250 ± 0.071"
    assert s.loc["C", "E après"] == "—"  # E indéfini : signalé, jamais remplacé par 0
    assert s.loc["C", "E défini"] == "0%"


def test_labels() -> None:
    row = pd.Series(
        {"family": "génératif", "variant": "C", "ablation": "no_attention", "method": ""}
    )
    assert report.label_of(row) == "C – sans attention"
    assert report.label_of(pd.Series({"family": "référence", "method": "glouton"})) == "glouton"
