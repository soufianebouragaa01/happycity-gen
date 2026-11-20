"""Démonstration : module partagé, commande ``happycity demo`` et interface Streamlit."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from typer.testing import CliRunner

from happycity import demo
from happycity.cli import app
from happycity.config import load_config
from happycity.models import build_networks
from happycity.models.inference import save_checkpoint

matplotlib.use("Agg")
APP_FILE = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def with_model(processed: DictConfig) -> DictConfig:
    """Ajoute un petit modèle C (non entraîné) rangé comme les vrais : seed42/C-full-s42."""
    cfg = load_config(["model=C", "model.ngf=8", "model.ndf=8",
                       f"paths.data_dir={processed.paths.data_dir}",
                       f"paths.experiments_dir={processed.paths.experiments_dir}"])  # fmt: skip
    run = Path(cfg.paths.experiments_dir) / "seed42" / "C-full-s42"
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    nets = build_networks(cfg.model, 8)
    extra = {"config": OmegaConf.to_container(cfg, resolve=True), "in_channels": 8}
    save_checkpoint(run / "checkpoints" / "best.pt", nets, extra)
    return cfg


def test_choices(with_model: DictConfig) -> None:
    tiles = demo.available_tiles(with_model)
    assert len(tiles) == 1
    assert {"greedy_0.01", "greedy_0.05", "pair_file", "A400_before"} <= set(tiles.columns)
    runs = demo.available_runs(with_model)
    chosen = demo.default_run(runs)
    assert chosen is not None
    assert chosen.name == "C-full-s42"
    assert chosen.projection


def test_scenario_is_valid_and_exact(with_model: DictConfig) -> None:
    tiles = demo.available_tiles(with_model)
    row = tiles.iloc[0]
    run = demo.default_run(demo.available_runs(with_model))
    assert run is not None
    s = demo.run_scenario(with_model, run, row["pair_file"], 0.02, float(row["greedy_0.02"]), n=2)
    assert len(s.options) == 3  # 2 propositions + glouton ciblé
    assert s.reachable
    assert s.greedy_max == pytest.approx(float(row["greedy_0.02"]))
    for option in s.options:
        assert option.validity.valid  # C est projeté ; la référence est valide par construction
        assert option.mask.sum() <= s.budget_px
    table = demo.scenario_table(s)
    assert table["option"].iloc[0] == "état initial"
    assert table["A400"].iloc[0] == pytest.approx(float(row["A400_before"]))
    assert {"A400", "E", "V", "|â − a|", "valide"} <= set(table.columns)
    assert demo.plot_scenario(s) is not None
    again = demo.run_scenario(
        with_model, run, row["pair_file"], 0.02, float(row["greedy_0.02"]), n=2
    )
    for a, b in zip(s.options, again.options, strict=True):
        np.testing.assert_array_equal(a.mask, b.mask)  # mêmes graines → même scénario


def test_cli_demo_writes_figure_and_table(with_model: DictConfig) -> None:
    tile_id = str(demo.available_tiles(with_model)["tile_id"].iloc[0])
    result = CliRunner().invoke(app, [
        "demo", f"paths.data_dir={with_model.paths.data_dir}",
        f"paths.experiments_dir={with_model.paths.experiments_dir}",
        "--tile", tile_id, "--budget", "0.02", "--n", "2",
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    out = Path(with_model.paths.experiments_dir) / "demo"
    assert list(out.glob(f"{tile_id}_*.png"))
    assert list(out.glob(f"{tile_id}_*.csv"))
    assert "bien-être" in result.output  # l'avertissement est toujours affiché


def test_cli_demo_unknown_tile(with_model: DictConfig) -> None:
    result = CliRunner().invoke(app, ["demo", f"paths.data_dir={with_model.paths.data_dir}",
                                      f"paths.experiments_dir={with_model.paths.experiments_dir}",
                                      "--tile", "inexistante"])  # fmt: skip
    assert result.exit_code == 2


def test_streamlit_app_runs(with_model: DictConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    paths = with_model.paths
    argv = ["streamlit_app.py", f"paths.data_dir={paths.data_dir}",
            f"paths.experiments_dir={paths.experiments_dir}"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    at = AppTest.from_file(str(APP_FILE), default_timeout=300).run()
    assert not at.exception
    assert at.metric[0].label == "A400 actuel"
    at.button[0].click().run()
    assert not at.exception
    table = at.dataframe[0].value
    assert table["option"].iloc[0] == "état initial"
    assert table["valide"].all()


def test_app_uses_only_the_shared_module() -> None:
    """L'interface n'a pas de logique propre : seuls happycity.demo et la config."""
    tree = ast.parse(APP_FILE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    ours = {m for m in imported if m.startswith("happycity")}
    assert ours == {"happycity.demo", "happycity.config.load_config"}
