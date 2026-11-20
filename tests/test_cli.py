"""Commandes de la CLI ``happycity``."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from happycity.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "show-config",
        "version",
        "extract",
        "coverage-report",
        "synthesize",
        "make-dataset",
        "train",
        "evaluate",
        "demo",
    ]:
        assert command in result.output


def test_show_config_applies_overrides() -> None:
    result = runner.invoke(app, ["show-config", "model=A", "data.source=synthetic"])
    assert result.exit_code == 0, result.output
    assert "cond_mode: concat" in result.output
    assert "source: synthetic" in result.output


def test_invalid_override_gives_clean_error() -> None:
    result = runner.invoke(app, ["show-config", "model=Z"])
    assert result.exit_code == 2


@pytest.mark.slow
def test_synthesize_rasterize_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chaîne complète sans réseau : synthèse → rastérisation → rapport de couverture."""
    import pandas as pd

    monkeypatch.setenv("HAPPYCITY_DATA_DIR", str(tmp_path))
    opts = ["data.synthetic.margin_m=200", "data.source=synthetic"]
    result = runner.invoke(app, ["synthesize", "--n", "2", "--workers", "1", *opts])
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "interim" / "synthetic").iterdir())) == 2
    result = runner.invoke(app, ["rasterize", "--workers", "1", *opts])
    assert result.exit_code == 0, result.output
    index = pd.read_parquet(tmp_path / "processed" / "tiles.parquet")
    assert list(index["source"]) == ["synthetic", "synthetic"]
    assert set(index["split"]) == {"train"}
    assert all((tmp_path / "processed" / "tiles" / f).exists() for f in index["file"])
    assert (tmp_path / "processed" / "figures" / "tiles_synthetic.png").exists()
    result = runner.invoke(app, ["coverage-report", *opts])
    assert result.exit_code == 0, result.output
    report = (tmp_path / "interim" / "coverage" / "coverage.md").read_text(encoding="utf-8")
    assert "Calibrage synthétique" in report
