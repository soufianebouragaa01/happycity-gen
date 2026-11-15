"""Générateur synthétique, format commun des couches et rapport de couverture."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from omegaconf import DictConfig

from happycity.config import load_config
from happycity.data import coverage
from happycity.data.layers import LAYER_COLUMNS, UrbanLayers
from happycity.data.synthetic import choose_typology, generate_map, map_seed

SMALL = ["data.synthetic.margin_m=150"]


def test_map_seed_is_stable_and_distinct() -> None:
    assert map_seed(0, 3) == map_seed(0, 3)
    assert len({map_seed(0, i) for i in range(50)}) == 50
    assert map_seed(0, 3) != map_seed(1, 3)


def test_generation_is_reproducible() -> None:
    data = load_config(SMALL).data
    a, b = generate_map(4, data), generate_map(4, data)
    assert a.meta == b.meta
    assert a.buildings.geometry.equals(b.buildings.geometry)
    assert nx.utils.graphs_equal(a.graph, b.graph)
    assert generate_map(5, data).meta["seed"] != a.meta["seed"]


def test_typology_weights(cfg: DictConfig) -> None:
    rng = np.random.default_rng(0)
    draws = [choose_typology(rng, cfg.data.synthetic.typologies) for _ in range(3000)]
    for name, typ in cfg.data.synthetic.typologies.items():
        assert draws.count(name) / len(draws) == pytest.approx(typ.weight, abs=0.04)


def test_synthetic_invariants(small_synthetic: UrbanLayers, cfg: DictConfig) -> None:
    layers = small_synthetic
    layers.validate()
    assert layers.source == "synthetic"
    assert nx.is_weakly_connected(layers.graph)
    assert not layers.buildings.empty
    assert not layers.origins.empty
    assert (layers.candidates["area_m2"] >= cfg.data.candidates.min_area_m2).all()
    if not layers.candidates.empty:
        overlap = layers.candidates.intersection(layers.buildings.union_all()).area
        assert overlap.max() < 1e-6
    assert set(layers.entrances["green_id"]) <= set(layers.green["green_id"])
    assert set(layers.origins["node"]) <= set(layers.graph.nodes)


def test_save_load_roundtrip(small_synthetic: UrbanLayers, tmp_path: Path) -> None:
    small_synthetic.save(tmp_path / "map")
    loaded = UrbanLayers.load(tmp_path / "map")
    for name in LAYER_COLUMNS:
        assert len(loaded.layer(name)) == len(small_synthetic.layer(name)), name
    assert loaded.zone.equals(small_synthetic.zone)
    assert loaded.graph.number_of_edges() == small_synthetic.graph.number_of_edges()
    assert loaded.meta == small_synthetic.meta


def test_validate_rejects_missing_columns(small_synthetic: UrbanLayers) -> None:
    broken = UrbanLayers(
        **{**small_synthetic.__dict__, "origins": small_synthetic.origins.drop(columns="node")}
    )
    with pytest.raises(ValueError, match="origins"):
        broken.validate()


def test_validate_rejects_wrong_crs(small_synthetic: UrbanLayers) -> None:
    broken = UrbanLayers(
        **{**small_synthetic.__dict__, "green": small_synthetic.green.to_crs(4326)}
    )
    with pytest.raises(ValueError, match="green"):
        broken.validate()


def test_zone_stats_are_consistent(small_synthetic: UrbanLayers, cfg: DictConfig) -> None:
    stats = coverage.zone_stats(small_synthetic)
    assert stats["area_km2"] == pytest.approx(1.024**2, rel=1e-3)
    assert 0 < stats["built_ratio"] < 1
    assert 0 <= stats["V"] < 1
    assert stats["n_origins"] <= len(small_synthetic.origins)
    assert stats["walk_density_km_per_km2"] > 0
    table = coverage.coverage_table([small_synthetic], cfg.data.coverage)
    assert list(table["zone"]) == [small_synthetic.name]


def test_coverage_flags(cfg: DictConfig) -> None:
    ok = {"built_ratio": 0.3, "n_origins": 500, "n_candidates": 10, "walk_density_km_per_km2": 20}
    assert coverage.coverage_flags(ok, cfg.data.coverage) == []
    poor = {**ok, "built_ratio": 0.01, "n_candidates": 0}
    assert coverage.coverage_flags(poor, cfg.data.coverage) == [
        "faible emprise bâtie",
        "trop peu de parcelles candidates",
    ]


def test_plots_are_written(small_synthetic: UrbanLayers, tmp_path: Path) -> None:
    coverage.plot_layers(small_synthetic, tmp_path / "map.png")
    coverage.plot_overview([small_synthetic], {small_synthetic.name: "train"}, tmp_path / "ov.png")
    assert (tmp_path / "map.png").stat().st_size > 10_000
    assert (tmp_path / "ov.png").exists()


def test_to_markdown() -> None:
    import pandas as pd

    md = coverage.to_markdown(pd.DataFrame({"a": [1], "b": ["x"]}))
    assert md.splitlines() == ["| a | b |", "|---|---|", "| 1 | x |"]
