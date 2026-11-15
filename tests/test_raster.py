"""Rastérisation : cas calculables à la main (128 px × 8 m, origine au coin nord-ouest)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from omegaconf import DictConfig
from shapely.geometry import LineString, Point, box

from conftest import CRS, X0, Y0, make_grid_graph
from happycity.data.layers import UrbanLayers, empty_layer
from happycity.data.network import graph_edges
from happycity.data.raster import (
    TileRaster,
    TileSpec,
    count_raster,
    coverage_raster,
    id_raster,
    line_raster,
    pixel_of,
    rasterize_tile,
    split_conflicts,
    tile_annotations,
    tile_candidates,
    tile_grid,
)

SPEC = TileSpec("t", "osm", "z", "train", X0, Y0, 1024.0)
YMAX = Y0 + 1024.0


# --------------------------------------------------------------------------- grille


def test_tile_grid_counts() -> None:
    zone = box(X0, Y0, X0 + 2048, Y0 + 2048)
    assert len(tile_grid(zone, 1024, 512)) == 9
    assert len(tile_grid(zone, 1024, 1024)) == 4
    assert len(tile_grid(box(X0, Y0, X0 + 1000, Y0 + 2048), 1024, 512)) == 0
    i, j, x, y = tile_grid(zone, 1024, 1024)[-1]
    assert (i, j, x, y) == (1, 1, X0 + 1024, Y0 + 1024)


def test_split_conflicts_use_distance() -> None:
    train = box(X0, Y0, X0 + 2048, Y0 + 2048)
    tile = box(X0 + 1024, Y0, X0 + 2048, Y0 + 1024)  # bord est de la zone train
    far = {"a": (train, "train"), "b": (box(X0 + 2448, Y0, X0 + 4000, Y0 + 2048), "test")}
    near = {"a": (train, "train"), "b": (box(X0 + 2348, Y0, X0 + 4000, Y0 + 2048), "test")}
    assert split_conflicts(tile, "train", far, 400) == []  # exactement 400 m : autorisé
    assert split_conflicts(tile, "train", near, 400) == ["b"]  # 300 m : exclue
    same = {"b": (near["b"][0], "train")}
    assert split_conflicts(tile, "train", same, 400) == []  # même split : pas de conflit


# --------------------------------------------------------------------------- primitives


def test_full_pixels_coverage() -> None:
    # x ∈ [16, 32] → colonnes 2, 3 ; y ∈ [ymax−24, ymax−8] → lignes 1, 2
    building = box(X0 + 16, YMAX - 24, X0 + 32, YMAX - 8)
    cov = coverage_raster([building], SPEC, 128, 4)
    expected = np.zeros((128, 128), dtype=np.float32)
    expected[1:3, 2:4] = 1
    np.testing.assert_array_equal(cov, expected)


def test_half_pixel_coverage() -> None:
    cov = coverage_raster([box(X0, YMAX - 8, X0 + 4, YMAX)], SPEC, 128, 4)
    assert cov[0, 0] == pytest.approx(0.5)
    assert cov.sum() == pytest.approx(0.5)


def test_coverage_preserves_area_on_subgrid() -> None:
    geoms = [box(X0 + 100, Y0 + 300, X0 + 130, Y0 + 342), box(X0 + 500, Y0 + 10, X0 + 502, Y0 + 12)]
    cov = coverage_raster(geoms, SPEC, 128, 4)
    assert cov.sum() * 8 * 8 == pytest.approx(30 * 42 + 4)
    assert cov.max() <= 1


def test_empty_coverage() -> None:
    assert coverage_raster([], SPEC, 128, 4).sum() == 0


def test_line_raster_first_row() -> None:
    line = LineString([(X0, YMAX - 4), (X0 + 1024, YMAX - 4)])
    out = line_raster([line], SPEC, 128)
    assert out[0].sum() == 128
    assert out[1:].sum() == 0


def test_point_pixels_and_counts() -> None:
    x = np.array([X0 + 12, X0 + 12, X0 + 2000])
    y = np.array([YMAX - 20, YMAX - 20, YMAX - 20])
    row, col = pixel_of(x[:1], y[:1], SPEC, 128)
    assert (row[0], col[0]) == (2, 1)
    counts = count_raster(x, y, SPEC, 128)
    assert counts[2, 1] == 2
    assert counts.sum() == 2  # point hors tuile ignoré


def test_id_raster() -> None:
    a = box(X0, YMAX - 16, X0 + 16, YMAX)  # lignes 0–1, colonnes 0–1
    b = box(X0 + 1000, Y0, X0 + 1024, Y0 + 24)  # coin sud-est
    ids = id_raster([a, b], SPEC, 128)
    assert (ids[:2, :2] == 0).all()
    assert (ids[-3:, -3:] == 1).all()
    assert (ids >= 0).sum() == 4 + 9


# --------------------------------------------------------------------------- tuile complète


def _scene(cfg: DictConfig) -> UrbanLayers:
    """Grille de rues tous les 200 m ; deux parcelles, l'une collée à une rue, l'autre isolée."""
    graph = make_grid_graph(n=7, spacing=200.0)  # couvre X0…X0+1200
    zone = box(X0, Y0, X0 + 1024, Y0 + 1024)
    buildings = gpd.GeoDataFrame(
        {"residential": [True, False], "rule": ["explicit"] * 2, "osm_id": ["way/1", "way/2"]},
        geometry=[
            box(X0 + 16, YMAX - 24, X0 + 32, YMAX - 8),
            box(X0 + 600, Y0 + 600, X0 + 640, Y0 + 640),
        ],
        crs=CRS,
    )
    candidates = gpd.GeoDataFrame(
        {
            "candidate_id": [0, 1],
            "rule": ["landuse=vacant", "landuse=grass"],
            "area_m2": [1600.0, 1600.0],
            "osm_id": [None, None],
        },
        geometry=[
            box(X0 + 205, Y0 + 205, X0 + 245, Y0 + 245),  # à 5 m des rues → adjacente
            box(X0 + 480, Y0 + 480, X0 + 520, Y0 + 520),  # à 80 m des rues → isolée
        ],
        crs=CRS,
    )
    origins = gpd.GeoDataFrame(
        {"building_id": [0], "node": [0], "snap_m": [10.0]},
        geometry=[Point(X0 + 24, YMAX - 16)],
        crs=CRS,
    )
    return UrbanLayers(
        name="scene",
        source="osm",
        crs=CRS,
        zone=zone,
        graph=graph,
        buildings=buildings,
        green=empty_layer("green", CRS),
        green_excluded=empty_layer("green_excluded", CRS),
        candidates=candidates,
        obstacles=empty_layer("obstacles", CRS),
        origins=origins,
        entrances=empty_layer("entrances", CRS),
    )


def test_candidate_adjacency(cfg: DictConfig) -> None:
    layers = _scene(cfg)
    ids, table = tile_candidates(layers, SPEC, graph_edges(layers.graph), cfg.data)
    assert list(table["adjacent"]) == [True, False]
    assert list(table["n_px"]) == [25, 25]  # 40 m = 5 px de côté
    assert (ids >= 0).sum() == 50


def test_rasterize_tile_scene(cfg: DictConfig) -> None:
    tile = rasterize_tile(_scene(cfg), SPEC, cfg.data)
    assert tile.channels.shape == (len(cfg.data.channels), 128, 128)
    assert tile.channel("residential_buildings")[1:3, 2:4].min() == 1
    assert tile.channel("other_buildings").sum() == pytest.approx(25)  # 40×40 m
    assert tile.channel("origins")[2, 3] == 1
    # rue à y = Y0 + 200 : limite entre les lignes 102 et 103 (824 m / 8 m) → traverse la tuile
    assert tile.channel("walk_network")[102:104].max(axis=0).sum() == 128
    assert list(tile.origins[["row", "col"]].iloc[0]) == [2, 3]


def test_annotations_flags(cfg: DictConfig) -> None:
    layers = _scene(cfg)
    stats = tile_annotations(layers, rasterize_tile(layers, SPEC, cfg.data), cfg.data)
    assert stats["n_candidates"] == 2
    assert stats["n_candidates_adjacent"] == 1
    assert stats["usable"] is False  # une seule origine
    assert "peu d'origines" in stats["flags"]
    assert "aucun espace vert accessible" in stats["flags"]


def test_tile_roundtrip(small_synthetic: UrbanLayers, cfg: DictConfig, tmp_path: Path) -> None:
    xmin, ymin, _, _ = small_synthetic.zone.bounds
    spec = TileSpec(
        small_synthetic.name, "synthetic", small_synthetic.name, "train", xmin, ymin, 1024.0
    )
    tile = rasterize_tile(small_synthetic, spec, cfg.data)
    assert tile.channel("origins").sum() == len(tile.origins)
    assert int(tile.candidate_mask.sum()) == int(tile.candidates["n_px"].sum())
    tile.save(tmp_path / "t.npz")
    back = TileRaster.load(tmp_path / "t.npz")
    assert back.spec == tile.spec
    assert back.channel_names == tile.channel_names
    np.testing.assert_allclose(back.channels, tile.channels, atol=1e-3)  # float16
    np.testing.assert_array_equal(back.candidate_ids, tile.candidate_ids)
    assert back.candidates.equals(tile.candidates.astype(back.candidates.dtypes.to_dict()))
    assert len(back.origins) == len(tile.origins)


def test_scale_is_enforced(small_synthetic: UrbanLayers, cfg: DictConfig) -> None:
    """Échelle constante : une tuile de 512 m en 128 px (4 m/px) est refusée."""
    xmin, ymin, _, _ = small_synthetic.zone.bounds
    spec = TileSpec("x", "synthetic", "x", "train", xmin, ymin, 512.0)
    with pytest.raises(ValueError, match="échelle"):
        rasterize_tile(small_synthetic, spec, cfg.data)
