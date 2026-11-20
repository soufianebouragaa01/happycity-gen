"""Fixtures partagées : configuration, mini-graphes et objets OSM factices (sans réseau)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import matplotlib
import networkx as nx
import pytest
from omegaconf import DictConfig
from shapely.geometry import LineString

from happycity.config import load_config
from happycity.data.layers import UrbanLayers
from happycity.data.osm import RELEVANT_KEYS
from happycity.data.synthetic import generate_map

matplotlib.use("Agg")  # tests sans écran

CRS = "EPSG:32629"
X0, Y0 = 500_000.0, 3_500_000.0


@pytest.fixture(scope="session")
def cfg() -> DictConfig:
    return load_config()


def make_grid_graph(
    n: int = 5, spacing: float = 100.0, x0: float = X0, y0: float = Y0
) -> nx.MultiDiGraph:
    """Grille n×n bidirectionnelle (format OSMnx projeté)."""
    graph = nx.MultiDiGraph(crs=CRS)
    for i in range(n):
        for j in range(n):
            graph.add_node(i * n + j, x=x0 + i * spacing, y=y0 + j * spacing)
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < n and j + dj < n:
                    a, b = i * n + j, (i + di) * n + (j + dj)
                    pa, pb = graph.nodes[a], graph.nodes[b]
                    line = LineString([(pa["x"], pa["y"]), (pb["x"], pb["y"])])
                    for u, v in ((a, b), (b, a)):
                        graph.add_edge(
                            u, v, key=0, length=spacing, highway="residential", geometry=line
                        )
    return graph


@pytest.fixture
def grid_graph() -> Callable[..., nx.MultiDiGraph]:
    return make_grid_graph


def make_features(rows: list[tuple[dict[str, Any], Any]]) -> gpd.GeoDataFrame:
    """Objets au format de ``prepare_features`` à partir de (tags, géométrie)."""
    data: dict[str, list[Any]] = {k: [] for k in ("osm_id", *RELEVANT_KEYS)}
    for idx, (tags, _) in enumerate(rows):
        data["osm_id"].append(tags.get("osm_id", f"way/{idx}"))
        for key in RELEVANT_KEYS:
            data[key].append(tags.get(key, math.nan))
    return gpd.GeoDataFrame(data, geometry=[g for _, g in rows], crs=CRS)


@pytest.fixture(scope="session")
def small_synthetic() -> UrbanLayers:
    """Carte synthétique (une tuile de 1024 m, contexte réduit à 150 m) pour des tests rapides."""
    small = load_config(["data.synthetic.margin_m=150"])
    return generate_map(1, small.data)


@pytest.fixture(scope="session")
def processed(small_synthetic: UrbanLayers, tmp_path_factory: pytest.TempPathFactory) -> DictConfig:
    """Mini dossier de données : une tuile de test synthétique, ses paires et références.

    Arborescence : ``<racine>/processed/{tiles,context,pairs.parquet,baselines.parquet}`` et
    ``<racine>/experiments`` (vide), via ``paths.data_dir`` et ``paths.experiments_dir``.
    """
    import pandas as pd

    from happycity.data.access_context import build_context, metric_params, with_channel
    from happycity.data.pairs import tile_pairs
    from happycity.data.raster import TileSpec, rasterize_tile
    from happycity.metrics import served_origins, unserved_raster

    root = tmp_path_factory.mktemp("data")
    cfg = load_config([f"paths.data_dir={root.as_posix()}",
                       f"paths.experiments_dir={(root / 'experiments').as_posix()}",
                       "eval.n_proposals=2", "eval.random_baseline_draws=3"])  # fmt: skip
    out = root / "processed"
    xmin, ymin, _, _ = small_synthetic.zone.bounds
    spec = TileSpec(small_synthetic.name, "synthetic", small_synthetic.name, "test_synth",
                    xmin, ymin, 1024.0)  # fmt: skip
    tile = rasterize_tile(small_synthetic, spec, cfg.data)
    ctx = build_context(small_synthetic, tile, cfg.data)
    params = metric_params(cfg.data)
    tile = with_channel(tile, "unserved_origins", unserved_raster(ctx, served_origins(ctx, params)))
    name = f"{spec.tile_id}.npz"
    tile.save(out / "tiles" / name)
    ctx.save(out / "context" / name)
    pairs, _, baselines = tile_pairs(tile, ctx, "test_synth", cfg.data, params)
    table = pd.DataFrame(pairs).assign(pair_file=name, mask_index=range(len(pairs)))
    table["pair_id"] = [f"p{i}" for i in range(len(table))]
    table.to_parquet(out / "pairs.parquet", index=False)
    pd.DataFrame(baselines).to_parquet(out / "baselines.parquet", index=False)
    return cfg
