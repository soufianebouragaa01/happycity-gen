"""Fixtures partagées : configuration, mini-graphes et objets OSM factices (sans réseau)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import geopandas as gpd
import networkx as nx
import pytest
from omegaconf import DictConfig
from shapely.geometry import LineString

from happycity.config import load_config
from happycity.data.layers import UrbanLayers
from happycity.data.osm import RELEVANT_KEYS
from happycity.data.synthetic import generate_map

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
