"""Rattachement des origines et des entrées au graphe piéton."""

from __future__ import annotations

from collections.abc import Callable

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Point, box

from conftest import CRS, X0, Y0
from happycity.data.network import attach_origins, graph_nodes, infer_entrances


def test_graph_nodes(grid_graph: Callable[..., nx.MultiDiGraph]) -> None:
    nodes = graph_nodes(grid_graph(n=3))
    assert len(nodes) == 9
    assert nodes.loc[4].geometry.equals(Point(X0 + 100, Y0 + 100))


def test_attach_origins_snaps_and_drops(grid_graph: Callable[..., nx.MultiDiGraph]) -> None:
    graph = grid_graph(n=3)  # nœuds de X0 à X0+200
    buildings = gpd.GeoDataFrame(
        {"residential": [True, True, False], "rule": ["x"] * 3, "osm_id": [None] * 3},
        geometry=[
            box(X0 + 5, Y0 + 5, X0 + 15, Y0 + 15),  # centroïde à ~14 m du nœud 0
            box(X0 + 990, Y0, X0 + 1010, Y0 + 20),  # à ~800 m du réseau → écartée
            box(X0 + 95, Y0 + 95, X0 + 105, Y0 + 105),  # non résidentiel
        ],
        crs=CRS,
    )
    origins, dropped = attach_origins(buildings, graph, max_snap_m=100)
    assert dropped == 1
    assert len(origins) == 1
    assert origins.iloc[0]["node"] == 0
    assert origins.iloc[0]["building_id"] == 0
    assert origins.iloc[0]["snap_m"] == pytest.approx(10 * 2**0.5)


def _park(xmin: float, ymin: float, side: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"green_id": [0], "kind": ["leisure=park"], "name": [None], "osm_id": [None]},
        geometry=[box(xmin, ymin, xmin + side, ymin + side)],
        crs=CRS,
    )


def test_entrances_from_boundary(grid_graph: Callable[..., nx.MultiDiGraph]) -> None:
    graph = grid_graph(n=3)
    park = _park(X0 + 10, Y0 + 10, 80)  # coins du parc à ~14 m des 4 nœuds voisins
    entrances = infer_entrances(park, graph, snap_m=15)
    assert set(entrances["node"]) == {0, 1, 3, 4}
    assert set(entrances["method"]) == {"boundary"}


def test_entrances_prefer_osm_gates(grid_graph: Callable[..., nx.MultiDiGraph]) -> None:
    graph = grid_graph(n=3)
    park = _park(X0 + 10, Y0 + 10, 80)
    gate = gpd.GeoSeries([Point(X0 + 10, Y0 + 50)], crs=CRS)  # sur le contour ouest
    entrances = infer_entrances(park, graph, snap_m=15, osm_entrances=gate)
    assert list(entrances["method"]) == ["osm"]
    assert entrances.iloc[0]["node"] in {0, 1}


def test_isolated_park_has_no_entrance(grid_graph: Callable[..., nx.MultiDiGraph]) -> None:
    graph = grid_graph(n=3)
    park = _park(X0 + 600, Y0 + 600, 50)
    assert infer_entrances(park, graph, snap_m=15).empty
