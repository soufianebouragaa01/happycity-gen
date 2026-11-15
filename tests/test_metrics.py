"""Métriques V, A400, E sur des mini-cartes calculables à la main."""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import box

from happycity.metrics import (
    AccessPoints,
    BudgetExceededError,
    MetricParams,
    TileAccessContext,
    WalkGraph,
    access_points_for,
    accessibility,
    added_green_m2,
    check_budget,
    equity,
    evaluate_tile,
    green_cover,
    network_distances,
    sector_of,
    served_origins,
    unserved_raster,
)


def line_graph(n: int = 11, step: float = 100.0, y: float = 0.0) -> nx.Graph:
    """Rue rectiligne : nœuds 0…n−1 tous les ``step`` mètres le long de x."""
    g = nx.Graph()
    for i in range(n):
        g.add_node(i, x=i * step, y=y)
    for i in range(n - 1):
        g.add_edge(i, i + 1, length=step)
    return g


@pytest.fixture
def street() -> WalkGraph:
    return WalkGraph.from_networkx(line_graph())


# --------------------------------------------------------------------------- A400


@pytest.mark.parametrize(
    ("park_node", "served"),
    [(3, True), (4, True), (5, False)],  # 300 m → accès ; 400 m pile → accès ; 500 m → non
)
def test_park_distance_threshold(street: WalkGraph, park_node: int, served: bool) -> None:
    res = accessibility(street, np.array([0]), AccessPoints.at_nodes([park_node]))
    assert res.served.tolist() == [served]
    assert res.a400 == (1.0 if served else 0.0)


def test_access_point_on_edge(street: WalkGraph) -> None:
    def point(k: int, t: float) -> AccessPoints:  # sur le tronçon k → k+1, à t m de k
        return AccessPoints(np.array([k]), np.array([k + 1]), np.array([t]), np.array([100.0]))

    assert accessibility(street, np.array([0]), point(3, 50)).distance_m[0] == pytest.approx(350)
    assert not accessibility(street, np.array([0]), point(4, 50)).served[0]  # 450 m


def test_nearest_of_several_parks(street: WalkGraph) -> None:
    points = AccessPoints.at_nodes([9, 2])
    dist = network_distances(street, points, 400)
    assert dist[0] == pytest.approx(200)
    assert dist[5] == pytest.approx(300)  # 5 → 2 (300 m) plutôt que 5 → 9 (400 m)
    assert dist[6] == pytest.approx(300)


def test_network_not_euclidean() -> None:
    """Chemin en U : parc à 50 m à vol d'oiseau mais 650 m par le réseau → pas d'accès."""
    g = nx.Graph()
    for n, (x, y) in enumerate([(0, 0), (0, 300), (50, 300), (50, 0)]):
        g.add_node(n, x=x, y=y)
    g.add_edge(0, 1, length=300)
    g.add_edge(1, 2, length=50)
    g.add_edge(2, 3, length=300)
    walk = WalkGraph.from_networkx(g)
    res = accessibility(walk, np.array([0]), AccessPoints.at_nodes([3]))
    assert res.distance_m[0] == math.inf
    assert res.a400 == 0


def test_disconnected_origin_is_not_served() -> None:
    g = line_graph(3)
    g.add_node(99, x=0, y=500)
    walk = WalkGraph.from_networkx(g)
    res = accessibility(walk, walk.node_index([0, 99]), AccessPoints.at_nodes(walk.node_index([1])))
    assert res.served.tolist() == [True, False]


def test_parallel_edges_keep_shortest() -> None:
    g = nx.MultiGraph()
    g.add_node(0, x=0, y=0)
    g.add_node(1, x=100, y=0)
    g.add_edge(0, 1, length=500)
    g.add_edge(0, 1, length=100)
    walk = WalkGraph.from_networkx(g)
    assert network_distances(walk, AccessPoints.at_nodes([1]), 400)[0] == pytest.approx(100)


def test_a400_is_a_proportion_and_nan_without_origins(street: WalkGraph) -> None:
    res = accessibility(street, np.array([0, 1, 2, 9]), AccessPoints.at_nodes([1]))
    assert res.a400 == pytest.approx(0.75)
    assert math.isnan(
        accessibility(street, np.array([], dtype=int), AccessPoints.at_nodes([1])).a400
    )


def test_no_green_means_no_access(street: WalkGraph) -> None:
    assert accessibility(street, np.array([0, 5]), AccessPoints.empty()).a400 == 0


def test_access_points_from_polygon(street: WalkGraph) -> None:
    near = box(340, 10, 360, 30)  # à 10 m de la rue → point d'accès en x = 340..360
    far = box(340, 20, 360, 40)  # à 20 m > 15 m → aucun accès
    pts = access_points_for([near], street, reach_m=15)
    assert len(pts) == 1
    assert pts.u[0] == 3
    assert 40 <= pts.t[0] <= 60
    assert len(access_points_for([far], street, reach_m=15)) == 0


# --------------------------------------------------------------------------- V et budget


def test_green_cover_and_budget() -> None:
    existing = np.zeros((4, 4))
    existing[0, 0] = 0.5
    added = np.zeros((4, 4), dtype=bool)
    added[0, 0] = added[1, 1] = True
    before = green_cover(existing, 8.0)
    after = green_cover(existing, 8.0, added)
    assert pytest.approx(0.5 / 16) == before.V
    assert after.green_m2 == pytest.approx(2 * 64)  # pixel déjà à moitié vert non compté deux fois
    delta = added_green_m2(before, after)
    assert delta == pytest.approx(1.5 * 64)
    check_budget(delta, 96.0)  # ΔA = b : accepté
    with pytest.raises(BudgetExceededError):
        check_budget(delta, 95.0)
    with pytest.raises(ValueError, match="négatif"):
        check_budget(0.0, -1.0)
    with pytest.raises(ValueError, match="masque"):
        green_cover(existing, 8.0, np.zeros((3, 3), dtype=bool))


# --------------------------------------------------------------------------- E


def test_sector_of_quadrants() -> None:
    rows = np.array([0, 0, 127, 127, 63, 64])
    cols = np.array([0, 127, 0, 127, 64, 63])
    assert sector_of(rows, cols, 128, 2).tolist() == [0, 1, 2, 3, 1, 2]


def test_equity_two_sectors() -> None:
    served = np.array([True, True, True, False])
    res = equity(served, np.array([0, 0, 1, 1]), 4, min_origins=2)
    assert pytest.approx(1 - (1.0 - 0.5)) == res.E
    assert res.n_valid == 2
    assert math.isnan(res.sector_a400[2])  # secteurs vides ignorés


def test_equity_ignores_small_sectors_and_needs_two() -> None:
    served = np.array([True] * 10 + [False])
    sectors = np.array([0] * 10 + [1])
    assert math.isnan(equity(served, sectors, 4, min_origins=10).E)  # 1 seul secteur valide
    assert pytest.approx(0.0) == equity(served, sectors, 4, min_origins=1).E


def test_equity_at_zero_access_is_one() -> None:
    """Égalité parfaite à zéro accès : E = 1 — d'où l'obligation de rapporter A400."""
    res = equity(np.zeros(20, dtype=bool), np.array([0] * 10 + [1] * 10), 4, min_origins=10)
    assert res.E == 1.0


# --------------------------------------------------------------------------- tuile


def _tile_context() -> TileAccessContext:
    """Tuile de 128 px ; rue le long de la ligne 0 (y = ymax − 4), nœuds tous les 100 m."""
    xmin, ymin, size, res = 0.0, 0.0, 128, 8.0
    ymax = ymin + size * res
    walk = WalkGraph.from_networkx(line_graph(n=11, y=ymax - 4))
    return TileAccessContext(
        tile_id="t",
        xmin=xmin,
        ymin=ymin,
        size_px=size,
        m_per_px=res,
        walk=walk,
        origin_nodes=walk.node_index([0, 9]),
        origin_rows=np.array([0, 0]),
        origin_cols=np.array([0, 112]),  # x = 0 → colonne 0 ; x = 900 → colonne 112
        existing=AccessPoints.empty(),
        green_fraction=np.zeros((size, size), dtype=np.float32),
    )


def test_evaluate_tile_before_after() -> None:
    ctx = _tile_context()
    params = MetricParams(min_origins=1)
    before = evaluate_tile(ctx, params=params)
    assert (before.A400, before.V, before.delta_green_m2) == (0.0, 0.0, 0.0)
    assert before.E == 1.0  # égalité à zéro accès

    added = np.zeros((128, 128), dtype=bool)
    added[1:3, 112] = True  # 2 px sous la rue, au droit du nœud 9
    after = evaluate_tile(ctx, added, params=params, budget_m2=128)
    assert after.delta_green_m2 == pytest.approx(128)
    assert pytest.approx(128 / 1024**2) == after.V
    assert pytest.approx(0.5) == after.A400  # nœud 9 desservi, nœud 0 à 900 m
    assert pytest.approx(0.0) == after.E  # secteur ouest 0 %, secteur est 100 %
    with pytest.raises(BudgetExceededError):
        evaluate_tile(ctx, added, params=params, budget_m2=100)


def test_added_green_far_from_network_gives_no_access() -> None:
    ctx = _tile_context()
    added = np.zeros((128, 128), dtype=bool)
    added[60:62, 112] = True  # à ~480 m de la rue
    assert evaluate_tile(ctx, added).A400 == 0


def test_unserved_raster_and_context_roundtrip(tmp_path: Path) -> None:
    ctx = _tile_context()
    ctx.existing = AccessPoints.at_nodes(ctx.walk.node_index([1]))  # parc au nœud 1
    served = served_origins(ctx)
    assert served.tolist() == [True, False]
    raster = unserved_raster(ctx, served)
    assert raster.sum() == 1
    assert raster[0, 112] == 1
    ctx.save(tmp_path / "ctx.npz")
    back = TileAccessContext.load(tmp_path / "ctx.npz")
    np.testing.assert_equal(evaluate_tile(back).as_dict(), evaluate_tile(ctx).as_dict())
