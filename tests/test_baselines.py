"""Références non génératives et paires, sur une scène calculable à la main."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from happycity.baselines import PlacementProblem, greedy_placement, random_placement
from happycity.data.access_context import build_context, metric_params
from happycity.data.layers import UrbanLayers
from happycity.data.pairs import (
    budget_pixels,
    mean_pairwise_iou_distance,
    stable_seed,
    synthetic_split,
    tile_pairs,
)
from happycity.data.raster import TileSpec, rasterize_tile
from happycity.metrics import (
    AccessPoints,
    MetricParams,
    TileAccessContext,
    WalkGraph,
    evaluate_tile,
)

SIZE, RES = 128, 8.0
YMAX = SIZE * RES
PARAMS = MetricParams(min_origins=1)


def _street() -> WalkGraph:
    g = nx.Graph()
    for i in range(11):  # nœuds tous les 100 m, rue le long de la ligne 0
        g.add_node(i, x=i * 100.0, y=YMAX - 4)
    for i in range(10):
        g.add_edge(i, i + 1, length=100.0)
    return WalkGraph.from_networkx(g)


def scene(existing_nodes: list[int] | None = None) -> PlacementProblem:
    """5 origines à x = 0, 2 à x = 1000 ; parcelles A (x 0–40), B (x 960–1000), C (milieu, inutile),
    D (non adjacente)."""
    walk = _street()
    origins = walk.node_index([0] * 5 + [10] * 2)
    ctx = TileAccessContext(
        tile_id="scene",
        xmin=0.0,
        ymin=0.0,
        size_px=SIZE,
        m_per_px=RES,
        walk=walk,
        origin_nodes=origins,
        origin_rows=np.zeros(7, dtype=np.int64),
        origin_cols=np.array([0] * 5 + [124] * 2),
        existing=AccessPoints.at_nodes(walk.node_index(existing_nodes or [])),
        green_fraction=np.zeros((SIZE, SIZE), dtype=np.float32),
    )
    ids = np.full((SIZE, SIZE), -1, dtype=np.int32)
    ids[1:5, 0:5] = 0  # A : 20 px
    ids[1:5, 120:125] = 1  # B : 20 px
    ids[1:5, 60:65] = 2  # C : 20 px, à 480 m des origines
    ids[1:5, 10:15] = 3  # D : non adjacente (déclarée comme telle)
    walk_raster = np.zeros((SIZE, SIZE), dtype=np.float32)
    walk_raster[0] = 1
    return PlacementProblem(ids, [0, 1, 2], walk_raster, ctx, PARAMS)


def test_problem_basics() -> None:
    problem = scene()
    assert problem.parcels == [0, 1, 2]  # D exclue : non adjacente
    assert problem.capacity_px == 60
    piece = problem.piece(0, 10)
    assert piece.sum() == 10
    assert piece[1, 0]  # ancré sur le pixel le plus proche de la rue
    from scipy.ndimage import label

    assert label(piece)[1] == 1  # un seul bloc connexe
    assert problem.served_by(0, 20).sum() == 5
    assert problem.served_by(2, 20).sum() == 0


@pytest.mark.parametrize(
    ("budget", "parcels", "a400"), [(20, [0], 5 / 7), (40, [0, 1], 1.0), (60, [0, 1], 1.0)]
)
def test_greedy_is_optimal_here(budget: int, parcels: list[int], a400: float) -> None:
    result = greedy_placement(scene(), budget)
    assert result.parcels == parcels  # jamais C (aucun gain) ni D (non adjacente)
    assert result.a400 == pytest.approx(a400)
    assert result.n_px <= budget


def test_greedy_takes_a_piece_when_budget_is_small() -> None:
    result = greedy_placement(scene(), 10)
    assert result.n_px == 10
    assert result.a400 == pytest.approx(5 / 7)


def test_internal_estimate_matches_independent_metric() -> None:
    problem = scene()
    for result in (
        greedy_placement(problem, 40),
        random_placement(problem, 40, np.random.default_rng(0)),
    ):
        scores = evaluate_tile(problem.ctx, result.mask, params=PARAMS, budget_m2=40 * RES**2)
        assert pytest.approx(result.a400) == scores.A400


def test_random_is_valid_and_spends_budget() -> None:
    problem = scene()
    for seed in range(20):
        result = random_placement(problem, 30, np.random.default_rng(seed))
        assert result.n_px == 30
        assert set(result.parcels) <= {0, 1, 2}
        assert not result.mask[1:5, 10:15].any()  # jamais sur D


def test_greedy_beats_random_on_average() -> None:
    problem = scene()
    greedy = greedy_placement(problem, 20).a400
    randoms = [random_placement(problem, 20, np.random.default_rng(s)).a400 for s in range(30)]
    assert greedy >= max(randoms)
    assert greedy > np.mean(randoms)


def test_stochastic_greedy_gives_several_good_solutions() -> None:
    problem = scene()
    solutions = {
        tuple(greedy_placement(problem, 20, np.random.default_rng(s), top_k=3).parcels)
        for s in range(20)
    }
    assert solutions == {(0,), (1,)}  # deux solutions à gain positif ; jamais C


def test_no_gain_means_no_change() -> None:
    problem = scene(existing_nodes=[0, 10])  # tout est déjà desservi
    assert greedy_placement(problem, 40).n_px == 0


# --------------------------------------------------------------------------- paires


def test_helpers() -> None:
    assert stable_seed("a", 1) == stable_seed("a", 1) != stable_seed("a", 2)
    names = [f"synth_{i:05d}" for i in range(2000)]
    share = np.mean([synthetic_split(n, 0.1) == "val_synth" for n in names])
    assert share == pytest.approx(0.1, abs=0.02)
    assert synthetic_split("synth_00007", 0.1) == synthetic_split("synth_00007", 0.1)
    three = [synthetic_split(n, 0.1, 0.1) for n in names]
    assert np.mean([s == "test_synth" for s in three]) == pytest.approx(0.1, abs=0.02)
    # val_synth inchangé quand on ajoute test_synth (les cartes de train ne migrent pas vers val)
    assert all(
        t == "val_synth"
        for n, t in zip(names, three, strict=True)
        if synthetic_split(n, 0.1) == "val_synth"
    )
    a = np.zeros((4, 4), dtype=bool)
    b = a.copy()
    a[0, 0] = b[0, 0] = b[0, 1] = True
    assert mean_pairwise_iou_distance([a, b]) == pytest.approx(0.5)
    assert np.isnan(mean_pairwise_iou_distance([a]))


def test_tile_pairs(small_synthetic: UrbanLayers, cfg: DictConfig) -> None:
    xmin, ymin, _, _ = small_synthetic.zone.bounds
    spec = TileSpec(
        small_synthetic.name, "synthetic", small_synthetic.name, "train", xmin, ymin, 1024.0
    )
    tile = rasterize_tile(small_synthetic, spec, cfg.data)
    ctx = build_context(small_synthetic, tile, cfg.data)
    params = metric_params(cfg.data)
    pairs, masks, baselines = tile_pairs(tile, ctx, "train", cfg.data, params)
    table = pd.DataFrame(pairs)
    assert len(masks) == len(table)
    assert len(baselines) == len(cfg.data.pairs.budget_fractions)
    kept = table[table["excluded"] == ""]
    assert (kept["n_px"] <= kept["budget_px"]).all()
    assert (kept["a_target"] >= kept["A400_before"]).all()
    assert np.allclose(kept["a_target"], kept["internal_a400"])
    allowed = tile.candidate_mask
    for k in kept.index:
        assert not (masks[k] & ~allowed).any()  # uniquement sur des parcelles
    for _, group in kept.groupby("b_frac"):  # solutions distinctes pour un même budget
        keys = {np.packbits(masks[k]).tobytes() for k in group.index}
        assert len(keys) == len(group)
    assert budget_pixels(0.01, cfg.data) == 163
