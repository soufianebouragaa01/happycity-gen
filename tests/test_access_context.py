"""Pont données → métriques et recoupement avec une implémentation indépendante (NetworkX)."""

from __future__ import annotations

import networkx as nx
import numpy as np
from omegaconf import DictConfig

from happycity.data.access_context import build_context, metric_params, with_channel
from happycity.data.layers import UrbanLayers
from happycity.data.raster import TileSpec, rasterize_tile
from happycity.metrics import (
    AccessPoints,
    accessibility,
    evaluate_tile,
    served_origins,
    unserved_raster,
)


def _tile(layers: UrbanLayers, cfg: DictConfig):  # type: ignore[no-untyped-def]
    xmin, ymin, _, _ = layers.zone.bounds
    spec = TileSpec(layers.name, "synthetic", layers.name, "train", xmin, ymin, 1024.0)
    return rasterize_tile(layers, spec, cfg.data)


def test_a400_matches_networkx_on_full_graph(small_synthetic: UrbanLayers, cfg: DictConfig) -> None:
    """Même A400 que NetworkX sur le graphe COMPLET : le sous-graphe ne perd aucun chemin."""
    tile = _tile(small_synthetic, cfg)
    ctx = build_context(small_synthetic, tile, cfg.data)
    # Sources arbitraires (1 nœud sur 37 du graphe COMPLET, y compris hors du sous-graphe).
    gates = sorted(small_synthetic.graph.nodes)[::37]
    idx = ctx.walk.node_index(gates)
    ctx.existing = AccessPoints.at_nodes(idx[idx >= 0])
    ours = accessibility(ctx.walk, ctx.origin_nodes, ctx.existing, 400.0)

    graph = nx.Graph()
    for u, v, data in small_synthetic.graph.edges(data=True):
        w = data["length"]
        if not graph.has_edge(u, v) or graph[u][v]["length"] > w:
            graph.add_edge(u, v, length=w)
    reached = nx.multi_source_dijkstra_path_length(graph, set(gates), cutoff=400, weight="length")
    expected = np.array([n in reached for n in tile.origins["node"]])
    np.testing.assert_array_equal(ours.served, expected)
    assert 0 < ours.a400 < 1


def test_context_and_unserved_channel(small_synthetic: UrbanLayers, cfg: DictConfig) -> None:
    tile = _tile(small_synthetic, cfg)
    ctx = build_context(small_synthetic, tile, cfg.data)
    params = metric_params(cfg.data)
    assert params.radius_m == 400
    assert len(ctx.origin_nodes) == len(tile.origins)
    assert (ctx.origin_nodes >= 0).all()  # toutes les origines sont dans le sous-graphe
    served = served_origins(ctx, params)
    channel = unserved_raster(ctx, served)
    assert channel.sum() == (~served).sum()
    enriched = with_channel(tile, "unserved_origins", channel)
    assert enriched.channel_names[-1] == "unserved_origins"
    assert (
        with_channel(enriched, "unserved_origins", channel).channels.shape
        == enriched.channels.shape
    )
    scores = evaluate_tile(ctx, params=params)
    assert served.mean() == scores.A400
    assert tile.channel("public_green").sum() / tile.channel("public_green").size == scores.V


def test_greening_a_candidate_never_reduces_access(
    small_synthetic: UrbanLayers, cfg: DictConfig
) -> None:
    tile = _tile(small_synthetic, cfg)
    ctx = build_context(small_synthetic, tile, cfg.data)
    params = metric_params(cfg.data)
    before = evaluate_tile(ctx, params=params)
    for local_id in tile.candidates["local_id"]:
        after = evaluate_tile(ctx, tile.candidate_ids == local_id, params=params)
        assert after.A400 >= before.A400
        assert after.delta_green_m2 > 0
