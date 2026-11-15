"""Construction du contexte d'évaluation d'une tuile (pont données → module de métriques).

Le module ``metrics`` est autonome : il ne lit ni couches OSM ni configuration. Ce module
lui prépare, pour chaque tuile, un :class:`~happycity.metrics.TileAccessContext` :

* sous-graphe piéton des nœuds à ≤ ``access_radius_m + max_snap_m`` de la tuile — tout
  chemin de ≤ 400 m partant d'une origine de la tuile y est contenu (une origine est à
  ≤ ``max_snap_m`` de son nœud), donc aucune distance n'est faussée ;
* points d'accès des espaces verts existants (contexte compris) : entrées OSM explicites
  si l'espace vert en a, sinon points des tronçons à ≤ ``entrance_snap_m`` du contour ;
* origines de la tuile et carte des espaces verts comptés.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from happycity.data.layers import UrbanLayers
from happycity.data.network import graph_nodes
from happycity.data.raster import TileRaster
from happycity.metrics import (
    AccessPoints,
    MetricParams,
    TileAccessContext,
    WalkGraph,
    access_points_for,
)


def metric_params(data_cfg: DictConfig) -> MetricParams:
    """Paramètres des métriques issus de la configuration."""
    return MetricParams(
        radius_m=float(data_cfg.access_radius_m),
        reach_m=float(data_cfg.green.entrance_snap_m),
        sectors_per_side=int(data_cfg.equity.sectors_per_side),
        min_origins=int(data_cfg.equity.min_origins),
    )


def build_context(layers: UrbanLayers, tile: TileRaster, data_cfg: DictConfig) -> TileAccessContext:
    """Contexte d'évaluation d'une tuile rastérisée."""
    spec = tile.spec
    params = metric_params(data_cfg)
    margin = params.radius_m + float(data_cfg.origins.max_snap_m)
    region = spec.geometry.buffer(margin, join_style="mitre")
    nodes = graph_nodes(layers.graph)
    inside = nodes.iloc[nodes.sindex.query(region, predicate="intersects")]
    walk = WalkGraph.from_networkx(layers.graph, inside["node"].tolist())

    green = layers.green.iloc[layers.green.sindex.query(region, predicate="intersects")]
    osm_gates = layers.entrances[layers.entrances["method"] == "osm"]
    gated = set(osm_gates["green_id"]) & set(green["green_id"])
    gate_nodes = walk.node_index(osm_gates[osm_gates["green_id"].isin(gated)]["node"])
    points = AccessPoints.at_nodes(gate_nodes[gate_nodes >= 0])
    others = green[~green["green_id"].isin(gated)]
    points = points.concat(access_points_for(list(others.geometry), walk, params.reach_m))

    origin_nodes = walk.node_index(tile.origins["node"])
    return TileAccessContext(
        tile_id=spec.tile_id,
        xmin=spec.xmin,
        ymin=spec.ymin,
        size_px=int(data_cfg.tile.size_px),
        m_per_px=float(data_cfg.tile.m_per_px),
        walk=walk,
        origin_nodes=origin_nodes,
        origin_rows=tile.origins["row"].to_numpy(dtype=np.int64),
        origin_cols=tile.origins["col"].to_numpy(dtype=np.int64),
        existing=points,
        green_fraction=tile.channel("public_green"),
    )


def with_channel(tile: TileRaster, name: str, values: np.ndarray) -> TileRaster:
    """Copie de la tuile avec un canal ajouté (ou remplacé)."""
    names = [n for n in tile.channel_names if n != name]
    channels = np.stack([tile.channel(n) for n in names] + [values.astype(np.float32)])
    return TileRaster(
        tile.spec, (*names, name), channels, tile.candidate_ids, tile.candidates, tile.origins
    )
