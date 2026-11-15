"""Rattachement au graphe piéton : origines résidentielles et entrées des espaces verts.

Ces fonctions sont partagées par l'extraction OSM et le générateur synthétique, pour que
les deux sources produisent exactement le même format.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


def graph_nodes(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Nœuds du graphe sous forme de points (index = identifiant du nœud)."""
    ids = list(graph.nodes)
    xs = [graph.nodes[n]["x"] for n in ids]
    ys = [graph.nodes[n]["y"] for n in ids]
    return gpd.GeoDataFrame(
        {"node": ids}, geometry=gpd.points_from_xy(xs, ys), crs=graph.graph["crs"], index=ids
    )


def graph_edges(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Tronçons du graphe, un seul sens conservé par paire de nœuds (réseau non orienté)."""
    edges = ox.graph_to_gdfs(graph, nodes=False).reset_index()
    low = edges[["u", "v"]].min(axis=1).astype(str)
    high = edges[["u", "v"]].max(axis=1).astype(str)
    pair = low + "-" + high + "-" + edges["key"].astype(str)
    return edges.assign(pair=pair).drop_duplicates("pair").reset_index(drop=True)


def attach_origins(
    buildings: gpd.GeoDataFrame, graph: nx.MultiDiGraph, max_snap_m: float
) -> tuple[gpd.GeoDataFrame, int]:
    """Origines = centroïdes des bâtiments résidentiels rattachés au nœud piéton le plus proche.

    Args:
        buildings: couche ``buildings`` (colonne booléenne ``residential``).
        graph: graphe piéton projeté.
        max_snap_m: au-delà, l'origine est écartée (bâtiment hors réseau).

    Returns:
        ``(origines, nombre d'origines écartées)``.
    """
    residential = buildings[buildings["residential"].astype(bool)]
    columns: dict[str, list[Any]] = {"building_id": [], "node": [], "snap_m": []}
    if residential.empty or graph.number_of_nodes() == 0:
        empty = gpd.GeoDataFrame(columns, geometry=[], crs=buildings.crs)
        return empty, len(residential)
    centroids = residential.geometry.centroid
    nodes, dists = ox.distance.nearest_nodes(
        graph, centroids.x.to_numpy(), centroids.y.to_numpy(), return_dist=True
    )
    origins = gpd.GeoDataFrame(
        {
            "building_id": residential.index.to_numpy(),
            "node": np.asarray(nodes, dtype=np.int64),
            "snap_m": np.asarray(dists, dtype=float),
        },
        geometry=centroids.to_numpy(),
        crs=buildings.crs,
    )
    kept = origins[origins["snap_m"] <= max_snap_m].reset_index(drop=True)
    return kept, len(origins) - len(kept)


def infer_entrances(
    green: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    snap_m: float,
    osm_entrances: gpd.GeoSeries | None = None,
) -> gpd.GeoDataFrame:
    """Entrées des espaces verts, exprimées en nœuds du graphe piéton.

    Pour chaque espace vert : si des entrées OSM (``entrance=*``, ``barrier=gate``) se
    trouvent à ≤ ``snap_m`` de son contour, on retient le nœud le plus proche de chacune
    (méthode ``osm``) ; sinon, tous les nœuds à ≤ ``snap_m`` du polygone (méthode
    ``boundary``). Un espace vert sans aucun nœud proche n'a pas d'entrée : il est signalé
    par le rapport de couverture et ne compte pas pour l'accessibilité.
    """
    columns: dict[str, list[Any]] = {"green_id": [], "node": [], "method": []}
    if green.empty or graph.number_of_nodes() == 0:
        return gpd.GeoDataFrame(columns, geometry=[], crs=green.crs)
    nodes = graph_nodes(graph)
    zones = gpd.GeoDataFrame(
        {"green_id": green["green_id"].to_numpy()},
        geometry=green.geometry.buffer(snap_m).to_numpy(),
        crs=green.crs,
    )
    frames: list[pd.DataFrame] = []
    with_osm: set[int] = set()
    if osm_entrances is not None and not osm_entrances.empty:
        points = gpd.GeoDataFrame(geometry=osm_entrances.to_numpy(), crs=green.crs)
        hits = gpd.sjoin(points, zones, predicate="within")
        if not hits.empty:
            snapped = ox.distance.nearest_nodes(
                graph, hits.geometry.x.to_numpy(), hits.geometry.y.to_numpy()
            )
            osm_frame = pd.DataFrame(
                {
                    "green_id": hits["green_id"].to_numpy(),
                    "node": np.asarray(snapped, dtype=np.int64),
                }
            )
            osm_frame["method"] = "osm"
            frames.append(osm_frame)
            with_osm = set(osm_frame["green_id"].tolist())
    near = gpd.sjoin(nodes[["node", "geometry"]], zones, predicate="within")
    near = near[~near["green_id"].isin(with_osm)]
    boundary_frame = pd.DataFrame(
        {"green_id": near["green_id"].to_numpy(), "node": near["node"].to_numpy(dtype=np.int64)}
    )
    boundary_frame["method"] = "boundary"
    frames.append(boundary_frame)
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["green_id", "node"])
    geometry = nodes.loc[result["node"].to_numpy(), "geometry"].to_numpy()
    return gpd.GeoDataFrame(result.reset_index(drop=True), geometry=geometry, crs=green.crs)
