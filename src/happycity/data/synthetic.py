"""Générateur procédural de cartes synthétiques (``source=synthetic``).

Chaque carte est produite dans le MÊME format que l'extraction OSM (:class:`UrbanLayers`),
avec les mêmes fonctions de rattachement au graphe (``network.py``), si bien que la
rastérisation et les métriques ne distinguent pas les deux sources.

Algorithme (paramètres par typologie dans ``data.synthetic.typologies``) :

1. trame viaire : grille tournée, pas tirés dans ``block_m``, nœuds bruités
   (``jitter_m``), tronçons supprimés avec la probabilité ``edge_drop`` ;
2. îlots : polygonisation de la trame, retrait de ``street_half_width_m`` ;
3. occupation de chaque îlot : parc, terrain vacant, parking de surface, équipement
   (bâtiment non résidentiel), obstacle (cimetière), îlot vide (non bâti, non
   candidat : terrains non cartographiés ou privés) ou résidentiel ;
4. îlots résidentiels découpés en lots (côté ``lot_m``), un bâtiment par lot dont
   l'emprise suit ``building_coverage`` ;
5. origines et entrées calculées comme pour les données réelles.
"""

from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from omegaconf import DictConfig
from shapely.geometry import LineString, Polygon, box

from happycity.data.layers import UrbanLayers, empty_layer
from happycity.data.network import attach_origins, infer_entrances

#: Origine fictive des cartes synthétiques (coordonnées UTM valides, hors Marrakech).
SYNTH_ORIGIN = (300_000.0, 3_300_000.0)
LAND_USES = ("park", "vacant", "parking", "other", "obstacle", "empty")


def map_seed(base_seed: int, index: int) -> int:
    """Graine d'une carte : déterministe et indépendante des autres cartes."""
    return int(np.random.SeedSequence([base_seed, index]).generate_state(1)[0])


def choose_typology(rng: np.random.Generator, typologies: DictConfig) -> str:
    """Tire une typologie selon les poids configurés."""
    names = list(typologies)
    weights = np.array([float(typologies[n].weight) for n in names])
    return str(names[rng.choice(len(names), p=weights / weights.sum())])


def street_graph(
    rng: np.random.Generator, typ: DictConfig, bounds: Polygon, crs: str
) -> tuple[nx.MultiDiGraph, dict[str, float]]:
    """Trame viaire tournée et bruitée, réduite à sa plus grande composante connexe."""
    xmin, ymin, xmax, ymax = bounds.bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    lo, hi = float(typ.block_m[0]), float(typ.block_m[1])
    sx, sy = float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))
    angle = float(rng.uniform(-45, 45))
    theta = math.radians(angle)
    radius = 0.75 * (xmax - xmin)
    ni, nj = math.ceil(radius / sx), math.ceil(radius / sy)
    grow = bounds.buffer(max(sx, sy), join_style="mitre")

    positions: dict[tuple[int, int], tuple[float, float]] = {}
    for i in range(-ni, ni + 1):
        for j in range(-nj, nj + 1):
            u = i * sx + rng.normal(0, typ.jitter_m)
            v = j * sy + rng.normal(0, typ.jitter_m)
            x = cx + u * math.cos(theta) - v * math.sin(theta)
            y = cy + u * math.sin(theta) + v * math.cos(theta)
            if grow.contains(shapely.Point(x, y)):
                positions[(i, j)] = (x, y)

    undirected = nx.Graph()
    ids = {key: n for n, key in enumerate(sorted(positions))}
    for key, (x, y) in positions.items():
        undirected.add_node(ids[key], x=x, y=y)
    for i, j in positions:
        for di, dj in ((1, 0), (0, 1)):
            nb = (i + di, j + dj)
            if nb in positions and rng.random() >= typ.edge_drop:
                undirected.add_edge(ids[(i, j)], ids[nb])
    largest = max(nx.connected_components(undirected), key=len)
    undirected = undirected.subgraph(largest).copy()

    graph = nx.MultiDiGraph(crs=crs)
    for n, data in undirected.nodes(data=True):
        graph.add_node(n, x=data["x"], y=data["y"], street_count=undirected.degree(n))
    for a, b in undirected.edges:
        pa, pb = undirected.nodes[a], undirected.nodes[b]
        length = math.dist((pa["x"], pa["y"]), (pb["x"], pb["y"]))
        line = LineString([(pa["x"], pa["y"]), (pb["x"], pb["y"])])
        for u, v, geom in ((a, b, line), (b, a, line.reverse())):
            graph.add_edge(u, v, key=0, length=length, highway="residential", geometry=geom)
    return graph, {"block_x_m": sx, "block_y_m": sy, "angle_deg": angle}


def city_blocks(graph: nx.MultiDiGraph, bounds: Polygon, half_width: float) -> list[Polygon]:
    """Îlots = faces de la trame, en retrait de la demi-largeur de rue."""
    lines = [data["geometry"] for u, v, data in graph.edges(data=True) if u < v]
    faces = shapely.get_parts(shapely.polygonize(lines))
    blocks = []
    for face in faces:
        shrunk = face.buffer(-half_width, join_style="mitre").intersection(bounds)
        for part in shapely.get_parts(shrunk):
            if isinstance(part, Polygon) and part.area > 200:
                blocks.append(part)
    return blocks


def subdivide(
    blocks: list[Polygon], lot_m: float, angle_deg: float, coverage: np.ndarray, crs: str
) -> gpd.GeoDataFrame:
    """Découpe des îlots en lots carrés (grille tournée) et bâtiment centré par lot."""
    if not blocks:
        return gpd.GeoDataFrame({"block": []}, geometry=[], crs=crs)
    union = shapely.union_all(blocks)
    cx, cy = union.centroid.x, union.centroid.y
    half = 0.75 * max(union.bounds[2] - union.bounds[0], union.bounds[3] - union.bounds[1])
    n = math.ceil(half / lot_m)
    cells = [
        box(cx + i * lot_m, cy + j * lot_m, cx + (i + 1) * lot_m, cy + (j + 1) * lot_m)
        for i in range(-n, n)
        for j in range(-n, n)
    ]
    grid = gpd.GeoSeries(cells, crs=crs).rotate(angle_deg, origin=(cx, cy))
    grid = gpd.GeoDataFrame(geometry=grid.to_numpy(), crs=crs)
    blk = gpd.GeoDataFrame({"block": np.arange(len(blocks))}, geometry=blocks, crs=crs)
    lots = blk.overlay(grid, how="intersection", keep_geom_type=True).explode(index_parts=False)
    lots = lots[lots.area > 30]
    shrink = np.sqrt(lots.area.to_numpy()) * (1 - np.sqrt(coverage[lots["block"].to_numpy()])) / 2
    lots["geometry"] = lots.geometry.buffer(-shrink, join_style="mitre")
    lots = lots[~lots.geometry.is_empty].explode(index_parts=False)
    return lots[lots.area > 20].reset_index(drop=True)


def generate_map(index: int, data_cfg: DictConfig) -> UrbanLayers:
    """Génère la carte synthétique numéro ``index`` (reproductible)."""
    syn = data_cfg.synthetic
    seed = map_seed(int(syn.seed), index)
    rng = np.random.default_rng(seed)
    crs = str(data_cfg.crs)
    typology = choose_typology(rng, syn.typologies)
    typ = syn.typologies[typology]

    x0, y0 = SYNTH_ORIGIN
    extent, margin = float(syn.extent_m), float(syn.margin_m)
    bounds = box(x0, y0, x0 + extent + 2 * margin, y0 + extent + 2 * margin)
    zone = box(x0 + margin, y0 + margin, x0 + margin + extent, y0 + margin + extent)

    graph, params = street_graph(rng, typ, bounds, crs)
    blocks = city_blocks(graph, bounds, float(syn.street_half_width_m))
    probs = np.array(
        [typ.p_park, typ.p_vacant, typ.p_parking, typ.p_other, typ.p_obstacle, typ.p_empty]
    )
    draws = rng.random(len(blocks))
    cumulative = np.cumsum(probs)
    uses = [
        LAND_USES[int(np.searchsorted(cumulative, d, side="right"))]
        if d < cumulative[-1]
        else "residential"
        for d in draws
    ]
    by_use: dict[str, list[Polygon]] = {u: [] for u in (*LAND_USES, "residential")}
    for block, use in zip(blocks, uses, strict=True):
        by_use[use].append(block)

    # Bâtiments résidentiels (lots) et équipements.
    lot_m = float(rng.uniform(float(typ.lot_m[0]), float(typ.lot_m[1])))
    cov_lo, cov_hi = float(typ.building_coverage[0]), float(typ.building_coverage[1])
    coverage = rng.uniform(cov_lo, cov_hi, size=len(by_use["residential"]))
    lots = subdivide(by_use["residential"], lot_m, params["angle_deg"], coverage, crs)
    other = [
        b.buffer(-0.25 * math.sqrt(b.area) * (1 - float(np.mean(typ.building_coverage))))
        for b in by_use["other"]
    ]
    other = [g for g in other if not g.is_empty]
    buildings = gpd.GeoDataFrame(
        {
            "residential": [True] * len(lots) + [False] * len(other),
            "rule": ["synthetic"] * (len(lots) + len(other)),
            "osm_id": [None] * (len(lots) + len(other)),
        },
        geometry=[*lots.geometry, *other],
        crs=crs,
    )

    def layer(polys: list[Polygon], **columns: Any) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {k: [v] * len(polys) for k, v in columns.items()}, geometry=polys, crs=crs
        )

    green_polys = [p for p in by_use["park"] if p.area >= data_cfg.green.min_area_m2]
    green = layer(green_polys, kind="leisure=park", name=None, osm_id=None)
    green.insert(0, "green_id", np.arange(len(green)))
    cand_polys = by_use["vacant"] + by_use["parking"]
    candidates = gpd.GeoDataFrame(
        {
            "rule": ["landuse=vacant"] * len(by_use["vacant"])
            + ["amenity=parking"] * len(by_use["parking"]),
            "osm_id": [None] * len(cand_polys),
        },
        geometry=cand_polys,
        crs=crs,
    )
    candidates = candidates[candidates.area >= data_cfg.candidates.min_area_m2].reset_index(
        drop=True
    )
    candidates["area_m2"] = candidates.area.to_numpy()
    candidates["candidate_id"] = np.arange(len(candidates))
    obstacles = layer(by_use["obstacle"], kind="landuse=cemetery", osm_id=None)

    origins, dropped = attach_origins(buildings, graph, float(data_cfg.origins.max_snap_m))
    entrances = infer_entrances(green, graph, float(data_cfg.green.entrance_snap_m))
    layers = UrbanLayers(
        name=f"synth_{index:05d}",
        source="synthetic",
        crs=crs,
        zone=zone,
        graph=graph,
        buildings=buildings,
        green=green,
        green_excluded=empty_layer("green_excluded", crs),
        candidates=candidates[["candidate_id", "rule", "area_m2", "osm_id", "geometry"]],
        obstacles=obstacles,
        origins=origins,
        entrances=entrances,
        meta={
            "index": index,
            "seed": seed,
            "typology": typology,
            "lot_m": lot_m,
            "origins_dropped": dropped,
            "margin_m": margin,
            **params,
        },
    )
    layers.validate()
    return layers
