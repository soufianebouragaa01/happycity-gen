"""Extraction OpenStreetMap d'une zone d'étude et construction des couches.

Deux étapes séparées pour pouvoir tester sans réseau :

* :func:`fetch_osm` interroge Overpass (objets + graphe piéton OSMnx) ;
* :func:`build_layers` applique les règles (``rules.py``) et la géométrie, sur des
  données déjà projetées en EPSG:32629.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeVar

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from omegaconf import DictConfig
from shapely.geometry import Polygon

from happycity.data import rules
from happycity.data.layers import UrbanLayers, empty_layer
from happycity.data.network import attach_origins, infer_entrances

#: Tags conservés depuis Overpass (les autres colonnes sont ignorées).
RELEVANT_KEYS = (
    "building",
    "leisure",
    "landuse",
    "natural",
    "amenity",
    "parking",
    "access",
    "fee",
    "tourism",
    "shop",
    "office",
    "craft",
    "healthcare",
    "railway",
    "waterway",
    "entrance",
    "barrier",
    "name",
)
POLYGONAL = ("Polygon", "MultiPolygon")
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- zones


def load_zone(boundary_path: Path, crs: str) -> Polygon:
    """Lit un polygone de zone (GeoJSON) et le projette ; lève ``ValueError`` si invalide."""
    gdf = gpd.read_file(boundary_path).to_crs(crs)
    geom = gdf.union_all()
    if not isinstance(geom, Polygon) or not geom.is_valid or geom.area <= 0:
        raise ValueError(f"{boundary_path} : un unique polygone valide est attendu")
    return geom


def is_usable_osm_boundary(
    gdf: gpd.GeoDataFrame, min_km2: float = 0.2, max_km2: float = 30.0
) -> bool:
    """Indique si un résultat de géocodage est une limite de quartier exploitable.

    Rejette les objets ponctuels (hôtel, mosquée… renvoyés pour un nom de quartier), les
    classes autres que ``boundary``/``place`` et les surfaces aberrantes.
    """
    if gdf.empty:
        return False
    row = gdf.iloc[0]
    if row.geometry.geom_type not in POLYGONAL or row.get("class") not in ("boundary", "place"):
        return False
    area_km2 = float(gdf.to_crs(gdf.estimate_utm_crs()).area.iloc[0]) / 1e6
    return min_km2 <= area_km2 <= max_km2


# --------------------------------------------------------------------------- Overpass


def overpass_tags(data_cfg: DictConfig) -> dict[str, bool | str | list[str]]:
    """Union des tags nécessaires à toutes les règles (une seule requête par zone)."""
    specs: Iterable[Mapping[str, Sequence[str]]] = (
        data_cfg.green.include_tags,
        data_cfg.green.exclude_tags,
        data_cfg.candidates.include_tags,
        data_cfg.candidates.exclude_tags,
        data_cfg.obstacles.tags,
        data_cfg.obstacles.line_tags,
    )
    values: dict[str, set[str]] = {"barrier": {"gate"}}
    for spec in specs:
        for key, vals in spec.items():
            values.setdefault(str(key), set()).update(str(v) for v in vals)
    tags: dict[str, bool | str | list[str]] = {k: sorted(v) for k, v in values.items()}
    for key in ("building", "landuse", "entrance"):  # toutes valeurs (classement des bâtiments)
        tags[key] = True
    tags.setdefault("tourism", [])
    tags["tourism"] = sorted({*tags["tourism"], "hotel", "resort"})  # type: ignore[misc]
    return tags


def configure_osmnx(extraction_cfg: DictConfig, cache_dir: Path) -> None:
    """Paramètre OSMnx : point d'accès, User-Agent générique, cache disque."""
    ox.settings.overpass_url = extraction_cfg.overpass_url
    ox.settings.http_user_agent = extraction_cfg.user_agent
    ox.settings.requests_timeout = int(extraction_cfg.timeout_s)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.log_console = False
    for tag in ("footway", "sidewalk"):
        if tag not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way = [*ox.settings.useful_tags_way, tag]


T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    attempts: int,
    base_wait_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Appelle ``fn`` avec attente exponentielle si Overpass est saturé (502/504/429…)."""
    for attempt in range(attempts):
        try:
            return fn()
        except (ox._errors.ResponseStatusCodeError, requests.RequestException) as exc:
            if attempt == attempts - 1:
                raise
            wait = base_wait_s * 2**attempt
            log.warning(
                "Overpass indisponible (%s) ; nouvel essai dans %.0f s",
                exc.__class__.__name__,
                wait,
            )
            sleep(wait)
    raise AssertionError("unreachable")


def fetch_overpass(
    area_wgs84: Polygon, data_cfg: DictConfig, cache_dir: Path
) -> tuple[gpd.GeoDataFrame, nx.MultiDiGraph]:
    """Télécharge les objets OSM et le graphe piéton d'une emprise (WGS84).

    Deux requêtes d'objets (bâtiments, puis le reste) pour ménager le serveur public.
    """
    extraction = data_cfg.extraction
    configure_osmnx(extraction, cache_dir)
    tags = overpass_tags(data_cfg)
    groups: list[dict[str, bool | str | list[str]]] = [
        {"building": True},
        {k: v for k, v in tags.items() if k != "building"},
    ]

    def retry(fn: Callable[[], T]) -> T:
        return with_retries(fn, int(extraction.max_retries), float(extraction.retry_base_s))

    parts = [retry(functools.partial(ox.features_from_polygon, area_wgs84, g)) for g in groups]
    features = pd.concat(parts)
    features = features[~features.index.duplicated(keep="first")]
    graph = retry(
        lambda: ox.graph_from_polygon(
            area_wgs84,
            network_type=data_cfg.network.network_type,
            retain_all=False,
            truncate_by_edge=True,
        )
    )
    return gpd.GeoDataFrame(features, crs=parts[0].crs), graph


def read_osm_xml(
    objects_xml: Path, walk_xml: Path, area_wgs84: Polygon, data_cfg: DictConfig
) -> tuple[gpd.GeoDataFrame, nx.MultiDiGraph]:
    """Lit les fichiers découpés par :func:`happycity.data.pbf.clip_pbf`."""
    configure_osmnx(data_cfg.extraction, Path(objects_xml).parent)
    features = ox.features_from_xml(objects_xml, polygon=area_wgs84, tags=overpass_tags(data_cfg))
    bidirectional = data_cfg.network.network_type in ox.settings.bidirectional_network_types
    graph = ox.graph_from_xml(walk_xml, bidirectional=bidirectional, retain_all=True)
    graph = ox.truncate.truncate_graph_polygon(graph, area_wgs84, truncate_by_edge=True)
    graph = ox.truncate.largest_component(graph, strongly=False)
    return features, graph


def prepare_features(features: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Aplatit l'index OSMnx en ``osm_id`` (``way/123``), garde les tags utiles, projette."""
    flat = features.reset_index()
    out = gpd.GeoDataFrame(
        {"osm_id": flat["element"].astype(str) + "/" + flat["id"].astype(str)},
        geometry=flat.geometry.to_numpy(),
        crs=features.crs,
    )
    for key in RELEVANT_KEYS:
        out[key] = flat[key].to_numpy() if key in flat.columns else None
    out = out.to_crs(crs)
    out["geometry"] = out.geometry.make_valid()
    return out


# --------------------------------------------------------------------------- géométrie


def _polygonal(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ne garde que les parties polygonales (après ``make_valid``/``clip``), une par ligne."""
    exploded = gdf.explode(index_parts=False)
    return exploded[exploded.geom_type == "Polygon"]


def _records(gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    return gdf[list(RELEVANT_KEYS)].to_dict("records")  # type: ignore[no-any-return]


def _subset(gdf: gpd.GeoDataFrame, spec: Mapping[str, Sequence[str]]) -> gpd.GeoDataFrame:
    mask = [rules.match_tags(t, spec) is not None for t in _records(gdf)]
    return gdf[np.asarray(mask, dtype=bool)] if len(gdf) else gdf


def _point_within(
    gdf: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, exclude_self: bool = False
) -> np.ndarray:
    """Masque : le point représentatif de chaque objet est dans l'une des ``areas``."""
    if gdf.empty or areas.empty:
        return np.zeros(len(gdf), dtype=bool)
    points = gpd.GeoDataFrame(
        {"osm_id": gdf["osm_id"].to_numpy(), "row": np.arange(len(gdf))},
        geometry=gdf.geometry.representative_point().to_numpy(),
        crs=gdf.crs,
    )
    hits = gpd.sjoin(points, areas[["osm_id", "geometry"]], predicate="within")
    if exclude_self:
        hits = hits[hits["osm_id_left"] != hits["osm_id_right"]]
    mask = np.zeros(len(gdf), dtype=bool)
    mask[hits["row"].to_numpy()] = True
    return mask


def _frame(
    data: Mapping[str, Sequence[Any] | np.ndarray], geometry: Any, crs: str
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(dict(data), geometry=geometry, crs=crs).reset_index(drop=True)


# --------------------------------------------------------------------------- couches


def _buildings(polys: gpd.GeoDataFrame, data_cfg: DictConfig) -> gpd.GeoDataFrame:
    cfg = data_cfg.buildings
    built = polys[polys["building"].notna()]
    if built.empty:
        return empty_layer("buildings", polys.crs)
    res_landuse = _subset(polys, {"landuse": ["residential"]})
    non_res = _subset(
        polys, {"landuse": list(cfg.non_residential_landuse), "tourism": ["hotel", "resort"]}
    )
    in_res = _point_within(built, res_landuse)
    in_non_res = _point_within(built, non_res, exclude_self=True)
    decisions = [
        rules.building_decision(
            tags,
            in_residential_landuse=bool(a),
            in_non_residential_area=bool(b),
            residential_values=list(cfg.residential_values),
            non_residential_values=list(cfg.non_residential_values),
            non_residential_keys=list(cfg.non_residential_keys),
            default_yes=cfg.default_yes,
        )
        for tags, a, b in zip(_records(built), in_res, in_non_res, strict=True)
    ]
    return _frame(
        {
            "residential": [d[0] for d in decisions],
            "rule": [d[1] for d in decisions],
            "osm_id": built["osm_id"].to_numpy(),
        },
        built.geometry.to_numpy(),
        polys.crs,
    )


def _green(
    polys: gpd.GeoDataFrame, area: Polygon, data_cfg: DictConfig
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    cfg = data_cfg.green
    records = _records(polys)
    decisions = [
        rules.green_decision(
            t, cfg.include_tags, cfg.exclude_tags, list(cfg.exclude_access), bool(cfg.exclude_fee)
        )
        for t in records
    ]
    is_green = np.array([reason != "not_green" for _, reason in decisions], dtype=bool)
    green = polys[is_green].copy()
    if green.empty:
        return empty_layer("green", polys.crs), empty_layer("green_excluded", polys.crs)
    green_decisions = [d for d, keep in zip(decisions, is_green, strict=True) if keep]
    green["included"] = [d[0] for d in green_decisions]
    green["reason"] = [d[1] for d in green_decisions]
    green["kind"] = [rules.match_tags(t, cfg.include_tags) for t in _records(green)]
    # Exclusion spatiale : contenu dans un hôtel, un golf, une oliveraie…
    containers = _subset(polys, cfg.exclude_tags)
    inside = _point_within(green, containers, exclude_self=True)
    green.loc[inside & green["included"].to_numpy(), "reason"] = "inside_excluded_area"
    green.loc[inside, "included"] = False
    manual = dict(cfg.manual_exclude)
    is_manual = green["osm_id"].isin(list(manual)).to_numpy()
    green.loc[is_manual, "reason"] = green.loc[is_manual, "osm_id"].map(
        lambda i: f"manual:{manual[i]}"
    )
    green.loc[is_manual, "included"] = False
    green = _polygonal(green.clip(area))
    green = green[green.area >= cfg.min_area_m2]

    kept = green[green["included"]]
    included = _frame(
        {
            "green_id": np.arange(len(kept)),
            "kind": kept["kind"].to_numpy(),
            "name": kept["name"].to_numpy(),
            "osm_id": kept["osm_id"].to_numpy(),
        },
        kept.geometry.to_numpy(),
        polys.crs,
    )
    dropped = green[~green["included"]]
    excluded = _frame(
        {
            "kind": dropped["kind"].to_numpy(),
            "name": dropped["name"].to_numpy(),
            "reason": dropped["reason"].to_numpy(),
            "osm_id": dropped["osm_id"].to_numpy(),
        },
        dropped.geometry.to_numpy(),
        polys.crs,
    )
    return included, excluded


def _obstacles(
    feats: gpd.GeoDataFrame, polys: gpd.GeoDataFrame, area: Polygon, data_cfg: DictConfig
) -> gpd.GeoDataFrame:
    cfg = data_cfg.obstacles
    areas = _subset(polys, cfg.tags)
    lines = feats[feats.geom_type.isin(["LineString", "MultiLineString"])]
    lines = _subset(lines, cfg.line_tags)
    kinds = [rules.match_tags(t, cfg.tags) for t in _records(areas)]
    kinds += [rules.match_tags(t, cfg.line_tags) for t in _records(lines)]
    geometry = [*areas.geometry, *lines.geometry.buffer(cfg.line_buffer_m)]
    if not geometry:
        return empty_layer("obstacles", polys.crs)
    obstacles = _frame(
        {"kind": kinds, "osm_id": [*areas["osm_id"], *lines["osm_id"]]}, geometry, polys.crs
    )
    return _polygonal(obstacles.clip(area)).reset_index(drop=True)


def _candidates(
    polys: gpd.GeoDataFrame,
    area: Polygon,
    blockers: Sequence[gpd.GeoDataFrame],
    data_cfg: DictConfig,
) -> gpd.GeoDataFrame:
    cfg = data_cfg.candidates
    matched = [
        rules.candidate_decision(t, cfg.include_tags, list(cfg.parking_types))
        for t in _records(polys)
    ]
    mask = np.array([m is not None for m in matched], dtype=bool)
    raw = polys[mask]
    if raw.empty:
        return empty_layer("candidates", polys.crs)
    cand = _frame(
        {"rule": [m for m in matched if m is not None], "osm_id": raw["osm_id"].to_numpy()},
        raw.geometry.to_numpy(),
        polys.crs,
    )
    cand = _polygonal(cand.clip(area))
    blocking = [
        b[["geometry"]] for b in (*blockers, _subset(polys, cfg.exclude_tags)) if not b.empty
    ]
    if blocking and not cand.empty:
        mask_gdf = gpd.GeoDataFrame(pd.concat(blocking, ignore_index=True), crs=polys.crs)
        cand = _polygonal(cand.overlay(mask_gdf, how="difference", keep_geom_type=True))
    cand = cand[cand.area >= cfg.min_area_m2].reset_index(drop=True)
    cand["area_m2"] = cand.area.to_numpy()
    cand["candidate_id"] = np.arange(len(cand))
    return cand[["candidate_id", "rule", "area_m2", "osm_id", "geometry"]]


def build_layers(
    name: str,
    zone: Polygon,
    features: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    data_cfg: DictConfig,
    source: Literal["osm", "synthetic"] = "osm",
) -> UrbanLayers:
    """Construit les couches d'une zone à partir d'objets OSM préparés et d'un graphe projeté.

    Args:
        name: identifiant de la zone.
        zone: polygone de la zone d'étude (CRS métrique).
        features: sortie de :func:`prepare_features`.
        graph: graphe piéton projeté dans ``data_cfg.crs``.
        data_cfg: section ``data`` de la configuration.
        source: ``"osm"`` (ou ``"synthetic"`` pour des tests).
    """
    crs = str(data_cfg.crs)
    area = zone.buffer(float(data_cfg.extraction.margin_m), join_style="mitre")
    feats = features[features.intersects(area)]
    polys = _polygonal(feats[feats.geom_type.isin(POLYGONAL)])

    buildings = _buildings(polys, data_cfg)
    green, green_excluded = _green(polys, area, data_cfg)
    obstacles = _obstacles(feats, polys, area, data_cfg)
    candidates = _candidates(polys, area, [buildings, green, obstacles], data_cfg)

    origins, dropped = attach_origins(buildings, graph, float(data_cfg.origins.max_snap_m))
    points = feats[feats.geom_type == "Point"]
    gate = points["entrance"].notna() | (points["barrier"] == "gate")
    entrances = infer_entrances(
        green, graph, float(data_cfg.green.entrance_snap_m), points[gate].geometry
    )
    meta = {
        "extracted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "margin_m": float(data_cfg.extraction.margin_m),
        "origins_dropped": int(dropped),
    }
    layers = UrbanLayers(
        name=name,
        source=source,
        crs=crs,
        zone=zone,
        graph=graph,
        buildings=buildings,
        green=green,
        green_excluded=green_excluded,
        candidates=candidates,
        obstacles=obstacles,
        origins=origins,
        entrances=entrances,
        meta=meta,
    )
    layers.validate()
    return layers


def layers_from_osm(
    name: str,
    zone: Polygon,
    features: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    data_cfg: DictConfig,
) -> UrbanLayers:
    """Projette les objets et le graphe bruts (WGS84) puis construit les couches."""
    crs = str(data_cfg.crs)
    graph = ox.project_graph(graph, to_crs=crs)
    graph.graph["crs"] = crs
    return build_layers(name, zone, prepare_features(features, crs), graph, data_cfg)


def extraction_area(zone: Polygon, data_cfg: DictConfig) -> Polygon:
    """Emprise d'extraction (zone + marge de contexte) en WGS84."""
    crs = str(data_cfg.crs)
    area = zone.buffer(float(data_cfg.extraction.margin_m), join_style="mitre")
    area_wgs84 = gpd.GeoSeries([area], crs=crs).to_crs(4326).iloc[0]
    assert isinstance(area_wgs84, Polygon)
    return area_wgs84
