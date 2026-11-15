"""Construction des couches à partir d'objets OSM factices (sans réseau)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from omegaconf import DictConfig
from shapely.geometry import LineString, Point, Polygon, box

from conftest import CRS, X0, Y0, make_features
from happycity.config import load_config
from happycity.data.layers import UrbanLayers
from happycity.data.osm import (
    build_layers,
    is_usable_osm_boundary,
    load_zone,
    overpass_tags,
    with_retries,
)
from happycity.data.pbf import network_predicate, overpass_filter_predicate


def b(x: float, y: float, w: float, h: float | None = None) -> Polygon:
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + (h if h is not None else w))


@pytest.fixture
def scene(cfg: DictConfig, grid_graph: Callable[..., nx.MultiDiGraph]) -> UrbanLayers:
    """Zone de 400 m avec un échantillon de cas couverts par les règles."""
    rows = [
        ({"landuse": "residential", "osm_id": "way/1"}, b(0, 0, 200)),
        ({"building": "yes", "osm_id": "way/2"}, b(20, 20, 10)),  # résidentiel (landuse)
        ({"building": "apartments", "osm_id": "way/3"}, b(250, 20, 10)),  # explicite
        ({"building": "yes", "shop": "bakery", "osm_id": "way/4"}, b(60, 20, 10)),  # autre
        ({"leisure": "park", "name": "Parc public", "osm_id": "way/5"}, b(110, 110, 80)),
        (
            {"leisure": "garden", "fee": "yes", "name": "Jardin payant", "osm_id": "way/6"},
            b(210, 210, 60),
        ),
        ({"tourism": "hotel", "osm_id": "way/7"}, b(300, 300, 90)),
        ({"leisure": "garden", "name": "Jardin d'hôtel", "osm_id": "way/8"}, b(320, 320, 30)),
        ({"landuse": "brownfield", "osm_id": "way/9"}, b(210, 110, 30)),  # 900 m²
        ({"building": "yes", "osm_id": "way/10"}, b(210, 110, 10)),  # grignote la friche
        ({"landuse": "grass", "osm_id": "way/11"}, b(310, 110, 20)),  # 400 m² < 500 : rejetée
        ({"amenity": "parking", "parking": "multi-storey", "osm_id": "way/12"}, b(110, 210, 40)),
        ({"amenity": "parking", "osm_id": "way/13"}, b(10, 310, 40)),  # surface (1600 m²)
        (
            {"railway": "rail", "osm_id": "way/14"},
            LineString([(X0, Y0 + 395), (X0 + 400, Y0 + 395)]),
        ),
        ({"entrance": "yes", "osm_id": "node/15"}, Point(X0 + 110, Y0 + 150)),
    ]
    zone = b(0, 0, 400)
    small = load_config(["data.extraction.margin_m=0"])
    return build_layers("test", zone, make_features(rows), grid_graph(n=5), small.data)


def test_buildings_classification(scene: UrbanLayers) -> None:
    rules = dict(
        zip(
            scene.buildings["osm_id"],
            zip(scene.buildings["residential"], scene.buildings["rule"], strict=True),
            strict=True,
        )
    )
    assert rules["way/2"] == (True, "landuse")
    assert rules["way/3"] == (True, "explicit")
    assert rules["way/4"] == (False, "tag:shop")


def test_green_inclusion_and_audit(scene: UrbanLayers) -> None:
    assert list(scene.green["name"]) == ["Parc public"]
    reasons = dict(zip(scene.green_excluded["name"], scene.green_excluded["reason"], strict=True))
    assert reasons == {"Jardin payant": "fee", "Jardin d'hôtel": "inside_excluded_area"}


def test_park_entrance_uses_osm_gate(scene: UrbanLayers) -> None:
    assert set(scene.entrances["method"]) == {"osm"}
    assert set(scene.entrances["green_id"]) == {0}


def test_candidates_rule(scene: UrbanLayers) -> None:
    cands = scene.candidates.set_index("osm_id")
    assert set(cands.index) == {"way/9", "way/13"}  # herbe trop petite, parking à étages exclus
    assert cands.loc["way/9", "area_m2"] == pytest.approx(900 - 100)  # bâtiment retiré
    assert cands.loc["way/13", "rule"] == "amenity=parking"
    overlap = scene.candidates.intersection(scene.buildings.union_all()).area
    assert overlap.max() < 1e-6  # contact par un bord autorisé, pas de recouvrement


def test_obstacles_include_buffered_railway(scene: UrbanLayers) -> None:
    rail = scene.obstacles[scene.obstacles["kind"] == "railway=rail"]
    assert len(rail) == 1
    assert rail.area.iloc[0] == pytest.approx(400 * 10, rel=0.05)


def test_origins_are_residential_only(scene: UrbanLayers) -> None:
    residential_ids = set(scene.buildings.index[scene.buildings["residential"]])
    assert set(scene.origins["building_id"]) <= residential_ids
    assert len(scene.origins) == int(scene.buildings["residential"].sum())


def test_overpass_tags_cover_all_rules(cfg: DictConfig) -> None:
    tags = overpass_tags(cfg.data)
    assert tags["building"] is True
    assert tags["landuse"] is True
    for key, value in [("leisure", "park"), ("leisure", "golf_course"), ("amenity", "parking")]:
        values = tags[key]
        assert isinstance(values, list)
        assert value in values
    assert tags["railway"] == ["rail"]


def test_with_retries_backs_off() -> None:
    import requests

    calls: list[int] = []
    waits: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise requests.ConnectionError("busy")
        return "ok"

    assert with_retries(flaky, attempts=4, base_wait_s=10, sleep=waits.append) == "ok"
    assert waits == [10, 20]
    with pytest.raises(requests.ConnectionError):
        with_retries(lambda: (_ for _ in ()).throw(requests.ConnectionError()), 2, 1, waits.append)


# --------------------------------------------------------------------------- zones


def test_geocoding_results_are_checked() -> None:
    hotel = gpd.GeoDataFrame(
        {"class": ["tourism"]}, geometry=[box(-8.01, 31.62, -8.009, 31.621)], crs=4326
    )
    assert not is_usable_osm_boundary(hotel)
    district = gpd.GeoDataFrame(
        {"class": ["place"]}, geometry=[box(-8.02, 31.62, -8.00, 31.64)], crs=4326
    )
    assert is_usable_osm_boundary(district)
    assert not is_usable_osm_boundary(district.iloc[:0])


def test_load_zone_rejects_multiple_parts(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)], crs=CRS)
    path = tmp_path / "two.geojson"
    gdf.to_file(path, driver="GeoJSON")
    with pytest.raises(ValueError, match="polygone"):
        load_zone(path, CRS)


BOUNDARIES = Path(__file__).resolve().parents[1] / "data" / "boundaries"


def test_repository_boundaries(cfg: DictConfig) -> None:
    """Zones manuelles : présentes, carrées de 2048 m, et documentées."""
    for district in cfg.data.districts:
        path = BOUNDARIES / district.boundary_file
        zone = load_zone(path, CRS)
        assert zone.area == pytest.approx(2048**2, rel=1e-3)
        props = json.loads(path.read_text(encoding="utf-8"))["features"][0]["properties"]
        assert props["anchor"]
        assert props["method"]


def test_zones_of_different_splits_respect_buffer(cfg: DictConfig) -> None:
    split = cfg.data.split
    part = {n: p for p in ("train", "val", "test") for n in split[p]}
    zones = {d.name: load_zone(BOUNDARIES / d.boundary_file, CRS) for d in cfg.data.districts}
    names = sorted(zones)
    for i, a in enumerate(names):
        for c in names[i + 1 :]:
            if part[a] != part[c]:
                # tolérance de 1 cm : aller-retour UTM → WGS84 (GeoJSON) → UTM
                assert zones[a].distance(zones[c]) >= split.border_buffer_m - 0.01, (a, c)


# --------------------------------------------------------------------------- filtre piéton


def test_overpass_filter_predicate() -> None:
    pred = overpass_filter_predicate('["highway"]["foot"!~"no"]["service"~"alley"]')
    assert pred({"highway": "service", "service": "alley"})
    assert not pred({"highway": "service", "service": "alley", "foot": "no"})
    assert not pred({"highway": "service"})
    assert not pred({"service": "alley"})


def test_walk_predicate_matches_osmnx_filter() -> None:
    walk = network_predicate("walk")
    assert walk({"highway": "footway"})
    assert walk({"highway": "residential"})
    assert not walk({"highway": "motorway"})
    assert not walk({"highway": "cycleway"})
    assert not walk({"highway": "residential", "access": "private"})
    assert not walk({"highway": "primary", "sidewalk": "separate"})
    assert not walk({"building": "yes"})
