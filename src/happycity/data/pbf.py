"""Source OSM hors ligne : extrait Geofabrik (.osm.pbf) découpé localement avec pyosmium.

Avantages sur Overpass : instantané daté (reproductible), aucune charge sur les serveurs
publics, pas de saturation. Pour chaque zone, on écrit deux fichiers OSM XML lisibles
par OSMnx :

* ``<zone>.osm`` : objets intersectant l'emprise (+ objets référencés) → couches ;
* ``<zone>_walk.osm`` : uniquement les voies du filtre piéton OSMnx → graphe.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import requests
from osmnx import _overpass
from shapely import prepared
from shapely.geometry import Point, Polygon

log = logging.getLogger(__name__)

_CLAUSE = re.compile(r'\["([^"]+)"(?:(!?~)"([^"]*)")?\]')


def overpass_filter_predicate(filter_str: str) -> Callable[[Mapping[str, str]], bool]:
    """Traduit un filtre Overpass (``["k"]``, ``["k"~"re"]``, ``["k"!~"re"]``) en prédicat.

    Sémantique Overpass : ``["k"]`` exige la clé ; ``~`` exige la clé et une valeur qui
    correspond à l'expression ; ``!~`` est satisfait si la clé est absente ou ne
    correspond pas.
    """
    clauses = [(k, op, re.compile(v) if op else None) for k, op, v in _CLAUSE.findall(filter_str)]
    if not clauses:
        raise ValueError(f"filtre Overpass illisible : {filter_str!r}")

    def predicate(tags: Mapping[str, str]) -> bool:
        for key, op, regex in clauses:
            value = tags.get(key)
            if not op:
                if value is None:
                    return False
            elif op == "~":
                if value is None or regex is None or not regex.search(value):
                    return False
            elif value is not None and regex is not None and regex.search(value):
                return False
        return True

    return predicate


def network_predicate(network_type: str) -> Callable[[Mapping[str, str]], bool]:
    """Prédicat des voies retenues par OSMnx pour ``network_type`` (ex. ``walk``)."""
    return overpass_filter_predicate(_overpass._get_network_filter(network_type))


def download_pbf(url: str, target: Path, user_agent: str) -> Path:
    """Télécharge l'extrait s'il est absent et note sa date (``<fichier>.json``)."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(url, headers={"User-Agent": user_agent}, stream=True, timeout=60) as r:
        r.raise_for_status()
        with partial.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        info = {"url": url, "last_modified": r.headers.get("Last-Modified")}
    partial.replace(target)
    target.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return target


def snapshot_info(pbf: Path) -> dict[str, str | None]:
    """Métadonnées de l'instantané (URL, date) si connues."""
    path = pbf.with_suffix(".json")
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")))
    return {"url": None, "last_modified": None}


class _Area:
    def __init__(self, polygon: Polygon) -> None:
        self.bounds = polygon.bounds
        self.shape = prepared.prep(polygon)

    def contains(self, lon: float, lat: float) -> bool:
        xmin, ymin, xmax, ymax = self.bounds
        if not (xmin <= lon <= xmax and ymin <= lat <= ymax):
            return False
        return bool(self.shape.contains(Point(lon, lat)))


def clip_pbf(
    pbf: Path,
    areas_wgs84: Mapping[str, Polygon],
    out_dir: Path,
    network_type: str = "walk",
) -> dict[str, tuple[Path, Path]]:
    """Découpe l'extrait en un passage pour plusieurs zones.

    Une voie est retenue si l'un de ses nœuds est dans l'emprise ; une relation
    multipolygone si l'un de ses membres retenu. Les objets référencés (nœuds des voies,
    membres hors emprise) sont ajoutés par ``BackReferenceWriter``.

    Returns:
        Zone → ``(fichier objets, fichier réseau piéton)``.
    """
    # Import local : la DLL native d'osmium n'est nécessaire qu'ici (et peut être bloquée
    # par certaines politiques Windows, sans que le reste du module en dépende).
    import osmium

    out_dir.mkdir(parents=True, exist_ok=True)
    walkable = network_predicate(network_type)
    areas = {name: _Area(poly) for name, poly in areas_wgs84.items()}
    paths = {n: (out_dir / f"{n}.osm", out_dir / f"{n}_walk.osm") for n in areas}
    writers = {
        n: tuple(osmium.BackReferenceWriter(str(p), ref_src=str(pbf), overwrite=True) for p in pair)
        for n, pair in paths.items()
    }
    kept_ways: dict[str, set[int]] = {n: set() for n in areas}
    ux0 = min(a.bounds[0] for a in areas.values())
    uy0 = min(a.bounds[1] for a in areas.values())
    ux1 = max(a.bounds[2] for a in areas.values())
    uy1 = max(a.bounds[3] for a in areas.values())
    try:
        objects: Iterable[Any] = osmium.FileProcessor(str(pbf)).with_locations()
        for obj in objects:
            if obj.is_node():
                if not obj.tags or not obj.location.valid():
                    continue
                lon, lat = obj.location.lon, obj.location.lat
                if not (ux0 <= lon <= ux1 and uy0 <= lat <= uy1):
                    continue
                for name, area in areas.items():
                    if area.contains(lon, lat):
                        writers[name][0].add_node(obj)
            elif obj.is_way():
                coords = [(n.lon, n.lat) for n in obj.nodes if n.location.valid()]
                if not any(ux0 <= x <= ux1 and uy0 <= y <= uy1 for x, y in coords):
                    continue
                tags = dict(obj.tags)
                for name, area in areas.items():
                    if any(area.contains(x, y) for x, y in coords):
                        kept_ways[name].add(obj.id)
                        writers[name][0].add_way(obj)
                        if "highway" in tags and walkable(tags):
                            writers[name][1].add_way(obj)
            elif obj.is_relation() and obj.tags.get("type") == "multipolygon":
                members = [m.ref for m in obj.members if m.type == "w"]
                for name in areas:
                    if any(ref in kept_ways[name] for ref in members):
                        writers[name][0].add_relation(obj)
    finally:
        for pair in writers.values():
            for writer in pair:
                writer.close()
    return paths
