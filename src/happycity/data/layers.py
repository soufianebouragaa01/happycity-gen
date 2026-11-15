"""Format commun des couches vectorielles d'une zone (réelle OSM ou synthétique).

Une :class:`UrbanLayers` contient tout ce dont la rastérisation et les métriques ont
besoin : zone d'étude, graphe piéton métrique, bâtiments, espaces verts comptés (et
exclus, pour audit), parcelles candidates, obstacles, origines et entrées.
Toutes les géométries sont dans le même CRS métrique (EPSG:32629).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely import wkt
from shapely.geometry import Polygon

#: Colonnes obligatoires (hors ``geometry``) de chaque couche.
LAYER_COLUMNS: dict[str, tuple[str, ...]] = {
    "buildings": ("residential", "rule", "osm_id"),
    "green": ("green_id", "kind", "name", "osm_id"),
    "green_excluded": ("kind", "name", "reason", "osm_id"),
    "candidates": ("candidate_id", "rule", "area_m2", "osm_id"),
    "obstacles": ("kind", "osm_id"),
    "origins": ("building_id", "node", "snap_m"),
    "entrances": ("green_id", "node", "method"),
}

SOURCES = ("osm", "synthetic")


def empty_layer(name: str, crs: str) -> gpd.GeoDataFrame:
    """Couche vide avec les colonnes attendues."""
    columns: dict[str, list[Any]] = {c: [] for c in LAYER_COLUMNS[name]}
    return gpd.GeoDataFrame(columns, geometry=[], crs=crs)


@dataclass
class UrbanLayers:
    """Couches vectorielles d'une zone d'étude."""

    name: str
    source: str
    crs: str
    zone: Polygon
    graph: nx.MultiDiGraph
    buildings: gpd.GeoDataFrame
    green: gpd.GeoDataFrame
    green_excluded: gpd.GeoDataFrame
    candidates: gpd.GeoDataFrame
    obstacles: gpd.GeoDataFrame
    origins: gpd.GeoDataFrame
    entrances: gpd.GeoDataFrame
    meta: dict[str, Any] = field(default_factory=dict)

    def layer(self, name: str) -> gpd.GeoDataFrame:
        """Renvoie une couche par son nom."""
        value = getattr(self, name)
        assert isinstance(value, gpd.GeoDataFrame)
        return value

    def validate(self) -> None:
        """Vérifie source, CRS et colonnes ; lève ``ValueError`` sinon."""
        if self.source not in SOURCES:
            raise ValueError(f"source inconnue : {self.source!r}")
        if str(self.graph.graph.get("crs")).upper() != self.crs.upper():
            raise ValueError(f"CRS du graphe ({self.graph.graph.get('crs')}) ≠ {self.crs}")
        for name, columns in LAYER_COLUMNS.items():
            gdf = self.layer(name)
            if gdf.crs is None or gdf.crs != gpd.GeoSeries([], crs=self.crs).crs:
                raise ValueError(f"{name} : CRS {gdf.crs} ≠ {self.crs}")
            missing = set(columns) - set(gdf.columns)
            if missing:
                raise ValueError(f"{name} : colonnes manquantes {sorted(missing)}")

    def save(self, directory: Path) -> None:
        """Écrit les couches (GeoParquet), le graphe (GraphML) et les métadonnées (JSON)."""
        self.validate()
        directory.mkdir(parents=True, exist_ok=True)
        for name in LAYER_COLUMNS:
            self.layer(name).to_parquet(directory / f"{name}.parquet")
        ox.io.save_graphml(self.graph, directory / "graph.graphml")
        header = {
            "name": self.name,
            "source": self.source,
            "crs": self.crs,
            "zone_wkt": self.zone.wkt,
            "meta": self.meta,
        }
        (directory / "meta.json").write_text(
            json.dumps(header, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> UrbanLayers:
        """Relit une zone écrite par :meth:`save`."""
        header = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        layers = {name: gpd.read_parquet(directory / f"{name}.parquet") for name in LAYER_COLUMNS}
        zone = wkt.loads(header["zone_wkt"])
        assert isinstance(zone, Polygon)
        result = cls(
            name=header["name"],
            source=header["source"],
            crs=header["crs"],
            zone=zone,
            graph=ox.io.load_graphml(directory / "graph.graphml"),
            meta=header["meta"],
            **layers,
        )
        result.validate()
        return result
