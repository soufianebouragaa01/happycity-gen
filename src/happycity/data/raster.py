"""Rastérisation des couches vectorielles en tuiles 128×128 à échelle métrique constante.

Conventions (identiques pour toutes les tuiles) :

* tuile carrée de ``size_px × m_per_px`` mètres (128 × 8 m = 1024 m) ;
* ligne 0 = nord, colonne 0 = ouest ; le pixel ``(r, c)`` couvre
  ``[xmin + c·res, xmin + (c+1)·res] × [ymax − (r+1)·res, ymax − r·res]`` ;
* canaux surfaciques = **fraction** du pixel couverte, obtenue par suréchantillonnage ;
* réseau piéton = 1 si un tronçon touche le pixel ;
* ``origins`` = nombre d'origines résidentielles dans le pixel ;
* ``R`` (parcelles candidates) et ``candidate_ids`` : échantillonnage au centre du pixel,
  un identifiant local par parcelle (``-1`` hors parcelle).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from happycity.data.layers import UrbanLayers
from happycity.data.network import graph_edges

# --------------------------------------------------------------------------- tuiles


@dataclass(frozen=True)
class TileSpec:
    """Emprise et étiquettes d'une tuile."""

    tile_id: str
    source: str
    zone: str
    split: str
    xmin: float
    ymin: float
    size_m: float

    @property
    def geometry(self) -> Polygon:
        """Polygone de la tuile."""
        return box(self.xmin, self.ymin, self.xmin + self.size_m, self.ymin + self.size_m)


def tile_grid(zone: Polygon, size_m: float, stride_m: float) -> list[tuple[int, int, float, float]]:
    """Tuiles entièrement contenues dans la zone : ``(i, j, xmin, ymin)``, i vers l'est."""
    xmin, ymin, xmax, ymax = zone.bounds
    xs = np.arange(xmin, xmax - size_m + 1e-6, stride_m)
    ys = np.arange(ymin, ymax - size_m + 1e-6, stride_m)
    inside = zone.buffer(1e-6)
    return [
        (i, j, float(x), float(y))
        for i, x in enumerate(xs)
        for j, y in enumerate(ys)
        if inside.contains(box(x, y, x + size_m, y + size_m))
    ]


def split_conflicts(
    tile: Polygon, split: str, zones: Mapping[str, tuple[Polygon, str]], buffer_m: float
) -> list[str]:
    """Zones d'un AUTRE split à moins de ``buffer_m`` de la tuile (→ tuile exclue).

    Tolérance de 1 cm : les zones manuelles sont espacées d'exactement 400 m, à
    l'arrondi près de l'aller-retour WGS84 ↔ UTM.
    """
    return sorted(
        name
        for name, (geom, other) in zones.items()
        if other != split and tile.distance(geom) < buffer_m - 0.01
    )


# --------------------------------------------------------------------------- primitives


def _transform(spec: TileSpec, res: float) -> Any:
    return from_origin(spec.xmin, spec.ymin + spec.size_m, res, res)


def coverage_raster(
    geoms: Iterable[BaseGeometry], spec: TileSpec, size_px: int, supersample: int
) -> np.ndarray:
    """Fraction de chaque pixel couverte par l'union des géométries (suréchantillonnage)."""
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((size_px, size_px), dtype=np.float32)
    fine_px = size_px * supersample
    fine = rasterize(
        shapes,
        out_shape=(fine_px, fine_px),
        transform=_transform(spec, spec.size_m / fine_px),
        fill=0,
        dtype="uint8",
    )
    blocks = fine.reshape(size_px, supersample, size_px, supersample)
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)


def line_raster(lines: Iterable[BaseGeometry], spec: TileSpec, size_px: int) -> np.ndarray:
    """1 pour chaque pixel touché par une ligne."""
    shapes = [(g, 1) for g in lines if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((size_px, size_px), dtype=np.float32)
    out = rasterize(
        shapes,
        out_shape=(size_px, size_px),
        transform=_transform(spec, spec.size_m / size_px),
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    return np.asarray(out, dtype=np.float32)


def pixel_of(
    x: np.ndarray, y: np.ndarray, spec: TileSpec, size_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """Indices (ligne, colonne) de points en coordonnées métriques (peuvent sortir de la tuile)."""
    res = spec.size_m / size_px
    col = np.floor((x - spec.xmin) / res).astype(np.int64)
    row = np.floor((spec.ymin + spec.size_m - y) / res).astype(np.int64)
    return row, col


def count_raster(x: np.ndarray, y: np.ndarray, spec: TileSpec, size_px: int) -> np.ndarray:
    """Nombre de points par pixel (points hors tuile ignorés)."""
    out = np.zeros((size_px, size_px), dtype=np.float32)
    row, col = pixel_of(x, y, spec, size_px)
    ok = (row >= 0) & (row < size_px) & (col >= 0) & (col < size_px)
    np.add.at(out, (row[ok], col[ok]), 1.0)
    return out


def id_raster(geoms: Sequence[BaseGeometry], spec: TileSpec, size_px: int) -> np.ndarray:
    """Identifiant (0…n−1) de la géométrie contenant le centre du pixel, −1 sinon."""
    if len(geoms) == 0:
        return np.full((size_px, size_px), -1, dtype=np.int32)
    out = rasterize(
        [(g, i + 1) for i, g in enumerate(geoms)],
        out_shape=(size_px, size_px),
        transform=_transform(spec, spec.size_m / size_px),
        fill=0,
        dtype="int32",
    )
    return np.asarray(out, dtype=np.int32) - 1


# --------------------------------------------------------------------------- tuile complète


@dataclass
class TileRaster:
    """Rasters et tables d'une tuile."""

    spec: TileSpec
    channel_names: tuple[str, ...]
    channels: np.ndarray  # (C, H, W) float32
    candidate_ids: np.ndarray  # (H, W) int32, −1 hors parcelle
    candidates: pd.DataFrame  # local_id, candidate_id, rule, area_m2, n_px, adjacent
    origins: pd.DataFrame  # row, col, x, y, node, building_id

    @property
    def candidate_mask(self) -> np.ndarray:
        """Masque R des parcelles candidates."""
        return self.candidate_ids >= 0

    def channel(self, name: str) -> np.ndarray:
        """Un canal par son nom."""
        return np.asarray(self.channels[self.channel_names.index(name)])

    def save(self, path: Path) -> None:
        """Écrit la tuile (``.npz`` compressé)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "channels": self.channels.astype(np.float16),
            "channel_names": np.array(self.channel_names),
            "candidate_ids": self.candidate_ids.astype(np.int16),
            "spec": np.array([str(self.spec.__dict__)]),
        }
        arrays |= {f"cand_{c}": _plain(self.candidates[c]) for c in self.candidates.columns}
        arrays |= {f"orig_{c}": _plain(self.origins[c]) for c in self.origins.columns}
        np.savez_compressed(path, allow_pickle=False, **arrays)

    @classmethod
    def load(cls, path: Path) -> TileRaster:
        """Relit une tuile écrite par :meth:`save`."""
        import ast

        with np.load(path, allow_pickle=False) as data:
            spec = TileSpec(**ast.literal_eval(str(data["spec"][0])))
            cand = {k[5:]: data[k] for k in data.files if k.startswith("cand_")}
            orig = {k[5:]: data[k] for k in data.files if k.startswith("orig_")}
            return cls(
                spec=spec,
                channel_names=tuple(str(n) for n in data["channel_names"]),
                channels=data["channels"].astype(np.float32),
                candidate_ids=data["candidate_ids"].astype(np.int32),
                candidates=pd.DataFrame(cand, columns=CANDIDATE_COLUMNS),
                origins=pd.DataFrame(orig, columns=ORIGIN_COLUMNS),
            )


def _plain(column: pd.Series) -> np.ndarray:
    """Tableau sans objets Python (relisible avec ``allow_pickle=False``)."""
    values = column.to_numpy()
    return np.asarray(values.astype(str) if values.dtype == object else values)


CANDIDATE_COLUMNS = ["local_id", "candidate_id", "rule", "area_m2", "n_px", "adjacent"]
ORIGIN_COLUMNS = ["row", "col", "x", "y", "node", "building_id"]


def _in_box(gdf: gpd.GeoDataFrame, geom: Polygon) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    return gdf.iloc[gdf.sindex.query(geom, predicate="intersects")]


def _clipped(gdf: gpd.GeoDataFrame, geom: Polygon) -> list[BaseGeometry]:
    return [g.intersection(geom) for g in _in_box(gdf, geom).geometry]


def tile_candidates(
    layers: UrbanLayers, spec: TileSpec, edges: gpd.GeoDataFrame, data_cfg: DictConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    """Parcelles candidates de la tuile : carte d'identifiants et table (adjacence au réseau)."""
    size_px = int(data_cfg.tile.size_px)
    res = spec.size_m / size_px
    tile = spec.geometry
    sel = _in_box(layers.candidates, tile)
    parts = [g.intersection(tile) for g in sel.geometry]
    keep = [k for k, g in enumerate(parts) if g.area >= float(data_cfg.candidates.min_area_m2)]
    geoms = [parts[k] for k in keep]
    ids = id_raster(geoms, spec, size_px)
    n_px = np.bincount(ids[ids >= 0].ravel(), minlength=len(geoms))
    present = np.flatnonzero(n_px > 0)  # parcelles trop fines : aucun centre de pixel
    remap = np.full(len(geoms) + 1, -1, dtype=np.int32)
    remap[present] = np.arange(len(present))
    ids = remap[ids]  # −1 → dernier élément de remap (= −1)
    reach = float(data_cfg.network.adjacency_px) * res
    adjacent = [
        bool(len(edges.sindex.query(geoms[k].buffer(reach), predicate="intersects")))
        for k in present
    ]
    table = pd.DataFrame(
        {
            "local_id": np.arange(len(present)),
            "candidate_id": sel["candidate_id"].to_numpy()[np.asarray(keep, dtype=int)[present]]
            if len(present)
            else np.array([], dtype=int),
            "rule": sel["rule"].to_numpy()[np.asarray(keep, dtype=int)[present]].astype(str)
            if len(present)
            else np.array([], dtype=str),
            "area_m2": np.array([geoms[k].area for k in present], dtype=float),
            "n_px": n_px[present].astype(int),
            "adjacent": np.array(adjacent, dtype=bool),
        },
        columns=CANDIDATE_COLUMNS,
    )
    return ids, table


def rasterize_tile(layers: UrbanLayers, spec: TileSpec, data_cfg: DictConfig) -> TileRaster:
    """Rastérise une tuile à partir des couches vectorielles de sa zone.

    Raises:
        ValueError: si la tuile ne respecte pas l'échelle constante ``size_px × m_per_px``.
    """
    size_px = int(data_cfg.tile.size_px)
    expected_m = size_px * float(data_cfg.tile.m_per_px)
    if abs(spec.size_m - expected_m) > 1e-6:
        raise ValueError(
            f"tuile de {spec.size_m} m ≠ {expected_m} m : l'échelle doit être constante"
        )
    sup = int(data_cfg.raster.supersample)
    tile = spec.geometry
    edges = graph_edges(layers.graph)
    res_b = layers.buildings[layers.buildings["residential"].astype(bool)]
    other_b = layers.buildings[~layers.buildings["residential"].astype(bool)]
    origins = _in_box(layers.origins, tile)
    ox_, oy_ = origins.geometry.x.to_numpy(), origins.geometry.y.to_numpy()

    makers: dict[str, Callable[[], np.ndarray]] = {
        "residential_buildings": lambda: coverage_raster(_clipped(res_b, tile), spec, size_px, sup),
        "other_buildings": lambda: coverage_raster(_clipped(other_b, tile), spec, size_px, sup),
        "walk_network": lambda: line_raster(_clipped(edges, tile), spec, size_px),
        "public_green": lambda: coverage_raster(_clipped(layers.green, tile), spec, size_px, sup),
        "obstacles": lambda: coverage_raster(_clipped(layers.obstacles, tile), spec, size_px, sup),
        "origins": lambda: count_raster(ox_, oy_, spec, size_px),
    }
    names = tuple(str(c) for c in data_cfg.channels)
    unknown = [n for n in names if n not in makers]
    if unknown:
        raise ValueError(f"canaux inconnus : {unknown}")
    channels = np.stack([makers[n]() for n in names])
    ids, cand = tile_candidates(layers, spec, edges, data_cfg)

    row, col = pixel_of(ox_, oy_, spec, size_px)
    ok = (row >= 0) & (row < size_px) & (col >= 0) & (col < size_px)
    orig = pd.DataFrame(
        {
            "row": row[ok],
            "col": col[ok],
            "x": ox_[ok],
            "y": oy_[ok],
            "node": origins["node"].to_numpy(dtype=np.int64)[ok],
            "building_id": origins["building_id"].to_numpy(dtype=np.int64)[ok],
        },
        columns=ORIGIN_COLUMNS,
    )
    return TileRaster(spec, names, channels, ids, cand, orig)


# --------------------------------------------------------------------------- annotations


def tile_annotations(
    layers: UrbanLayers, raster: TileRaster, data_cfg: DictConfig
) -> dict[str, Any]:
    """Couverture des annotations d'une tuile (contrôle avant entraînement)."""
    spec = raster.spec
    context = spec.geometry.buffer(float(data_cfg.access_radius_m), join_style="mitre")
    green_ctx = _in_box(layers.green, context)
    entrances = _in_box(layers.entrances, context)
    reachable_green = set(entrances["green_id"]) & set(green_ctx["green_id"])
    cand = raster.candidates
    n_origins = len(raster.origins)
    stats: dict[str, Any] = {
        "tile_id": spec.tile_id,
        "source": spec.source,
        "zone": spec.zone,
        "split": spec.split,
        "n_origins": n_origins,
        "residential_cover": round(float(raster.channel("residential_buildings").mean()), 4)
        if "residential_buildings" in raster.channel_names
        else None,
        "walk_px_share": round(float(raster.channel("walk_network").mean()), 4)
        if "walk_network" in raster.channel_names
        else None,
        "green_cover": round(float(raster.channel("public_green").mean()), 4)
        if "public_green" in raster.channel_names
        else None,
        "n_green_context": len(green_ctx),
        "n_green_with_entrance": len(reachable_green),
        "n_entrances_context": len(entrances),
        "n_candidates": len(cand),
        "n_candidates_adjacent": int(cand["adjacent"].sum()),
        "candidate_px": int(raster.candidate_mask.sum()),
        "adjacent_candidate_px": int(cand.loc[cand["adjacent"], "n_px"].sum()),
    }
    flags = []
    if n_origins < int(data_cfg.raster.min_origins):
        flags.append("peu d'origines")
    if stats["n_candidates_adjacent"] == 0:
        flags.append("aucune parcelle adjacente au réseau")
    if stats["n_green_with_entrance"] == 0:
        flags.append("aucun espace vert accessible dans le contexte")
    stats["flags"] = ", ".join(flags)
    # Une tuile sans parc accessible reste utile (A400 = 0 avant) ; elle doit avoir des
    # origines et au moins une parcelle aménageable reliée au réseau.
    stats["usable"] = (
        n_origins >= int(data_cfg.raster.min_origins) and stats["n_candidates_adjacent"] > 0
    )
    return stats
