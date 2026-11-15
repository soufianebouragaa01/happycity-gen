"""Rapport de couverture OSM / annotations par zone d'étude.

Les statistiques sont calculées DANS la zone (les marges de contexte sont exclues).
Elles servent (1) à décider si une zone réelle est exploitable avant le jalon J1 et
(2) à calibrer le générateur synthétique.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from omegaconf import DictConfig

from happycity.data.layers import UrbanLayers
from happycity.data.network import graph_edges
from happycity.data.raster import TileRaster
from happycity.metrics import TileAccessContext

PEDESTRIAN_HIGHWAYS = {"footway", "pedestrian", "path", "steps", "living_street"}
SIDEWALK_VALUES = {"both", "left", "right", "yes", "separate"}


def _in_zone(gdf: gpd.GeoDataFrame, layers: UrbanLayers) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    return gdf[gdf.geometry.representative_point().within(layers.zone)]


def _has(value: Any, accepted: set[str]) -> bool:
    values = value if isinstance(value, list) else [value]
    return any(str(v) in accepted for v in values)


def walk_edges(layers: UrbanLayers) -> gpd.GeoDataFrame:
    """Tronçons piétons non orientés (un seul sens conservé par paire de nœuds)."""
    return graph_edges(layers.graph)


def zone_stats(layers: UrbanLayers) -> dict[str, Any]:
    """Indicateurs de couverture d'une zone (bâti, réseau, espaces verts, parcelles…)."""
    area_m2 = layers.zone.area
    km2 = area_m2 / 1e6
    buildings = _in_zone(layers.buildings, layers)
    residential = buildings[buildings["residential"].astype(bool)]
    edges = walk_edges(layers)
    edges_in = edges.clip(layers.zone)
    walk_km = float(edges_in.length.sum()) / 1000
    dedicated = edges_in[[_has(h, PEDESTRIAN_HIGHWAYS) for h in edges_in.get("highway", [])]]
    sidewalk_flags = [
        _has(s, SIDEWALK_VALUES) or _has(f, {"sidewalk"})
        for s, f in zip(
            edges_in.get("sidewalk", pd.Series([None] * len(edges_in))),
            edges_in.get("footway", pd.Series([None] * len(edges_in))),
            strict=True,
        )
    ]
    sidewalk = edges_in[sidewalk_flags]
    green = layers.green[layers.green.intersects(layers.zone)]
    green_area = float(green.clip(layers.zone).union_all().area) if not green.empty else 0.0
    with_entrance = set(layers.entrances["green_id"])
    entrance_methods = layers.entrances[layers.entrances["green_id"].isin(green["green_id"])]
    excluded = layers.green_excluded[layers.green_excluded.intersects(layers.zone)]
    candidates = layers.candidates[layers.candidates.intersects(layers.zone)]
    origins = layers.origins[layers.origins.within(layers.zone)]
    rules = residential["rule"].value_counts()
    return {
        "zone": layers.name,
        "source": layers.source,
        "area_km2": round(km2, 3),
        "n_buildings": len(buildings),
        "built_ratio": round(float(layers.buildings.clip(layers.zone).area.sum()) / area_m2, 3),
        "residential_share": round(len(residential) / max(len(buildings), 1), 3),
        "res_explicit": int(rules.get("explicit", 0)),
        "res_landuse": int(rules.get("landuse", 0)),
        "res_default": int(rules.get("default", 0)),
        "walk_km": round(walk_km, 1),
        "walk_density_km_per_km2": round(walk_km / km2, 1),
        "pedestrian_share": round(float(dedicated.length.sum()) / max(walk_km * 1000, 1), 3),
        "sidewalk_tag_share": round(float(sidewalk.length.sum()) / max(walk_km * 1000, 1), 3),
        "n_green": len(green),
        "green_ha": round(green_area / 1e4, 2),
        "V": round(green_area / area_m2, 4),
        "green_without_entrance": int((~green["green_id"].isin(with_entrance)).sum()),
        "entrances_osm": int((entrance_methods["method"] == "osm").sum()),
        "entrances_boundary": int((entrance_methods["method"] == "boundary").sum()),
        "n_green_excluded": len(excluded),
        "green_excluded_reasons": "; ".join(
            f"{k} ×{v}" for k, v in excluded["reason"].value_counts().items()
        ),
        "n_candidates": len(candidates),
        "candidates_ha": round(float(candidates.clip(layers.zone).area.sum()) / 1e4, 2),
        "candidate_rules": "; ".join(
            f"{k} ×{v}" for k, v in candidates["rule"].value_counts().items()
        ),
        "n_origins": len(origins),
        "origins_dropped": int(layers.meta.get("origins_dropped", 0)),
        "n_obstacles": int(layers.obstacles.intersects(layers.zone).sum()),
    }


def coverage_flags(stats: dict[str, Any], cfg: DictConfig) -> list[str]:
    """Problèmes bloquants d'une zone au regard des seuils ``data.coverage``."""
    flags = []
    if stats["built_ratio"] < cfg.min_built_ratio:
        flags.append("faible emprise bâtie")
    if stats["n_origins"] < cfg.min_origins:
        flags.append("trop peu d'origines")
    if stats["n_candidates"] < cfg.min_candidates:
        flags.append("trop peu de parcelles candidates")
    if stats["walk_density_km_per_km2"] < cfg.min_walk_density_km_per_km2:
        flags.append("réseau piéton lacunaire")
    return flags


def coverage_table(zones: Sequence[UrbanLayers], cfg: DictConfig) -> pd.DataFrame:
    """Tableau de couverture (une ligne par zone) avec verdict."""
    rows = []
    for layers in zones:
        stats = zone_stats(layers)
        flags = coverage_flags(stats, cfg)
        stats["flags"] = ", ".join(flags)
        stats["usable"] = not flags
        rows.append(stats)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures

COLORS = {
    "residential": "#d9a66c",
    "other": "#8c96a8",
    "green": "#3f9a4a",
    "green_excluded": "#b9dcb0",
    "candidates": "#f28e2b",
    "obstacles": "#5b6b7a",
    "network": "#333333",
    "entrances": "#d62728",
}


def plot_layers(layers: UrbanLayers, path: Path, title: str | None = None) -> None:
    """Carte de contrôle visuel d'une zone (zone en pointillés, marge de contexte autour)."""
    fig, ax = plt.subplots(figsize=(9, 9))
    edges = walk_edges(layers)
    if not layers.obstacles.empty:
        layers.obstacles.plot(ax=ax, color=COLORS["obstacles"], alpha=0.6)
    if not layers.green_excluded.empty:
        layers.green_excluded.plot(ax=ax, color=COLORS["green_excluded"], hatch="//")
    if not layers.green.empty:
        layers.green.plot(ax=ax, color=COLORS["green"])
    if not layers.candidates.empty:
        layers.candidates.plot(ax=ax, color=COLORS["candidates"], alpha=0.85)
    res = layers.buildings["residential"].astype(bool)
    if res.any():
        layers.buildings[res].plot(ax=ax, color=COLORS["residential"], linewidth=0)
    if (~res).any():
        layers.buildings[~res].plot(ax=ax, color=COLORS["other"], linewidth=0)
    edges.plot(ax=ax, color=COLORS["network"], linewidth=0.3)
    if not layers.entrances.empty:
        layers.entrances.plot(ax=ax, color=COLORS["entrances"], markersize=3)
    gpd.GeoSeries([layers.zone.exterior], crs=layers.crs).plot(
        ax=ax, color="black", linestyle="--", linewidth=1.5
    )
    frame = layers.zone.buffer(float(layers.meta.get("margin_m", 500)) + 50).bounds
    ax.set_xlim(frame[0], frame[2])
    ax.set_ylim(frame[1], frame[3])
    ax.set_title(title or f"{layers.name} ({layers.source})")
    ax.set_axis_off()
    handles = [Patch(color=c) for c in COLORS.values()]
    ax.legend(handles, list(COLORS), loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_overview(zones: Sequence[UrbanLayers], splits: dict[str, str], path: Path) -> None:
    """Vue d'ensemble des zones réelles (placement, split, distances entre zones)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = {"train": "#4e79a7", "val": "#f28e2b", "test": "#e15759"}
    for layers in zones:
        walk_edges(layers).plot(ax=ax, color="#bbbbbb", linewidth=0.2)
    for layers in zones:
        split = splits.get(layers.name, "?")
        gpd.GeoSeries([layers.zone.exterior], crs=layers.crs).plot(
            ax=ax, color=palette.get(split, "black"), linewidth=2
        )
        c = layers.zone.centroid
        ax.annotate(
            f"{layers.name}\n({split})", (c.x, c.y), ha="center", fontsize=10, weight="bold"
        )
    ax.set_title("Zones d'étude réelles (EPSG:32629)")
    ax.set_aspect("equal")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def min_gaps(zones: Sequence[UrbanLayers], splits: dict[str, str]) -> pd.DataFrame:
    """Distance minimale entre zones, et respect du tampon entre splits différents."""
    rows = []
    for i, a in enumerate(zones):
        for b in zones[i + 1 :]:
            rows.append(
                {
                    "zone_a": a.name,
                    "zone_b": b.name,
                    "split_a": splits.get(a.name),
                    "split_b": splits.get(b.name),
                    "gap_m": round(a.zone.distance(b.zone), 1),
                }
            )
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Tableau Markdown sans dépendance à ``tabulate``."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)


def composite(tile: TileRaster) -> np.ndarray:
    """Image RVB d'une tuile rastérisée (mêmes couleurs que les cartes vectorielles)."""
    img = np.ones((*tile.candidate_ids.shape, 3), dtype=np.float32)
    layers = [
        ("obstacles", COLORS["obstacles"]),
        ("public_green", COLORS["green"]),
        ("other_buildings", COLORS["other"]),
        ("residential_buildings", COLORS["residential"]),
        ("walk_network", COLORS["network"]),
    ]
    for name, color in layers:
        if name in tile.channel_names:
            alpha = np.clip(tile.channel(name), 0, 1)[..., None]
            img = img * (1 - alpha) + alpha * _rgb(color)
    cand = tile.candidate_mask[..., None].astype(np.float32) * 0.85
    return np.asarray(img * (1 - cand) + cand * _rgb(COLORS["candidates"]), dtype=np.float32)


def plot_tiles(tiles: Sequence[TileRaster], zones: Sequence[UrbanLayers], path: Path) -> None:
    """Planche de validation : raster, raster + contours vectoriels, R, origines."""
    fig, axes = plt.subplots(len(tiles), 4, figsize=(16, 4 * len(tiles)), squeeze=False)
    for row, (tile, layers) in zip(axes, zip(tiles, zones, strict=True), strict=True):
        spec = tile.spec
        extent = (spec.xmin, spec.xmin + spec.size_m, spec.ymin, spec.ymin + spec.size_m)
        img = composite(tile)
        row[0].imshow(img, interpolation="nearest")
        row[0].set_title(f"{spec.tile_id} ({spec.split})", fontsize=9)
        row[1].imshow(img, extent=extent, interpolation="nearest")
        geom = spec.geometry
        for gdf, color in (
            (layers.buildings, "black"),
            (layers.green, "darkgreen"),
            (layers.candidates, "red"),
        ):
            clipped = gdf.clip(geom) if not gdf.empty else gdf
            if not clipped.empty:
                clipped.boundary.plot(ax=row[1], color=color, linewidth=0.3)
        row[1].set_xlim(extent[0], extent[1])
        row[1].set_ylim(extent[2], extent[3])
        row[1].set_title("raster + contours vectoriels", fontsize=9)
        ids = np.ma.masked_less(tile.candidate_ids, 0)
        row[2].imshow(tile.channel("walk_network"), cmap="Greys", vmin=0, vmax=3)
        row[2].imshow(ids, cmap="tab20", interpolation="nearest")
        n_adj = int(tile.candidates["adjacent"].sum())
        row[2].set_title(f"R : {len(tile.candidates)} parcelles ({n_adj} adjacentes)", fontsize=9)
        row[3].imshow(np.log1p(tile.channel("origins")), cmap="magma_r")
        row[3].set_title(f"origines : {len(tile.origins)}", fontsize=9)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_access(ctx: TileAccessContext, served: np.ndarray, title: str, path: Path) -> None:
    """Contrôle visuel de l'A400 : origines desservies / non desservies, points d'accès."""
    walk = ctx.walk
    fig, ax = plt.subplots(figsize=(8, 8))
    ymax = ctx.ymin + ctx.size_px * ctx.m_per_px
    xmax = ctx.xmin + ctx.size_px * ctx.m_per_px
    green = np.ma.masked_less_equal(ctx.green_fraction, 0)
    ax.imshow(green, extent=(ctx.xmin, xmax, ctx.ymin, ymax), cmap="Greens", vmin=0, vmax=1.2)
    for k in range(len(walk.u)):
        line = walk.edge_coords[walk.edge_offsets[k] : walk.edge_offsets[k + 1]]
        ax.plot(line[:, 0], line[:, 1], color="#999999", linewidth=0.4)
    pts = ctx.existing
    a, b = walk.node_xy[pts.u], walk.node_xy[pts.v]
    frac = np.divide(pts.t, pts.length, out=np.zeros_like(pts.t), where=pts.length > 0)[:, None]
    xy = a + (b - a) * frac  # position approchée sur le tronçon
    ax.scatter(xy[:, 0], xy[:, 1], s=6, color=COLORS["green"], marker="s", label="accès parc")
    res = ctx.m_per_px
    ox = ctx.xmin + (ctx.origin_cols + 0.5) * res
    oy = ymax - (ctx.origin_rows + 0.5) * res
    ax.scatter(ox[served], oy[served], s=4, color="#1f77b4", label="origine desservie")
    ax.scatter(ox[~served], oy[~served], s=4, color="#d62728", label="origine non desservie")
    ax.plot([ctx.xmin, xmax, xmax, ctx.xmin, ctx.xmin], [ctx.ymin, ctx.ymin, ymax, ymax, ctx.ymin],
            "k--", linewidth=1)  # fmt: skip
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=7, markerscale=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pairs(
    tile: TileRaster, masks: Sequence[np.ndarray], titles: Sequence[str], path: Path
) -> None:
    """Planche « carte initiale, parcelles, propositions, différence » d'une tuile."""
    base = composite(tile)
    n = len(masks)
    extra = 3 if n >= 2 else 2
    fig, axes = plt.subplots(1, n + extra, figsize=(4 * (n + extra), 4.3))
    axes[0].imshow(base, interpolation="nearest")
    axes[0].set_title(f"{tile.spec.tile_id} : état initial", fontsize=9)
    axes[1].imshow(np.ma.masked_less(tile.candidate_ids, 0), cmap="tab20", interpolation="nearest")
    axes[1].imshow(tile.channel("walk_network"), cmap="Greys", alpha=0.3)
    axes[1].set_title("parcelles candidates R", fontsize=9)
    green = _rgb(COLORS["green"])
    for ax, mask, title in zip(axes[2 : 2 + n], masks, titles, strict=True):
        img = base.copy()
        img[mask] = green
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=9)
    if n >= 2:
        diff = np.zeros((*masks[0].shape, 3), dtype=np.float32) + 1
        diff[masks[0] & ~masks[1]] = (0.12, 0.47, 0.71)
        diff[masks[1] & ~masks[0]] = (0.84, 0.15, 0.16)
        diff[masks[0] & masks[1]] = (0.2, 0.2, 0.2)
        axes[-1].imshow(diff, interpolation="nearest")
        axes[-1].set_title("différence (bleu : 1 seule, rouge : 2 seule)", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
