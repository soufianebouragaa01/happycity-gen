"""Évaluation indépendante d'une tuile : V, A400 et E avant/après un masque d'ajouts.

Ce module ne connaît AUCUN modèle : il reçoit un :class:`TileAccessContext` (graphe
piéton fixe, origines fixes, espaces verts existants) et un masque binaire de nouvelles
surfaces vertes, quelle que soit la méthode qui l'a produit (GAN, glouton, aléatoire…).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import Polygon, shape

from happycity.metrics.access import (
    _TOL,
    AccessPoints,
    WalkGraph,
    access_points_for,
    accessibility,
    distances_from_nodes,
)
from happycity.metrics.equity import equity, sector_of
from happycity.metrics.green import added_green_m2, check_budget, green_cover


@dataclass(frozen=True)
class MetricParams:
    """Paramètres des métriques (issus de la configuration ``data``)."""

    radius_m: float = 400.0
    reach_m: float = 15.0  # un espace vert est accessible depuis un tronçon à ≤ reach_m
    sectors_per_side: int = 2
    min_origins: int = 10


@dataclass
class TileAccessContext:
    """Tout ce qu'il faut pour évaluer une tuile, indépendamment des couches d'origine."""

    tile_id: str
    xmin: float
    ymin: float
    size_px: int
    m_per_px: float
    walk: WalkGraph
    origin_nodes: np.ndarray  # (N,) indices internes de nœuds
    origin_rows: np.ndarray  # (N,)
    origin_cols: np.ndarray  # (N,)
    existing: AccessPoints  # points d'accès des espaces verts existants (contexte compris)
    green_fraction: np.ndarray  # (H, W) espaces verts comptés, dans la tuile

    def save(self, path: Path) -> None:
        """Écrit le contexte (``.npz`` sans objets Python)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        w, e = self.walk, self.existing
        np.savez_compressed(
            path,
            header=np.array([self.tile_id]),
            geo=np.array([self.xmin, self.ymin, self.m_per_px, self.size_px], dtype=float),
            node_ids=w.node_ids,
            node_xy=w.node_xy,
            u=w.u,
            v=w.v,
            length=w.length,
            edge_coords=w.edge_coords,
            edge_offsets=w.edge_offsets,
            origin_nodes=self.origin_nodes,
            origin_rows=self.origin_rows,
            origin_cols=self.origin_cols,
            ex_u=e.u,
            ex_v=e.v,
            ex_t=e.t,
            ex_length=e.length,
            green_fraction=self.green_fraction.astype(np.float16),
        )

    @classmethod
    def load(cls, path: Path) -> TileAccessContext:
        """Relit un contexte écrit par :meth:`save`."""
        with np.load(path, allow_pickle=False) as d:
            xmin, ymin, res, size = d["geo"]
            walk = WalkGraph(
                d["node_ids"], d["node_xy"], d["u"], d["v"], d["length"],
                d["edge_coords"], d["edge_offsets"],
            )  # fmt: skip
            return cls(
                tile_id=str(d["header"][0]),
                xmin=float(xmin),
                ymin=float(ymin),
                size_px=int(size),
                m_per_px=float(res),
                walk=walk,
                origin_nodes=d["origin_nodes"],
                origin_rows=d["origin_rows"],
                origin_cols=d["origin_cols"],
                existing=AccessPoints(d["ex_u"], d["ex_v"], d["ex_t"], d["ex_length"]),
                green_fraction=d["green_fraction"].astype(np.float32),
            )


@dataclass(frozen=True)
class TileScores:
    """Indicateurs d'une tuile dans un état donné (avant ou après)."""

    V: float
    A400: float
    E: float
    n_origins: int
    n_served: int
    n_valid_sectors: int
    sector_a400: tuple[float, ...]
    green_m2: float
    delta_green_m2: float

    def as_dict(self) -> dict[str, Any]:
        """Dictionnaire (tableaux de résultats)."""
        return asdict(self)


def mask_polygons(mask: np.ndarray, ctx: TileAccessContext) -> list[Polygon]:
    """Polygones (coordonnées métriques) des pixels ``True`` d'un masque."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    transform = from_origin(
        ctx.xmin, ctx.ymin + ctx.size_px * ctx.m_per_px, ctx.m_per_px, ctx.m_per_px
    )
    polygons = []
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        poly = shape(geom)
        if value == 1 and isinstance(poly, Polygon):
            polygons.append(poly)
    return polygons


def evaluate_tile(
    ctx: TileAccessContext,
    added: np.ndarray | None = None,
    *,
    params: MetricParams = MetricParams(),
    budget_m2: float | None = None,
) -> TileScores:
    """V, A400 et E de la tuile, avant (``added=None``) ou après ajout de surfaces vertes.

    Raises:
        BudgetExceededError: si ``budget_m2`` est donné et que ΔA_vert > b.
    """
    before = green_cover(ctx.green_fraction, ctx.m_per_px)
    after = green_cover(ctx.green_fraction, ctx.m_per_px, added)
    delta = added_green_m2(before, after)
    if budget_m2 is not None:
        check_budget(delta, budget_m2)
    points = ctx.existing
    if added is not None:
        new = access_points_for(mask_polygons(added, ctx), ctx.walk, params.reach_m)
        points = points.concat(new)
    acc = accessibility(ctx.walk, ctx.origin_nodes, points, params.radius_m)
    k = params.sectors_per_side
    sectors = sector_of(ctx.origin_rows, ctx.origin_cols, ctx.size_px, k)
    eq = equity(acc.served, sectors, k * k, params.min_origins)
    return TileScores(
        V=after.V,
        A400=acc.a400,
        E=eq.E,
        n_origins=acc.n_origins,
        n_served=int(acc.served.sum()),
        n_valid_sectors=eq.n_valid,
        sector_a400=eq.sector_a400,
        green_m2=after.green_m2,
        delta_green_m2=delta,
    )


def served_origins(ctx: TileAccessContext, params: MetricParams = MetricParams()) -> np.ndarray:
    """Masque des origines desservies dans l'état existant."""
    return accessibility(ctx.walk, ctx.origin_nodes, ctx.existing, params.radius_m).served


def unserved_raster(ctx: TileAccessContext, served: np.ndarray) -> np.ndarray:
    """Nombre d'origines NON desservies par pixel (canal ``unserved_origins``)."""
    out = np.zeros((ctx.size_px, ctx.size_px), dtype=np.float32)
    keep = ~np.asarray(served, dtype=bool)
    np.add.at(out, (ctx.origin_rows[keep], ctx.origin_cols[keep]), 1.0)
    return out


def pixel_service(
    ctx: TileAccessContext,
    rows: np.ndarray,
    cols: np.ndarray,
    origins: np.ndarray,
    params: MetricParams = MetricParams(),
) -> np.ndarray:
    """Origines que desservirait CHAQUE pixel, pris seul, s'il devenait espace vert.

    Mêmes conventions que :func:`evaluate_tile` (points d'accès sur les tronçons à
    ≤ ``reach_m``, distance réseau ≤ ``radius_m``), appliquées pixel par pixel. Pour un
    masque binaire, l'union des lignes de ses pixels approche l'ensemble des origines
    desservies par le masque : elle peut le dépasser légèrement, car chaque pixel apporte
    son propre point d'accès sur un tronçon, alors qu'un polygone n'en apporte qu'un par
    tronçon (le plus proche).

    Args:
        ctx: contexte d'évaluation de la tuile.
        rows: lignes des pixels considérés.
        cols: colonnes des pixels considérés.
        origins: indices des origines (dans ``ctx.origin_nodes``) à tester.
        params: paramètres des métriques.

    Returns:
        ``(len(rows), len(origins))`` booléen.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    origins = np.asarray(origins, dtype=np.int64)
    out = np.zeros((len(rows), len(origins)), dtype=bool)
    nodes = ctx.origin_nodes[origins] if len(origins) else np.zeros(0, dtype=np.int64)
    valid = nodes >= 0
    if not len(rows) or not valid.any() or len(ctx.walk.u) == 0:
        return out
    res = ctx.m_per_px
    x0 = ctx.xmin + cols * res
    y1 = ctx.ymin + (ctx.size_px - rows) * res
    boxes = shapely.box(x0, y1 - res, x0 + res, y1)
    lines, tree = ctx.walk.edge_index()
    pix, edge = tree.query(boxes, predicate="dwithin", distance=params.reach_m)
    if not len(pix):
        return out
    line_arr = np.asarray(lines, dtype=object)[edge]
    on_line = shapely.get_point(shapely.shortest_line(boxes[pix], line_arr), 1)
    geo_len = np.maximum(shapely.length(line_arr), 1e-6)
    t = shapely.line_locate_point(line_arr, on_line) / geo_len * ctx.walk.length[edge]
    u, v, length = ctx.walk.u[edge], ctx.walk.v[edge], ctx.walk.length[edge]
    uniq, inv = np.unique(nodes[valid], return_inverse=True)
    dist = distances_from_nodes(ctx.walk, uniq, params.radius_m)  # (n_uniq, n_nodes)
    d = np.minimum(dist[:, u] + t, dist[:, v] + (length - t))  # (n_uniq, n_paires)
    served_pairs = (d <= params.radius_m + _TOL)[inv].T  # (n_paires, n_origines valides)
    sub = np.zeros((len(rows), int(valid.sum())), dtype=bool)
    np.logical_or.at(sub, pix, served_pairs)
    out[:, valid] = sub
    return out
