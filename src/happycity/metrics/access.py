"""Accessibilité A400 : part des origines à ≤ 400 m d'un espace vert par le réseau piéton.

A400 = (1/N) Σ_i 1[ min_{p∈P} d_G(r_i, p) ≤ 400 m ]

* ``d_G`` : plus court chemin sur le graphe piéton FIXE (Dijkstra multi-sources depuis
  les points d'accès des espaces verts, arrêt à 400 m) — jamais à vol d'oiseau ;
* ``r_i`` : nœud du graphe auquel l'origine résidentielle est rattachée ;
* ``P`` : points d'accès. Un point d'accès est soit un nœud (entrée OSM explicite), soit
  un point sur un tronçon (le plus proche du contour, à ≤ ``reach_m``) : sa distance aux
  deux extrémités du tronçon est prise en compte.

C'est une proportion d'origines (bâtiments), pas une proportion de population.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import networkx as nx
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

#: Poids minimal : scipy ignore les arêtes de poids nul.
_EPS = 1e-6
#: Tolérance de comparaison au seuil (absorbe _EPS) : 0,1 mm.
_TOL = 1e-4


@dataclass
class WalkGraph:
    """Graphe piéton non orienté, compact (tableaux) et sérialisable.

    ``edge_coords[edge_offsets[k]:edge_offsets[k+1]]`` est la polyligne du tronçon ``k``,
    orientée de ``u[k]`` vers ``v[k]``.
    """

    node_ids: np.ndarray  # (n,) int64 — identifiants d'origine
    node_xy: np.ndarray  # (n, 2) float64
    u: np.ndarray  # (m,) int64 — indices de nœuds
    v: np.ndarray  # (m,) int64
    length: np.ndarray  # (m,) float64, mètres
    edge_coords: np.ndarray  # (k, 2) float64
    edge_offsets: np.ndarray  # (m + 1,) int64

    @property
    def n_nodes(self) -> int:
        """Nombre de nœuds."""
        return len(self.node_ids)

    def edge_line(self, k: int) -> LineString:
        """Géométrie du tronçon ``k``."""
        return LineString(self.edge_coords[self.edge_offsets[k] : self.edge_offsets[k + 1]])

    def edge_index(self) -> tuple[list[LineString], shapely.STRtree]:
        """Géométries des tronçons et index spatial (calculés une fois, puis mis en cache)."""
        cached = self.__dict__.get("_edge_index")
        if cached is None:
            lines = [self.edge_line(k) for k in range(len(self.u))]
            cached = (lines, shapely.STRtree(lines))
            self.__dict__["_edge_index"] = cached
        return cached

    def node_index(self, ids: Iterable[int]) -> np.ndarray:
        """Indices internes d'identifiants de nœuds (−1 si absents)."""
        lookup = {int(n): i for i, n in enumerate(self.node_ids)}
        return np.array([lookup.get(int(n), -1) for n in ids], dtype=np.int64)

    @classmethod
    def from_networkx(
        cls, graph: nx.MultiDiGraph | nx.Graph, nodes: Iterable[int] | None = None
    ) -> WalkGraph:
        """Construit le graphe (éventuellement restreint à ``nodes``) depuis NetworkX.

        Les nœuds doivent porter ``x``, ``y`` ; les arêtes ``length`` (et ``geometry``
        facultative). Les deux sens d'une même arête sont fusionnés.
        """
        keep = list(graph.nodes) if nodes is None else [n for n in nodes if n in graph]
        index = {n: i for i, n in enumerate(keep)}
        xy = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in keep], dtype=float)
        seen: set[tuple[int, int, float]] = set()
        us, vs, lengths, coords, offsets = [], [], [], [], [0]
        for a, b, data in graph.edges(data=True):
            if a not in index or b not in index or a == b:
                continue
            i, j = index[a], index[b]
            key = (min(i, j), max(i, j), round(float(data["length"]), 3))
            if key in seen:
                continue
            seen.add(key)
            geom = data.get("geometry")
            line = np.asarray(geom.coords) if geom is not None else np.array([xy[i], xy[j]])
            if np.linalg.norm(line[0] - xy[i]) > np.linalg.norm(line[-1] - xy[i]):
                line = line[::-1]  # orientée de u vers v
            us.append(i)
            vs.append(j)
            lengths.append(float(data["length"]))
            coords.append(line[:, :2])
            offsets.append(offsets[-1] + len(line))
        return cls(
            node_ids=np.array(keep, dtype=np.int64),
            node_xy=xy.reshape(-1, 2),
            u=np.array(us, dtype=np.int64),
            v=np.array(vs, dtype=np.int64),
            length=np.array(lengths, dtype=float),
            edge_coords=np.concatenate(coords) if coords else np.zeros((0, 2)),
            edge_offsets=np.array(offsets, dtype=np.int64),
        )


@dataclass
class AccessPoints:
    """Points d'accès : distance ``t`` depuis ``u`` le long d'un tronçon de longueur ``L``.

    Un nœud seul est représenté par ``u == v`` et ``t == L == 0``.
    """

    u: np.ndarray
    v: np.ndarray
    t: np.ndarray
    length: np.ndarray

    @classmethod
    def empty(cls) -> AccessPoints:
        """Aucun point d'accès."""
        z = np.zeros(0)
        return cls(z.astype(np.int64), z.astype(np.int64), z, z)

    @classmethod
    def at_nodes(cls, nodes: Sequence[int] | np.ndarray) -> AccessPoints:
        """Points d'accès situés sur des nœuds (indices internes)."""
        idx = np.asarray(nodes, dtype=np.int64)
        z = np.zeros(len(idx))
        return cls(idx, idx.copy(), z, z.copy())

    def __len__(self) -> int:
        return len(self.u)

    def concat(self, other: AccessPoints) -> AccessPoints:
        """Union de deux ensembles."""
        return AccessPoints(
            np.concatenate([self.u, other.u]),
            np.concatenate([self.v, other.v]),
            np.concatenate([self.t, other.t]),
            np.concatenate([self.length, other.length]),
        )


def access_points_for(
    geoms: Sequence[BaseGeometry], walk: WalkGraph, reach_m: float
) -> AccessPoints:
    """Points d'accès d'espaces verts : point le plus proche de chaque tronçon à ≤ ``reach_m``."""
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms or len(walk.u) == 0:
        return AccessPoints.empty()
    lines, tree = walk.edge_index()
    us, vs, ts, ls = [], [], [], []
    for geom in geoms:
        for k in tree.query(geom, predicate="dwithin", distance=reach_m):
            line = lines[int(k)]
            on_line = nearest_points(geom, line)[1]
            t = float(line.project(on_line)) / max(line.length, _EPS) * walk.length[k]
            us.append(walk.u[k])
            vs.append(walk.v[k])
            ts.append(t)
            ls.append(walk.length[k])
    if not us:
        return AccessPoints.empty()
    return AccessPoints(
        np.array(us, dtype=np.int64), np.array(vs, dtype=np.int64), np.array(ts), np.array(ls)
    )


def _min_edges(rows: np.ndarray, cols: np.ndarray, w: np.ndarray, n: int) -> coo_matrix:
    """Matrice d'adjacence en gardant le poids MINIMAL des arêtes parallèles."""
    a, b = np.minimum(rows, cols), np.maximum(rows, cols)
    order = np.lexsort((w, b, a))
    a, b, w = a[order], b[order], w[order]
    first = np.ones(len(a), dtype=bool)
    first[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    a, b, w = a[first], b[first], np.maximum(w[first], _EPS)
    return coo_matrix((w, (a, b)), shape=(n, n))


def network_distances(walk: WalkGraph, points: AccessPoints, cutoff_m: float) -> np.ndarray:
    """Distance réseau de chaque nœud au point d'accès le plus proche (``inf`` au-delà du seuil).

    Dijkstra depuis une source virtuelle reliée à chaque point d'accès : vers ``u`` avec le
    poids ``t``, vers ``v`` avec ``L − t``.
    """
    n = walk.n_nodes
    if len(points) == 0 or n == 0:
        return np.full(n, np.inf)
    source = n
    rows = np.concatenate([walk.u, np.full(len(points), source), np.full(len(points), source)])
    cols = np.concatenate([walk.v, points.u, points.v])
    weights = np.concatenate([walk.length, points.t, points.length - points.t])
    matrix = _min_edges(rows, cols, weights, n + 1).tocsr()
    dist = dijkstra(matrix, directed=False, indices=source, limit=cutoff_m + _TOL)
    return np.asarray(dist[:n], dtype=float)


def distances_from_nodes(walk: WalkGraph, nodes: np.ndarray, cutoff_m: float) -> np.ndarray:
    """Distances réseau depuis chaque nœud de ``nodes`` vers tous les nœuds.

    Returns:
        ``(len(nodes), n_nodes)``, ``inf`` au-delà de ``cutoff_m``.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    n = walk.n_nodes
    if len(nodes) == 0 or n == 0:
        return np.full((len(nodes), n), np.inf)
    matrix = _min_edges(walk.u, walk.v, walk.length, n).tocsr()
    dist = dijkstra(matrix, directed=False, indices=nodes, limit=cutoff_m + _TOL)
    return np.atleast_2d(np.asarray(dist, dtype=float))


@dataclass
class AccessResult:
    """Résultat de l'accessibilité pour un ensemble d'origines."""

    a400: float  # NaN s'il n'y a aucune origine
    served: np.ndarray  # (N,) bool
    distance_m: np.ndarray  # (N,) float, inf si > seuil

    @property
    def n_origins(self) -> int:
        """Nombre d'origines."""
        return len(self.served)


def accessibility(
    walk: WalkGraph, origin_nodes: np.ndarray, points: AccessPoints, radius_m: float = 400.0
) -> AccessResult:
    """A400 des origines (indices internes de nœuds) vers les points d'accès."""
    dist_nodes = network_distances(walk, points, radius_m)
    origin_nodes = np.asarray(origin_nodes, dtype=np.int64)
    dist = np.where(origin_nodes >= 0, dist_nodes[np.clip(origin_nodes, 0, None)], np.inf)
    dist = dist if len(dist) else np.zeros(0)
    served = dist <= radius_m + _TOL
    a400 = float(served.mean()) if len(served) else float("nan")
    return AccessResult(a400=a400, served=served, distance_m=dist)
