"""Indicateurs V, A400 et E, calculés indépendamment de tout modèle.

Règle d'architecture (vérifiée par tests/test_architecture.py) : ce paquet n'importe ni
torch, ni aucun autre module ``happycity`` que lui-même. Il reçoit des tableaux et un
graphe, et peut donc évaluer n'importe quelle proposition (GAN, glouton, aléatoire).
"""

from happycity.metrics.access import (
    AccessPoints,
    AccessResult,
    WalkGraph,
    access_points_for,
    accessibility,
    distances_from_nodes,
    network_distances,
)
from happycity.metrics.equity import EquityResult, equity, sector_of
from happycity.metrics.green import (
    BudgetExceededError,
    GreenCover,
    added_green_m2,
    check_budget,
    green_cover,
)
from happycity.metrics.scores import (
    MetricParams,
    TileAccessContext,
    TileScores,
    evaluate_tile,
    mask_polygons,
    pixel_service,
    served_origins,
    unserved_raster,
)

__all__ = [
    "AccessPoints",
    "AccessResult",
    "BudgetExceededError",
    "EquityResult",
    "GreenCover",
    "MetricParams",
    "TileAccessContext",
    "TileScores",
    "WalkGraph",
    "access_points_for",
    "accessibility",
    "added_green_m2",
    "check_budget",
    "distances_from_nodes",
    "equity",
    "evaluate_tile",
    "green_cover",
    "mask_polygons",
    "network_distances",
    "pixel_service",
    "sector_of",
    "served_origins",
    "unserved_raster",
]
