"""Références non génératives : aléatoire valide, glouton, glouton stochastique."""

from happycity.baselines.placement import (
    Placement,
    PlacementProblem,
    greedy_placement,
    random_placement,
    targeted_greedy_placement,
)

__all__ = [
    "Placement",
    "PlacementProblem",
    "greedy_placement",
    "random_placement",
    "targeted_greedy_placement",
]
