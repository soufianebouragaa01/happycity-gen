"""Références non génératives : aléatoire valide, glouton, glouton stochastique."""

from happycity.baselines.placement import (
    Placement,
    PlacementProblem,
    greedy_placement,
    random_placement,
)

__all__ = ["Placement", "PlacementProblem", "greedy_placement", "random_placement"]
