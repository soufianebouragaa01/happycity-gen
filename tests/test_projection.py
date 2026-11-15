"""Opérateur de projection : validité garantie, déterminisme, idempotence."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import label

from happycity.projection import compose_after, network_neighbourhood, project

H = W = 32


def scene() -> tuple[np.ndarray, np.ndarray]:
    """Rue sur la ligne 0 ; R = deux parcelles collées à la rue + une parcelle isolée."""
    walk = np.zeros((H, W))
    walk[0] = 1
    allowed = np.zeros((H, W), dtype=bool)
    allowed[1:6, 0:6] = True  # parcelle adjacente
    allowed[1:6, 20:26] = True  # parcelle adjacente
    allowed[15:20, 10:15] = True  # parcelle non adjacente (la projection doit l'écarter)
    return walk, allowed


def check_valid(mask: np.ndarray, allowed: np.ndarray, walk: np.ndarray, budget: int) -> None:
    assert not (mask & ~allowed).any()
    assert mask.sum() <= budget
    labels, n = label(mask)
    near = network_neighbourhood(walk, 1)
    for k in range(1, n + 1):
        assert (near & (labels == k)).any(), "composante détachée du réseau"


def test_everything_proposed_is_restricted_and_budgeted() -> None:
    walk, allowed = scene()
    prob = np.full((H, W), 0.9)
    res = project(prob, allowed, walk, budget_px=20)
    check_valid(res.mask, allowed, walk, 20)
    assert res.removed_outside_r == H * W - allowed.sum()
    assert res.proposed_px == H * W


def test_detached_components_are_removed() -> None:
    walk, allowed = scene()
    prob = np.zeros((H, W))
    prob[15:20, 10:15] = 0.9  # uniquement la parcelle isolée
    res = project(prob, allowed, walk, budget_px=100)
    assert res.n_px == 0
    assert res.removed_detached == 25


def test_budget_keeps_most_probable_pixels() -> None:
    walk, allowed = scene()
    prob = np.zeros((H, W))
    prob[1:6, 0:6] = 0.6
    prob[1:3, 0:6] = 0.95  # les deux lignes contre la rue sont les plus probables
    res = project(prob, allowed, walk, budget_px=12)
    assert res.n_px == 12
    assert res.mask[1:3, 0:6].all()
    assert res.removed_budget == 30 - 12


def test_truncation_can_detach_and_is_cleaned() -> None:
    """Tronquer au budget peut isoler un morceau loin de la rue : il est retiré."""
    walk, allowed = scene()
    prob = np.zeros((H, W))
    prob[1:6, 0:6] = 0.6
    prob[5, 0:6] = 0.99  # ligne la plus éloignée de la rue, la plus probable
    res = project(prob, allowed, walk, budget_px=6)
    assert res.n_px == 0  # la seule ligne gardée par le budget n'est pas adjacente
    check_valid(res.mask, allowed, walk, 6)


@pytest.mark.parametrize("seed", range(10))
def test_random_maps_always_valid_and_idempotent(seed: int) -> None:
    rng = np.random.default_rng(seed)
    walk = (rng.random((H, W)) < 0.05).astype(float)
    allowed = rng.random((H, W)) < 0.4
    prob = rng.random((H, W))
    budget = int(rng.integers(0, 200))
    res = project(prob, allowed, walk, budget)
    check_valid(res.mask, allowed, walk, budget)
    again = project(res.mask.astype(float), allowed, walk, budget)
    np.testing.assert_array_equal(again.mask, res.mask)
    np.testing.assert_array_equal(project(prob, allowed, walk, budget).mask, res.mask)


def test_compose_after_keeps_fixed_elements() -> None:
    names = ("residential_buildings", "walk_network", "public_green")
    channels = np.random.default_rng(0).random((3, H, W)).astype(np.float32)
    added = np.zeros((H, W), dtype=bool)
    added[2:4, 2:4] = True
    after = compose_after(channels, names, added)
    np.testing.assert_array_equal(after[:2], channels[:2])
    assert (after[2][added] == 1).all()
    np.testing.assert_array_equal(after[2][~added], channels[2][~added])
