"""Graines et traçabilité."""

from __future__ import annotations

import random

import numpy as np
import torch

from happycity.utils.reproducibility import git_commit_hash, seed_everything


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))  # noqa: NPY002


def test_seed_everything_is_reproducible() -> None:
    seed_everything(7)
    first = _draw()
    seed_everything(7)
    assert _draw() == first
    seed_everything(8)
    assert _draw() != first


def test_git_commit_hash_is_hex_or_none() -> None:
    value = git_commit_hash()
    if value is not None:
        assert all(c in "0123456789abcdef" for c in value.removesuffix("-dirty"))
