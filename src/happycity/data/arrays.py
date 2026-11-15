"""Lecture des tableaux du jeu compact (numpy seul : utilisable sans pile géographique)."""

from __future__ import annotations

import numpy as np


def tile_patterns(arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Découpe ``pattern_data`` en une matrice (K, O) par tuile."""
    out, start = [], 0
    for k, o in np.asarray(arrays["pattern_shape"]):
        n = int(k) * int(o)
        out.append(np.asarray(arrays["pattern_data"][start : start + n]).reshape(int(k), int(o)))
        start += n
    return out


def unpack_masks(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Masques cibles ``(P, H, W)`` booléens."""
    shape = tuple(int(s) for s in arrays["mask_shape"])
    return np.unpackbits(arrays["masks"], axis=-1)[..., : shape[-1]].astype(bool)
