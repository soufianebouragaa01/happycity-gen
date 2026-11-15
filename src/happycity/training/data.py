"""Chargement des paires pour torch (depuis les ``.npz`` de ``make-dataset``).

Ne dépend que de numpy et torch : utilisable sur Kaggle sans pile géographique.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from happycity.data.arrays import tile_patterns, unpack_masks

WALK_CHANNEL = 2  # index de ``walk_network`` dans INPUT_CHANNELS


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Lit un split (``.npz`` sans objets Python)."""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


class PairDataset(Dataset[dict[str, Any]]):
    """Paires (entrées de la tuile, condition, masque cible, motifs de desserte).

    Args:
        path: fichier ``<split>.npz``.
        augment: applique au hasard l'une des 8 symétries du carré. Les motifs suivent :
            ``pattern_map`` est une image transformée comme les autres.
        max_pairs: limite (premières paires, ordre stable) pour les tests et la validation.
    """

    def __init__(self, path: Path, augment: bool = False, max_pairs: int | None = None) -> None:
        arrays = load_arrays(path)
        self.split = path.stem
        self.x = arrays["x"]
        self.pattern_map = arrays["pattern_map"]
        self.patterns = [p.astype(np.float32) for p in tile_patterns(arrays)]
        self.n_origins = arrays["n_origins"]
        self.n_base = arrays["n_base"]
        self.tile_ids = arrays["tile_ids"]
        self.masks = unpack_masks(arrays)
        n = (
            len(arrays["pair_ids"])
            if max_pairs is None
            else min(max_pairs, len(arrays["pair_ids"]))
        )
        self.pair_ids = arrays["pair_ids"][:n]
        self.pair_tile = arrays["pair_tile"][:n]
        self.b_frac = arrays["b_frac"][:n]
        self.budget_px = arrays["budget_px"][:n]
        self.a_target = arrays["a_target"][:n]
        self.a_before = arrays["a_before"][:n]
        self.input_channels = tuple(str(c) for c in arrays["input_channels"])
        self.augment = augment

    @property
    def in_channels(self) -> int:
        """Nombre de canaux d'entrée (M + R)."""
        return int(self.x.shape[1])

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, i: int) -> dict[str, Any]:
        t = int(self.pair_tile[i])
        x = self.x[t].astype(np.float32) / 255.0
        pmap = self.pattern_map[t].astype(np.int64)
        y = self.masks[i].astype(np.float32)
        if self.augment:
            k = int(torch.randint(4, (1,)).item())
            flip = bool(torch.randint(2, (1,)).item())
            x, pmap, y = (np.rot90(a, k, axes=(-2, -1)) for a in (x, pmap, y))
            if flip:
                x, pmap, y = (a[..., ::-1] for a in (x, pmap, y))
        return {
            "index": i,
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.from_numpy(np.ascontiguousarray(y))[None],
            "pmap": torch.from_numpy(np.ascontiguousarray(pmap)),
            "patterns": torch.from_numpy(self.patterns[t]),
            "n_origins": int(self.n_origins[t]),
            "n_base": int(self.n_base[t]),
            "b_frac": float(self.b_frac[i]),
            "budget_px": int(self.budget_px[i]),
            "a_target": float(self.a_target[i]),
        }


def collate(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Assemble un lot ; les matrices de motifs (K, O) sont complétées par des zéros."""
    k_max = max(1, max(int(it["patterns"].shape[0]) for it in items))
    o_max = max(1, max(int(it["patterns"].shape[1]) for it in items))
    patterns = torch.zeros(len(items), k_max, o_max)
    for b, it in enumerate(items):
        k, o = it["patterns"].shape
        patterns[b, :k, :o] = it["patterns"]
    batch: dict[str, torch.Tensor] = {
        "patterns": patterns,
        "x": torch.stack([it["x"] for it in items]),
        "y": torch.stack([it["y"] for it in items]),
        "pmap": torch.stack([it["pmap"] for it in items]),
    }
    for key, dtype in (
        ("index", torch.long),
        ("n_origins", torch.float32),
        ("n_base", torch.float32),
        ("b_frac", torch.float32),
        ("budget_px", torch.float32),
        ("a_target", torch.float32),
    ):
        batch[key] = torch.tensor([it[key] for it in items], dtype=dtype)
    return batch
