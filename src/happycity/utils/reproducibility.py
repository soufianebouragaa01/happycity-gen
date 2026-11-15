"""Graines aléatoires et traçabilité (hash Git) des exécutions."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Fixe les graines de ``random``, NumPy et PyTorch (si installé).

    Args:
        seed: graine commune.
        deterministic: active les algorithmes déterministes de cuDNN (plus lent).
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - graine globale pour les bibliothèques tierces
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch fait partie des dépendances
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def git_commit_hash(repo_dir: Path | None = None) -> str | None:
    """Renvoie le hash du commit courant (suffixe ``-dirty`` si modifications), ou ``None``."""
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if status else commit
