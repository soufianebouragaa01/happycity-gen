"""Figures de l'évaluation (PNG statiques pour le rapport et la documentation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from happycity.evaluation.report import _per_run

# Palette de référence (thème clair) : rôles fixes, jamais réordonnés.
SERIES = {"atteignable": "#2a78d6", "non atteignable": "#eb6834"}
SINGLE = "#2a78d6"
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_abs_error(pl: pd.DataFrame, splits: list[str], path: Path) -> None:
    """|â − a| par méthode et type de cible ; barres d'erreur = écart-type entre graines."""
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 4.6), squeeze=False,
                             facecolor=SURFACE)  # fmt: skip
    for ax, split in zip(axes[0], splits, strict=True):
        d = pl[(pl["split"] == split) & pl["deployed"]]
        labels = list(dict.fromkeys(_per_run(d)["label"]))
        y = np.arange(len(labels))
        for k, (name, flag) in enumerate((("atteignable", "oui"), ("non atteignable", "non"))):
            r = _per_run(d[d["atteignable_glouton"] == flag])
            stats = r.groupby("label")["abs_err"].agg(["mean", "std"]).reindex(labels)
            offset = (k - 0.5) * 0.36
            err = {"ecolor": MUTED, "lw": 1, "capsize": 2}
            std = stats["std"].fillna(0).to_numpy()
            mean = stats["mean"].to_numpy()
            low = np.minimum(std, np.nan_to_num(mean))  # une erreur absolue ne descend pas sous 0
            ax.barh(y + offset, mean, height=0.32, color=SERIES[name], label=name,
                    xerr=np.vstack([low, std]), error_kw=err)  # fmt: skip
            span = np.nanmax(mean + std) if np.isfinite(mean).any() else 1.0
            for yi, v, sd in zip(y + offset, mean, std, strict=True):
                if not np.isnan(v):
                    ax.text(v + sd + 0.02 * span, yi, f"{v:.3f}", va="center", fontsize=7.5,
                            color=MUTED)  # fmt: skip
        ax.set_xlim(left=0)
        ax.set_yticks(y, labels, color=TEXT)
        ax.invert_yaxis()
        _style(ax)
        ax.set_xlabel("|â − a| (A400 exact, Dijkstra)", color=MUTED, fontsize=9)
        title = "test réel (Guéliz)" if split == "test" else "test synthétique"
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    axes[0][0].legend(frameon=False, fontsize=9, loc="lower right", title="cible (glouton)",
                      title_fontsize=8)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_condition(pl: pd.DataFrame, split: str, labels: list[str], path: Path) -> None:
    """A400 atteint en fonction de la cible a (cibles a posteriori), un panneau par méthode.

    Une graine par méthode générative (la plus petite) : un point par problème.
    """
    d = pl[(pl["split"] == split) & pl["deployed"] & (pl["target_type"] == "a posteriori")]
    labels = [lab for lab in labels if lab in set(d["label"])]
    fig, axes = plt.subplots(1, len(labels), figsize=(3.4 * len(labels), 3.6), squeeze=False,
                             sharey=True, facecolor=SURFACE)  # fmt: skip
    for ax, lab in zip(axes[0], labels, strict=True):
        g = d[d["label"] == lab]
        g = g[g["seed"] == g["seed"].min()]
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=0)
        ax.scatter(g["a_target"], g["A400"], s=9, color=SINGLE, alpha=0.55, linewidths=0)
        ax.set_title(f"{lab} : |â−a| moyen {g['abs_err'].mean():.3f}", color=TEXT, fontsize=9.5,
                     loc="left")  # fmt: skip
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xlabel("cible a", color=MUTED, fontsize=9)
    axes[0][0].set_ylabel("A400 atteint (exact)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
