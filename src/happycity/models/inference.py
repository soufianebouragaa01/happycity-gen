"""Propositions d'un générateur entraîné (utilisé par l'évaluation, la CLI et l'interface)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from happycity.models.factory import Networks, build_networks
from happycity.models.generator import UNetGenerator, condition_vector
from happycity.projection import ProjectionResult, project

WALK_CHANNEL = 2  # index de ``walk_network`` dans INPUT_CHANNELS
REGION_CHANNEL = -1  # R est le dernier canal


@dataclass
class Proposal:
    """Une proposition : probabilités, sortie brute seuillée, sortie projetée."""

    prob: np.ndarray  # (H, W)
    raw: np.ndarray  # (H, W) bool, p ≥ seuil (peut violer les contraintes)
    projection: ProjectionResult  # toujours calculée, pour mesurer les violations

    def deployed(self, projected: bool) -> np.ndarray:
        """Sortie retenue : projetée (variante C) ou brute (A, B)."""
        return self.projection.mask if projected else self.raw


def propose(
    generator: UNetGenerator,
    x: np.ndarray,
    b_frac: float,
    a_target: float,
    budget_px: int,
    *,
    n: int = 1,
    seed: int = 0,
    threshold: float = 0.5,
    adjacency_px: int = 1,
) -> list[Proposal]:
    """``n`` propositions pour une tuile (codes z tirés avec une graine fixe).

    Args:
        generator: générateur entraîné.
        x: entrées ``(C, H, W)`` dans [0, 1] (M puis R).
        b_frac: budget en fraction de la surface de la tuile.
        a_target: cible d'A400.
        budget_px: budget en pixels (pour la projection).
        n: nombre de propositions.
        seed: graine des codes z.
        threshold: seuil de décision.
        adjacency_px: adjacence au réseau exigée par la projection.
    """
    device = next(generator.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    generator.eval()
    xb = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)[None].expand(n, -1, -1, -1)
    cond = condition_vector(torch.full((n,), float(b_frac)), torch.full((n,), float(a_target))).to(
        device
    )
    with torch.no_grad():
        z = generator.sample_z(n, gen)
        prob = generator(xb, cond, z).prob[:, 0].float().cpu().numpy()
    allowed = np.asarray(x[REGION_CHANNEL]) > 0.5
    walk = np.asarray(x[WALK_CHANNEL]) > 0.5
    out = []
    for p in prob:
        proj = project(p, allowed, walk, budget_px, threshold=threshold, adjacency_px=adjacency_px)
        out.append(Proposal(prob=p, raw=p >= threshold, projection=proj))
    return out


def save_checkpoint(path: Path, nets: Networks, extra: dict[str, Any]) -> None:
    """Checkpoint : poids des trois réseaux + métadonnées (config, époque, commit…)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "generator": nets.generator.state_dict(),
        "discriminator": nets.discriminator.state_dict(),
        "encoder": nets.encoder.state_dict() if nets.encoder is not None else None,
        **extra,
    }
    torch.save(state, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[Networks, dict[str, Any]]:
    """Recharge un checkpoint écrit par :func:`save_checkpoint` (config incluse)."""
    from omegaconf import OmegaConf

    state = torch.load(path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["config"])
    nets = build_networks(cfg.model, int(state["in_channels"]))
    nets.generator.load_state_dict(state["generator"])
    nets.discriminator.load_state_dict(state["discriminator"])
    if nets.encoder is not None and state.get("encoder") is not None:
        nets.encoder.load_state_dict(state["encoder"])
    for net in (nets.generator, nets.discriminator, nets.encoder):
        if net is not None:
            net.to(device).eval()
    return nets, state
