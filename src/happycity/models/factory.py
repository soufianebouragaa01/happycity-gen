"""Construction des réseaux à partir de la configuration."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig
from torch import nn

from happycity.models.discriminator import LatentEncoder, PatchDiscriminator
from happycity.models.generator import GeneratorSpec, UNetGenerator


@dataclass
class Networks:
    """Générateur, discriminateur et encodeur latent d'une variante."""

    generator: UNetGenerator
    discriminator: PatchDiscriminator
    encoder: LatentEncoder | None


def generator_spec(model_cfg: DictConfig, in_channels: int) -> GeneratorSpec:
    """Spécification du générateur depuis ``cfg.model``."""
    return GeneratorSpec(
        in_channels=in_channels,
        cond_mode=str(model_cfg.cond_mode),
        attention=bool(model_cfg.attention),
        projection=bool(model_cfg.projection),
        ngf=int(model_cfg.ngf),
        n_downs=int(model_cfg.n_downs),
        dropout=float(model_cfg.dropout),
        z_dim=int(model_cfg.z_dim),
        z_injection=str(model_cfg.z_injection),
    )


def build_networks(model_cfg: DictConfig, in_channels: int) -> Networks:
    """Réseaux d'une variante. L'encodeur latent n'existe que si sa perte est active."""
    spec = generator_spec(model_cfg, in_channels)
    use_encoder = float(model_cfg.diversity.latent_regression_weight) > 0 and spec.z_dim > 0
    return Networks(
        generator=UNetGenerator(spec),
        discriminator=PatchDiscriminator(in_channels, int(model_cfg.ndf)),
        encoder=LatentEncoder(in_channels, spec.z_dim) if use_encoder else None,
    )


def count_parameters(module: nn.Module) -> int:
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
