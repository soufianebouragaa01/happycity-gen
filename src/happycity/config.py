"""Chargement et validation de la configuration Hydra.

Les fichiers YAML de ``configs/`` sont composés par Hydra puis fusionnés dans un schéma
typé (dataclasses) : une clé inconnue ou un type invalide lève une erreur immédiatement.
La CLI, l'interface et les notebooks passent tous par :func:`load_config`.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR_ENV = "HAPPYCITY_CONFIG_DIR"


class DataSource(StrEnum):
    """Origine des cartes utilisées (bascule J1 : ``synthetic``)."""

    real = "real"
    synthetic = "synthetic"
    mixed = "mixed"


class SplitStrategy(StrEnum):
    """Unité de découpage train/val/test."""

    district = "district"
    spatial_blocks = "spatial_blocks"


class CondMode(StrEnum):
    """Manière d'injecter le conditionnement (b, a) dans le générateur."""

    concat = "concat"
    film = "film"
    adain = "adain"


class ZInjection(StrEnum):
    """Manière d'injecter le bruit latent z."""

    adain = "adain"
    concat = "concat"
    none = "none"


# --------------------------------------------------------------------------- chemins


@dataclass
class PathsConfig:
    """Répertoires de données et d'expériences."""

    data_dir: str = "data"
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    dataset_dir: str = "data/processed/dataset"
    boundaries_dir: str = "data/boundaries"
    experiments_dir: str = "experiments"


# --------------------------------------------------------------------------- données


@dataclass
class DistrictConfig:
    """Un quartier d'étude ; ``boundary_file`` remplace la limite OSM si renseigné."""

    name: str = "???"
    osm_name: str = "???"
    boundary_file: str | None = None


@dataclass
class TileConfig:
    """Géométrie d'une tuile raster."""

    size_px: int = 128
    m_per_px: float = 8.0


@dataclass
class ExtractionConfig:
    """Source OSM (extrait Geofabrik ou Overpass) et marge d'extraction autour des zones."""

    margin_m: float = 500.0
    source: str = "pbf"
    pbf_url: str = "https://download.geofabrik.de/africa/morocco-latest.osm.pbf"
    pbf_file: str = "morocco-latest.osm.pbf"
    overpass_url: str = "https://overpass-api.de/api"
    user_agent: str = "happycity-gen/0.1 (academic research)"
    timeout_s: int = 300
    max_retries: int = 6
    retry_base_s: float = 30.0


@dataclass
class BuildingRulesConfig:
    """Classement résidentiel / autre des bâtiments OSM."""

    residential_values: list[str] = field(default_factory=list)
    non_residential_values: list[str] = field(default_factory=list)
    non_residential_keys: list[str] = field(default_factory=list)
    non_residential_landuse: list[str] = field(default_factory=list)
    default_yes: str = "residential"


@dataclass
class ObstacleRulesConfig:
    """Éléments non modifiables et non traversables."""

    tags: dict[str, list[str]] = field(default_factory=dict)
    line_tags: dict[str, list[str]] = field(default_factory=dict)
    line_buffer_m: float = 5.0


@dataclass
class OriginsConfig:
    """Rattachement des origines résidentielles au graphe piéton."""

    max_snap_m: float = 100.0


@dataclass
class CoverageConfig:
    """Seuils du verdict de couverture d'une zone."""

    min_built_ratio: float = 0.10
    min_origins: int = 200
    min_candidates: int = 5
    min_walk_density_km_per_km2: float = 10.0


@dataclass
class TypologyConfig:
    """Paramètres d'une typologie urbaine synthétique."""

    weight: float = 1.0
    block_m: list[float] = field(default_factory=lambda: [100.0, 150.0])
    jitter_m: float = 5.0
    edge_drop: float = 0.05
    lot_m: list[float] = field(default_factory=lambda: [15.0, 30.0])
    building_coverage: list[float] = field(default_factory=lambda: [0.5, 0.8])
    p_park: float = 0.03
    p_vacant: float = 0.05
    p_parking: float = 0.03
    p_other: float = 0.1
    p_obstacle: float = 0.01
    p_empty: float = 0.2


@dataclass
class RasterConfig:
    """Rastérisation des tuiles."""

    supersample: int = 4
    real_stride_m: float = 512.0
    min_origins: int = 30
    workers: int = 1


@dataclass
class GreenRulesConfig:
    """Règle d'inclusion des espaces verts comptés (publics ou accessibles)."""

    include_tags: dict[str, list[str]] = field(default_factory=dict)
    exclude_tags: dict[str, list[str]] = field(default_factory=dict)
    exclude_access: list[str] = field(default_factory=list)
    exclude_fee: bool = True
    manual_exclude: dict[str, str] = field(default_factory=dict)
    entrance_snap_m: float = 15.0
    min_area_m2: float = 100.0


@dataclass
class CandidateRulesConfig:
    """Règle explicite d'identification des parcelles candidates (hypothétiques)."""

    include_tags: dict[str, list[str]] = field(default_factory=dict)
    parking_types: list[str] = field(default_factory=list)
    exclude_tags: dict[str, list[str]] = field(default_factory=dict)
    min_area_m2: float = 500.0


@dataclass
class NetworkConfig:
    """Réseau piéton (OSMnx)."""

    network_type: str = "walk"
    adjacency_px: int = 1


@dataclass
class EquityConfig:
    """Secteurs fixes pour l'indicateur d'équité E."""

    sectors_per_side: int = 2
    min_origins: int = 10


@dataclass
class SplitConfig:
    """Découpage train/val/test par quartier ou par blocs spatiaux."""

    strategy: SplitStrategy = SplitStrategy.district
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    test: list[str] = field(default_factory=list)
    border_buffer_m: float = 400.0
    synthetic_in: list[str] = field(default_factory=lambda: ["train"])


@dataclass
class SyntheticConfig:
    """Générateur procédural (``source=synthetic``)."""

    n_maps: int = 600
    seed: int = 0
    extent_m: float = 1024.0
    margin_m: float = 500.0
    street_half_width_m: float = 5.0
    typologies: dict[str, TypologyConfig] = field(default_factory=dict)


@dataclass
class PairsConfig:
    """Paires avant/après produites par le glouton stochastique."""

    budget_fractions: list[float] = field(default_factory=list)
    solutions_per_budget: int = 3
    top_k: int = 3
    attempts: int = 8
    min_relative_gain: float = 0.8
    seed: int = 0
    random_draws: int = 10
    synthetic_val_fraction: float = 0.1
    synthetic_test_fraction: float = 0.1


@dataclass
class DataConfig:
    """Corpus, règles OSM et paramètres géométriques."""

    source: DataSource = DataSource.mixed
    city: str = "Marrakech"
    crs: str = "EPSG:32629"
    districts: list[DistrictConfig] = field(default_factory=list)
    tile: TileConfig = field(default_factory=TileConfig)
    buffer_m: float = 400.0
    access_radius_m: float = 400.0
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    channels: list[str] = field(default_factory=list)
    raster: RasterConfig = field(default_factory=RasterConfig)
    buildings: BuildingRulesConfig = field(default_factory=BuildingRulesConfig)
    green: GreenRulesConfig = field(default_factory=GreenRulesConfig)
    candidates: CandidateRulesConfig = field(default_factory=CandidateRulesConfig)
    obstacles: ObstacleRulesConfig = field(default_factory=ObstacleRulesConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    origins: OriginsConfig = field(default_factory=OriginsConfig)
    equity: EquityConfig = field(default_factory=EquityConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)


# --------------------------------------------------------------------------- modèle


@dataclass
class DiversityConfig:
    """Mécanismes anti-effondrement de modes."""

    mode_seeking_weight: float = 1.0
    latent_regression_weight: float = 0.5


@dataclass
class ModelConfig:
    """Variante du générateur (A, B, C) et ses options."""

    name: str = "C"
    cond_mode: CondMode = CondMode.film
    attention: bool = True
    projection: bool = True
    ngf: int = 32
    ndf: int = 32
    n_downs: int = 5
    dropout: float = 0.5
    z_dim: int = 8
    z_injection: ZInjection = ZInjection.adain
    threshold: float = 0.5
    diversity: DiversityConfig = field(default_factory=DiversityConfig)


@dataclass
class AblationConfig:
    """Nom de l'ablation appliquée (``full`` = aucune)."""

    name: str = "full"


# --------------------------------------------------------------------------- entraînement


@dataclass
class LossWeightsConfig:
    """Pondération des termes de perte."""

    adversarial: float = 1.0
    reconstruction: float = 100.0
    violation: float = 100.0
    condition: float = 10.0


@dataclass
class TrainConfig:
    """Hyperparamètres d'entraînement."""

    seed: int = 42
    epochs: int = 40
    batch_size: int = 32
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: list[float] = field(default_factory=lambda: [0.5, 0.999])
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    instance_noise: float = 0.1
    num_workers: int = 2
    device: str = "auto"
    amp: bool = True
    deterministic: bool = False
    log_every: int = 50
    checkpoint_every: int = 5
    max_steps: int | None = None
    augment: bool = True
    val_z: int = 3
    val_max_pairs: int | None = None
    selection_split: str = "val_synth"


@dataclass
class EvalConfig:
    """Protocole d'évaluation."""

    n_proposals: int = 2
    budget_fractions: list[float] = field(default_factory=list)
    unreachable_target_offsets: list[float] = field(default_factory=list)
    split_by_reachability: bool = True
    random_baseline_draws: int = 10
    seed: int = 123


@dataclass
class AppConfig:
    """Configuration complète d'une exécution."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    run_name: str = "???"


# --------------------------------------------------------------------------- chargement


def default_config_dir() -> Path:
    """Renvoie le dossier ``configs/`` (variable ``HAPPYCITY_CONFIG_DIR`` prioritaire)."""
    env = os.environ.get(CONFIG_DIR_ENV)
    path = Path(env) if env else Path(__file__).resolve().parents[2] / "configs"
    if not (path / "config.yaml").is_file():
        msg = (
            f"Dossier de configuration introuvable : {path}. "
            f"Installez le paquet en mode éditable ou définissez {CONFIG_DIR_ENV}."
        )
        raise FileNotFoundError(msg)
    return path


def load_config(
    overrides: Sequence[str] = (),
    config_dir: Path | None = None,
    config_name: str = "config",
) -> DictConfig:
    """Compose la configuration Hydra, la valide contre :class:`AppConfig` et la résout.

    Args:
        overrides: surcharges Hydra, par ex. ``["model=A", "data.source=synthetic"]``.
        config_dir: dossier de configuration (par défaut :func:`default_config_dir`).
        config_name: nom du fichier racine sans extension.

    Returns:
        Configuration typée, en lecture seule, interpolations résolues.
    """
    directory = (config_dir or default_config_dir()).resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(directory), job_name="happycity"):
        raw = compose(config_name=config_name, overrides=list(overrides))
    schema = OmegaConf.structured(AppConfig)
    cfg = cast(DictConfig, OmegaConf.merge(schema, raw))
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convertit la configuration en dictionnaire sérialisable (checkpoints, métadonnées)."""
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
