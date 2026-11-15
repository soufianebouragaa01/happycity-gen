"""Composition et validation des configurations Hydra."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, ValidationError

from happycity.config import CondMode, DataSource, default_config_dir, load_config, to_dict

CONFIG_DIR = default_config_dir()


def test_default_config_is_model_c_full() -> None:
    cfg = load_config()
    assert cfg.model.name == "C"
    assert cfg.ablation.name == "full"
    assert cfg.run_name == "C-full"
    assert cfg.data.crs == "EPSG:32629"


@pytest.mark.parametrize(
    ("variant", "cond_mode", "attention", "projection"),
    [
        ("A", CondMode.concat, False, False),
        ("B", CondMode.film, False, False),
        ("C", CondMode.film, True, True),
    ],
)
def test_variants(variant: str, cond_mode: CondMode, attention: bool, projection: bool) -> None:
    model = load_config([f"model={variant}"]).model
    assert model.name == variant
    assert model.cond_mode == cond_mode
    assert model.attention is attention
    assert model.projection is projection


def test_variants_share_everything_but_studied_options() -> None:
    """A, B, C ne diffèrent que par le conditionnement, l'attention et la projection."""
    studied = {"name", "cond_mode", "attention", "projection"}
    models = [to_dict(load_config([f"model={v}"]))["model"] for v in "ABC"]
    for key in models[0].keys() - studied:
        assert models[0][key] == models[1][key] == models[2][key], key


@pytest.mark.parametrize(
    ("ablation", "check"),
    [
        ("no_modulation", lambda m: m.cond_mode == CondMode.concat),
        ("no_attention", lambda m: m.attention is False),
        ("no_projection", lambda m: m.projection is False),
        (
            "no_diversity",
            lambda m: (
                m.diversity.mode_seeking_weight == 0 and m.diversity.latent_regression_weight == 0
            ),
        ),
    ],
)
def test_ablations_modify_model_c(ablation: str, check: object) -> None:
    cfg = load_config([f"ablation={ablation}"])
    assert cfg.model.name == "C"
    assert cfg.run_name == f"C-{ablation}"
    assert callable(check)
    assert check(cfg.model)


def test_ablation_files_match_their_name() -> None:
    names = sorted(p.stem for p in (CONFIG_DIR / "ablation").glob("*.yaml"))
    assert names == ["no_attention", "no_diversity", "no_modulation", "no_projection", "none"]


def test_switch_to_synthetic_only_by_config() -> None:
    """Bascule J1 : un seul changement de configuration."""
    assert load_config(["data.source=synthetic"]).data.source == DataSource.synthetic


def test_invalid_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        load_config(["data.source=satellite"])


def test_unknown_key_is_rejected() -> None:
    with pytest.raises((ConfigKeyError, ValidationError)):
        load_config(["+data.unknown_option=1"])


def test_scale_covers_access_radius() -> None:
    """Échelle constante : la tuile (1024 m) couvre au moins deux rayons d'accès."""
    data = load_config().data
    extent_m = data.tile.size_px * data.tile.m_per_px
    assert extent_m == pytest.approx(1024.0)
    assert extent_m >= 2 * data.access_radius_m
    assert data.buffer_m >= data.access_radius_m


def test_split_is_disjoint_and_keeps_gueliz_hivernage_out_of_train() -> None:
    split = load_config().data.split
    groups = [set(split.train), set(split.val), set(split.test)]
    assert sum(len(g) for g in groups) == len(set().union(*groups))  # disjoints
    districts = {d.name for d in load_config().data.districts}
    assert set().union(*groups) == districts
    assert list(split.synthetic_in) == ["train"]
    assert split.border_buffer_m >= 400


def test_kaggle_paths() -> None:
    paths = load_config(["paths=kaggle"]).paths
    assert paths.processed_dir.startswith("/kaggle/input/")
    assert paths.experiments_dir.startswith("/kaggle/working/")


def test_no_hardcoded_local_paths_in_configs() -> None:
    drive = re.compile(r"\b[A-Za-z]:[\\/]")
    for path in Path(CONFIG_DIR).rglob("*.yaml"):
        assert not drive.search(path.read_text(encoding="utf-8")), path


def test_to_dict_is_serializable() -> None:
    import json

    json.dumps(to_dict(load_config()))
