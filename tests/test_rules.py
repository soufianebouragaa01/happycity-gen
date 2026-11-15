"""Règles d'inclusion OSM (fonctions pures)."""

from __future__ import annotations

import math

import pytest
from omegaconf import DictConfig

from happycity.data import rules


def green(tags: dict[str, object], cfg: DictConfig) -> tuple[bool, str]:
    g = cfg.data.green
    return rules.green_decision(
        tags, g.include_tags, g.exclude_tags, list(g.exclude_access), g.exclude_fee
    )


def test_tag_value_handles_nan() -> None:
    assert rules.tag_value({"a": math.nan}, "a") is None
    assert rules.tag_value({}, "a") is None
    assert rules.tag_value({"a": "x"}, "a") == "x"


@pytest.mark.parametrize(
    ("tags", "included", "reason"),
    [
        ({"leisure": "park"}, True, "leisure=park"),
        ({"leisure": "garden", "access": "yes"}, True, "leisure=garden"),
        ({"leisure": "park", "fee": "yes"}, False, "fee"),
        ({"leisure": "park", "fee": "70 MAD"}, False, "fee"),
        ({"leisure": "park", "fee": "no"}, True, "leisure=park"),
        ({"leisure": "garden", "access": "private"}, False, "access=private"),
        ({"leisure": "garden", "access": "customers"}, False, "access=customers"),
        ({"leisure": "garden", "tourism": "hotel"}, False, "tag:tourism=hotel"),
        ({"leisure": "golf_course"}, False, "not_green"),
        ({"landuse": "orchard"}, False, "not_green"),
        ({"landuse": "grass"}, False, "not_green"),
    ],
)
def test_green_decision(
    tags: dict[str, object], included: bool, reason: str, cfg: DictConfig
) -> None:
    assert green(tags, cfg) == (included, reason)


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"landuse": "brownfield"}, "landuse=brownfield"),
        ({"landuse": "grass"}, "landuse=grass"),
        ({"amenity": "parking"}, "amenity=parking"),
        ({"amenity": "parking", "parking": "surface"}, "amenity=parking"),
        ({"amenity": "parking", "parking": "multi-storey"}, None),
        ({"amenity": "parking", "parking": "underground"}, None),
        ({"landuse": "residential"}, None),
        ({"leisure": "park"}, None),
    ],
)
def test_candidate_decision(tags: dict[str, object], expected: str | None, cfg: DictConfig) -> None:
    c = cfg.data.candidates
    assert rules.candidate_decision(tags, c.include_tags, list(c.parking_types)) == expected


@pytest.mark.parametrize(
    ("tags", "in_res", "in_non_res", "expected"),
    [
        ({"building": "apartments"}, False, False, (True, "explicit")),
        ({"building": "school"}, True, False, (False, "explicit")),
        ({"building": "yes", "shop": "bakery"}, True, False, (False, "tag:shop")),
        ({"building": "yes"}, False, True, (False, "landuse")),
        ({"building": "yes"}, True, False, (True, "landuse")),
        ({"building": "yes"}, False, False, (True, "default")),
    ],
)
def test_building_decision(
    tags: dict[str, object],
    in_res: bool,
    in_non_res: bool,
    expected: tuple[bool, str],
    cfg: DictConfig,
) -> None:
    b = cfg.data.buildings
    result = rules.building_decision(
        tags,
        in_residential_landuse=in_res,
        in_non_residential_area=in_non_res,
        residential_values=list(b.residential_values),
        non_residential_values=list(b.non_residential_values),
        non_residential_keys=list(b.non_residential_keys),
        default_yes=b.default_yes,
    )
    assert result == expected


def test_building_default_can_be_switched_off(cfg: DictConfig) -> None:
    b = cfg.data.buildings
    result = rules.building_decision(
        {"building": "yes"},
        in_residential_landuse=False,
        in_non_residential_area=False,
        residential_values=list(b.residential_values),
        non_residential_values=list(b.non_residential_values),
        non_residential_keys=list(b.non_residential_keys),
        default_yes="other",
    )
    assert result == (False, "default")
