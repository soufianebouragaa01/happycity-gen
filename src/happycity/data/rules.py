"""Règles d'inclusion explicites, appliquées aux étiquettes (tags) OSM.

Fonctions pures (un dictionnaire de tags en entrée) : elles sont testées unitairement et
documentées dans docs/osm_rules.md. Les traitements géométriques sont dans ``osm.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

Tags = Mapping[str, Any]
TagSpec = Mapping[str, Sequence[str]]


def tag_value(tags: Tags, key: str) -> str | None:
    """Valeur d'un tag, ``None`` si absente ou NaN (colonnes pandas)."""
    value = tags.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def match_tags(tags: Tags, spec: TagSpec) -> str | None:
    """Premier ``clé=valeur`` de ``spec`` présent dans ``tags`` (``None`` sinon)."""
    for key, values in spec.items():
        value = tag_value(tags, key)
        if value is not None and value in values:
            return f"{key}={value}"
    return None


def green_decision(
    tags: Tags,
    include_tags: TagSpec,
    exclude_tags: TagSpec,
    exclude_access: Sequence[str],
    exclude_fee: bool,
) -> tuple[bool, str]:
    """Décide si un objet est un espace vert COMPTÉ (public ou explicitement accessible).

    Returns:
        ``(inclus, raison)`` ; la raison est ``"not_green"`` si l'objet n'est pas un
        espace vert candidat, sinon le tag d'inclusion ou le motif d'exclusion.
    """
    included = match_tags(tags, include_tags)
    if included is None:
        return False, "not_green"
    excluded = match_tags(tags, exclude_tags)
    if excluded is not None:
        return False, f"tag:{excluded}"
    access = tag_value(tags, "access")
    if access is not None and access in exclude_access:
        return False, f"access={access}"
    if exclude_fee and tag_value(tags, "fee") not in (None, "no"):
        return False, "fee"
    return True, included


def candidate_decision(
    tags: Tags, include_tags: TagSpec, parking_types: Sequence[str]
) -> str | None:
    """Règle des parcelles candidates (hypothétiques) ; renvoie le tag retenu ou ``None``.

    Un parking n'est retenu que s'il est en surface : ``parking`` absent ou dans
    ``parking_types`` (``multi-storey``, ``underground``… exclus).
    """
    matched = match_tags(tags, include_tags)
    if matched is None:
        return None
    if matched == "amenity=parking":
        parking = tag_value(tags, "parking")
        if parking is not None and parking not in parking_types:
            return None
    return matched


def building_decision(
    tags: Tags,
    *,
    in_residential_landuse: bool,
    in_non_residential_area: bool,
    residential_values: Sequence[str],
    non_residential_values: Sequence[str],
    non_residential_keys: Sequence[str],
    default_yes: str,
) -> tuple[bool, str]:
    """Classe un bâtiment en résidentiel ou non.

    Ordre : valeur ``building`` explicite → tags d'activité (``shop``, ``amenity``…) →
    occupation du sol englobante → hypothèse par défaut pour ``building=yes``.

    Returns:
        ``(résidentiel, règle appliquée)``.
    """
    value = tag_value(tags, "building") or "yes"
    if value in residential_values:
        return True, "explicit"
    if value in non_residential_values:
        return False, "explicit"
    for key in non_residential_keys:
        if tag_value(tags, key) is not None:
            return False, f"tag:{key}"
    if in_non_residential_area:
        return False, "landuse"
    if in_residential_landuse:
        return True, "landuse"
    return default_yes == "residential", "default"
