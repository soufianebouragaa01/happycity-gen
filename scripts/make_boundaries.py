"""Génère les polygones manuels des zones d'étude (data/boundaries/*.geojson).

OSM ne fournit aucune limite exploitable pour les quatre quartiers (voir docs/boundaries.md) :
chaque zone est un carré de 2048 m (2 × 2 tuiles) en UTM 29N, placé autour d'un point de
repère OSM. Les zones de splits différents sont séparées d'au moins 400 m.

Usage : ``uv run python scripts/make_boundaries.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import box, mapping

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "boundaries"

# (xmin, ymin) en EPSG:32629 ; côté 2048 m. Repères : identifiants OSM des nœuds.
ZONES = {
    "gueliz": {
        "xmin": 592_800,
        "ymin": 3_500_000,
        "anchor": "node/595960397 (place=suburb, Guéliz)",
        "note": "Au nord de l'avenue Hassan II (gare), qui sépare Guéliz de l'Hivernage.",
    },
    "hivernage": {
        "xmin": 593_000,
        "ymin": 3_497_552,
        "anchor": "node/7264948882 (place=neighbourhood, L'Hivernage)",
        "note": "Limite nord à 400 m de la zone Guéliz (tampon entre splits) ; élargie au sud.",
    },
    "massira": {
        "xmin": 587_500,
        "ymin": 3_499_400,
        "anchor": "node/2255736116 (Massira2), node/3942697014 (Massira 3)",
        "note": "Couvre Massira 2 et 3 ; Massira 1 n'a pas de repère OSM.",
    },
    "mabrouka": {
        "xmin": 590_300,
        "ymin": 3_501_734,
        "anchor": "node/8492176038 (amenity=pharmacy « Pharmacie Hay Mabrouka »)",
        "note": "Seul repère OSM du quartier : localisation INCERTAINE, à valider visuellement.",
    },
}
SIDE_M = 2048


def main() -> None:
    """Écrit un GeoJSON (WGS84) par zone, avec l'emprise UTM en propriétés."""
    transformer = Transformer.from_crs(32629, 4326, always_xy=True)

    def to_wgs84(coords: np.ndarray) -> np.ndarray:
        return np.column_stack(transformer.transform(coords[:, 0], coords[:, 1]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, zone in ZONES.items():
        xmin, ymin = float(zone["xmin"]), float(zone["ymin"])
        utm_box = box(xmin, ymin, xmin + SIDE_M, ymin + SIDE_M)
        feature = {
            "type": "Feature",
            "geometry": mapping(shapely.transform(utm_box, to_wgs84)),
            "properties": {
                "name": name,
                "method": "manual_square_around_osm_anchor",
                "anchor": zone["anchor"],
                "note": zone["note"],
                "utm_crs": "EPSG:32629",
                "utm_bounds": [xmin, ymin, xmin + SIDE_M, ymin + SIDE_M],
            },
        }
        collection = {"type": "FeatureCollection", "features": [feature]}
        path = OUT_DIR / f"{name}.geojson"
        path.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name} written")


if __name__ == "__main__":
    main()
