# Zones d'étude (limites des quartiers)

## Constat : aucune limite OSM exploitable

Vérification faite le 13/11/2026 :

| Quartier | Nominatim (`<nom>, Marrakech, Morocco`) | Overpass (`place=*`, limites admin.) |
|---|---|---|
| Guéliz | aucun polygone | nœud `place=suburb` (node/595960397) ; la limite admin. « Arrondissement de Gueliz » (rel/2799533) couvre un territoire bien plus vaste que le quartier |
| Hivernage | renvoie un **hôtel** (« Hivernage Hotel & SPA », 0,001 km²) | nœud `place=neighbourhood` « L'Hivernage » (node/7264948882) |
| Massira | renvoie une **mosquée** (0,002 km²) | nœuds « Massira2 » (node/2255736116) et « Massira 3 » (node/3942697014) ; Massira 1 absent |
| Mabrouka | aucun résultat | aucun lieu ; seul repère : pharmacie « Pharmacie Hay Mabrouka » (node/8492176038) |

La fonction `is_usable_osm_boundary` (`src/happycity/data/osm.py`) formalise ce contrôle :
seule une géométrie de classe `boundary` ou `place`, d'une surface comprise entre 0,2 et
30 km², serait acceptée. Les résultats « hôtel » et « mosquée » sont donc rejetés.

## Règle retenue : zones manuelles autour d'un repère OSM

Chaque zone est un **carré de 2048 m × 2048 m (2 × 2 tuiles)** en UTM 29N, placé autour
du repère OSM du quartier. Les zones sont stockées dans `data/boundaries/<zone>.geojson`
(WGS84, avec l'emprise UTM en propriétés) et générées par `scripts/make_boundaries.py`.

| Zone | Emprise UTM (xmin, ymin) | Repère | Remarque |
|---|---|---|---|
| gueliz | 592 800, 3 500 000 | node/595960397 | au nord de l'avenue Hassan II (gare), qui sépare Guéliz de l'Hivernage |
| hivernage | 593 000, 3 497 552 | node/7264948882 | limite nord à 400 m de Guéliz (tampon entre splits) ; zone élargie au sud |
| massira | 587 500, 3 499 400 | Massira2, Massira 3 | Massira 1 n'a pas de repère OSM |
| mabrouka | 590 300, 3 501 734 | node/8492176038 | **localisation incertaine** (un seul repère) |

Ce sont des **zones d'étude**, pas des limites administratives : elles approximent le cœur
de chaque quartier et débordent sur les tissus voisins. Le nom du quartier sert d'étiquette
pour le découpage train/val/test.

## Contraintes de découpage

- Zones de splits différents séparées d'au moins **400 m** (`data.split.border_buffer_m`) :
  vérifié par `tests/test_osm_build.py::test_zones_of_different_splits_respect_buffer`.
- Guéliz (test) et Hivernage (val) sont contigus dans la réalité : on sacrifie la bande de
  400 m qui les sépare (autour de l'avenue Hassan II) pour respecter ce tampon.

## À valider

La position de **Mabrouka** repose sur un seul point d'intérêt. Merci de vérifier sur
`data/interim/coverage/overview.png` et `zone_mabrouka.png`. Pour corriger, modifier
`ZONES` dans `scripts/make_boundaries.py`, relancer le script puis
`happycity extract -d mabrouka --force`.
