# HappyCity Gen

GAN conditionnel proposant plusieurs scénarios de localisation et de forme de nouveaux
espaces verts, à budget foncier fixé, pour améliorer l'**accessibilité à 400 m** des zones
résidentielles et réduire les disparités spatiales. Étude de cas : Marrakech (Guéliz,
Hivernage, Massira, Mabrouka). Résolution 128×128 px, 8 m/px.

> ⚠️ Parcelles candidates **hypothétiques** ; les indicateurs (V, A400, E) sont des proxies
> d'accessibilité spatiale, **pas** une mesure du bien-être subjectif.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/) (installe lui-même Python ; 3.12 en local, compatible 3.11).

```bash
uv sync --all-extras          # environnement + dépendances verrouillées
uv run pre-commit install     # hooks ruff / mypy avant chaque commit
uv run happycity --help
```

## Qualité

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest                  # -m "not slow and not network" pour la CI
```

## Configuration

Toute la configuration est dans `configs/` (Hydra) et validée par un schéma typé
(`src/happycity/config.py`). Chaque commande accepte des surcharges :

```bash
uv run happycity show-config model=A                 # variante A, B ou C
uv run happycity show-config ablation=no_attention   # ablation appliquée à C
uv run happycity show-config data.source=synthetic   # bascule J1 : corpus synthétique seul
uv run happycity show-config paths=kaggle            # chemins Kaggle
```

## Pipeline

| Commande | Rôle | Phase |
|---|---|---|
| `happycity extract` | couches OSM + graphe piéton par zone (extrait Geofabrik daté) | 1 |
| `happycity synthesize --n 600` | cartes synthétiques procédurales (`source=synthetic`) | 1 |
| `happycity coverage-report` | couverture OSM, cartes de contrôle, calibrage synthétique | 1 |
| `happycity rasterize` | tuiles 128×128 + couverture des annotations par tuile | 2 |
| `happycity access` | état initial V/A400/E par tuile + canal `unserved_origins` | 3 |
| `happycity make-pairs` | paires avant/après (glouton stochastique), références aléatoire/glouton, découpage | 4 |
| `happycity make-dataset` | jeu compact d'entraînement (.npz par split, motifs de desserte pour l'A400 doux) | 5 |
| `happycity train` | entraînement d'une variante / ablation, ex. `model=C ablation=no_attention` (Kaggle : `notebooks/kaggle_train.ipynb`) | 5 |
| `happycity evaluate` | comparaisons, ablations, figures | 6 |
| `happycity demo` | interface de démonstration | 7 |

## Arborescence

```
configs/       Hydra : paths/, data/, model/{A,B,C}, ablation/, train/, eval/
data/          raw/, interim/, processed/ (hors Git) ; boundaries/ (polygones manuels, versionnés)
docs/          décisions, règles OSM, métriques, limites
notebooks/     demo.ipynb, kaggle_train.ipynb
app/           interface Streamlit (bonus)
src/happycity/ data, metrics (indépendant), baselines, projection, models, training, evaluation, cli
tests/
```

Les décisions techniques (échelle, secteurs, règles d'inclusion, découpage) sont
justifiées dans [docs/decisions.md](docs/decisions.md). Voir aussi
[docs/osm_rules.md](docs/osm_rules.md) (règles d'inclusion),
[docs/boundaries.md](docs/boundaries.md) (zones d'étude),
[docs/coverage.md](docs/coverage.md) (rapport de couverture) et
[docs/synthetic.md](docs/synthetic.md) (générateur).

`happycity extract` télécharge une fois l'extrait OSM du Maroc (245 Mo, Geofabrik) dans
`data/raw/osm/`, puis découpe les zones (environ 10 min au premier lancement).
