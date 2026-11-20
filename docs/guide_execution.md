# Guide d'exécution : faire tourner HappyCity Gen de bout en bout

Ce guide suit l'ordre réel du projet. Chaque étape indique **ce qu'elle fait**, **la
commande**, **ce qu'elle produit** et **combien de temps elle prend**. Les durées ont été
mesurées sur la machine de développement (Windows 11, 6 processus, GTX 1650) et sur Kaggle
(GPU T4).

Toutes les commandes se lancent **depuis la racine du dépôt**. `uv` n'étant pas dans le
`PATH` sur cette machine, on peut l'appeler par son chemin complet :
`C:\Users\soufi\AppData\Roaming\Python\Python313\Scripts\uv.exe`.

```
données OSM ─┐
             ├─► tuiles 128×128 ─► accessibilité ─► paires ─► jeu compact ─► entraînement (Kaggle)
synthétique ─┘                                                                    │
                                     évaluation ◄──────────────────────────────────┤
                                     démonstration (CLI, notebook, Streamlit) ◄────┘
```

## Étape 0 : installer

```bash
git clone https://github.com/soufianebouragaa01/happycity-gen.git
cd happycity-gen
uv sync --all-extras          # Python 3.12, torch CUDA, géomatique, Streamlit, Jupyter
uv run happycity --help       # liste des commandes
uv run pytest -q              # ≈ 170 tests, ≈ 1 à 2 min
```

Sous Windows, le **premier** lancement de torch ou de GDAL peut prendre plusieurs minutes,
le temps que Smart App Control analyse les DLL. Si un test échoue sur une erreur « DLL load
failed… stratégie de contrôle d'application », relancez-le : ce blocage est intermittent.

## Étape 1 : données OpenStreetMap et synthétiques

| Commande | Ce qu'elle fait | Produit | Durée |
|---|---|---|---|
| `uv run happycity extract` | télécharge l'extrait Geofabrik du Maroc (≈ 250 Mo, une fois) et découpe les 4 quartiers | `data/raw/osm/`, `data/interim/osm/<quartier>/` | ≈ 5 min |
| `uv run happycity synthesize` | génère 600 cartes synthétiques (3 typologies) | `data/interim/synthetic/` | ≈ 6 min |
| `uv run happycity coverage-report` | tableau de couverture des annotations, figures | `data/interim/coverage/` | ≈ 1 min |

Les limites des quartiers sont versionnées dans `data/boundaries/`. Il n'y a rien à faire
pour elles.

## Étape 2 : tuiles, accessibilité, paires, jeu compact

| Commande | Ce qu'elle fait | Produit | Durée |
|---|---|---|---|
| `uv run happycity rasterize` | tuiles 128 × 128 px à 8 m/px (6 canaux + R) | `data/processed/tiles/`, `tiles.parquet` | ≈ 7 min |
| `uv run happycity access` | A400, V et E de l'état actuel (Dijkstra), canal des origines non desservies | `data/processed/context/` | ≈ 3 min |
| `uv run happycity make-pairs` | paires avant/après (cibles max **et intermédiaires**), références, découpage | `pairs.parquet`, `baselines.parquet`, `pairs_report.md` | ≈ 6 min |
| `uv run happycity make-dataset` | jeu compact pour l'entraînement (un `.npz` par split) | `data/processed/dataset/*.npz` | ≈ 1 min |

À vérifier : `data/processed/pairs_report.md` (paires par split, gains, références) et
`data/processed/dataset/dataset_report.md` (fidélité de l'A400 doux).

## Étape 3 : entraînement (Kaggle, GPU)

L'entraînement local est possible (`uv run happycity train model=C train.amp=false`), mais
la GTX 1650 met environ 6 min par époque. On entraîne donc sur Kaggle.

1. **Données** : sur kaggle.com, *Datasets → New Dataset* (ou *New Version* du dataset
   existant), déposez les 5 fichiers `data/processed/dataset/*.npz`.
2. **Notebook** : *File → Import Notebook* →
   `notebooks/kaggle_train.ipynb`. Dans *Input*, ajoutez le dataset. Dans *Settings*,
   choisissez *Accelerator* = **GPU T4** et *Internet* = **On**.
3. **Réglages** (première cellule) : `RUNS` liste les variantes, les ablations et leurs
   graines. Par défaut, il y a 13 exécutions de 40 époques.
4. **Lancement** : *Save Version → Save & Run All (Commit)*. Vous pouvez fermer le
   navigateur. Comptez ≈ 25 min par exécution, soit ≈ 5 h 30 pour les 13.
5. **Récupération** : onglet *Output* de la version, téléchargez `experiments.zip`, puis
   décompressez son **contenu** dans `experiments\`. On doit obtenir :

```
experiments\
├── seed42\   A-full-s42, B-full-s42, C-full-s42, C-no_attention-s42, …
├── seed43\   A-full-s43, B-full-s43, C-full-s43
└── seed44\   A-full-s44, B-full-s44, C-full-s44
```

Chaque exécution contient `config.yaml`, `meta.json` (commit git, GPU), `train_log.csv`,
`val_log.csv` et `checkpoints/{best,last}.pt`.

## Étape 4 : évaluation

```bash
uv run happycity evaluate            # ≈ 10 min avec 13 exécutions
```

Produit `experiments/evaluation/` :

- `report.md` : tous les tableaux ;
- `summary_test_synth.csv`, `summary_test.csv` ;
- `condition_response.csv` : **la cible a est-elle suivie ?** ;
- `figures/`.

Lecture :

- **test réel** (Guéliz) = test final, mais il discrimine peu ;
- **test synthétique** = rapporté à part ;
- E se lit toujours avec A400 ;
- « atteignable par le glouton » est une estimation par le bas.

## Étape 5 : démonstration

```bash
uv run happycity demo --tile gueliz_11 --budget 0.02          # figure + tableau dans experiments/demo/
uv run happycity demo --tile synth_00545 --budget 0.03 --target 0.85 --n 3
uv run happycity demo --ui                                    # interface : http://localhost:8501
```

Le notebook `notebooks/demo.ipynb` est livré exécuté. Pour le régénérer :
`uv run python scripts/build_demo_notebook.py`.

## Étape 6 : contrôle qualité

```bash
uv run ruff check src tests app scripts
uv run ruff format --check src tests app scripts
uv run python -m mypy
uv run pytest -q
```

La CI GitHub (`.github/workflows/ci.yml`) relance ces contrôles sous Python 3.11 et 3.12.

## Où lire les résultats

| Document | Contenu |
|---|---|
| `docs/decisions.md` | choix techniques D1 à D20 et questions ouvertes |
| `docs/coverage.md`, `docs/raster.md` | données et couverture |
| `docs/metrics.md` | définition de V, A400 et E |
| `docs/baselines_pairs.md` | références et paires |
| `docs/models.md` | modèles A/B/C, projection, diversité |
| `docs/evaluation.md` | résultats, limites, réponse à la condition |
| `docs/demo.md` | démonstration |
