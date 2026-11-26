# Journal des décisions

Les décisions techniques et leur justification. Toutes les valeurs numériques sont dans
`configs/` (aucune valeur en dur dans le code) ; ce document explique *pourquoi*.

> Rappel de périmètre : le projet mesure une **accessibilité spatiale** aux espaces verts
> publics. Il n'estime pas le bonheur subjectif et ne revendique aucun effet causal sur la
> satisfaction de vie. Les parcelles candidates sont **hypothétiques**.

## D1 — Échelle : 8 m/px, tuiles de 1024 m (`data.tile`)

- 128 px × 8 m = 1024 m ; le rayon d'accès de 400 m vaut 50 px.
- La tuile couvre plus de deux rayons d'accès : un ajout au centre sert des origines dans
  toute une moitié de tuile, et plusieurs choix de localisation sont réellement en concurrence.
- Une mesure préalable (prototype précédent) a montré qu'à 4 m/px (tuiles de 512 m) une seule
  parcelle desservait ≈ 58 % des origines : placement aléatoire ≈ glouton, la tâche était
  triviale. À 8 m/px : 11 à 23 parcelles par tuile, ≈ 22 % des origines par parcelle,
  écart glouton − aléatoire de 6 à 14 points.
- Contrepartie : un bâtiment résidentiel fait quelques pixels ; les origines sont donc des
  points (centroïdes rattachés au graphe), pas des pixels.

## D2 — Zone tampon de 400 m (`data.buffer_m`)

Le graphe piéton et les espaces verts existants sont extraits avec 400 m de marge autour de
chaque tuile ; les origines et les ajouts restent dans la tuile. Sans cela, une origine en
bord de tuile paraîtrait non desservie alors qu'un parc existe juste à côté. Le canal
`unserved_origins` de M transmet cette information au générateur, qui ne voit que la tuile.
Comme il dépend de l'A400, il est dérivé en phase 3 par le module de métriques ; la
phase 2 rastérise le canal `origins` (docs/raster.md).

## D3 — Secteurs de l'équité E (`data.equity`)

- Par tuile : 4 quadrants fixes de 512 m. Un secteur n'est retenu que s'il contient au moins
  10 origines ; s'il reste moins de 2 secteurs valides, E est indéfini (`NaN`) et compté à part.
- Par quartier (rapport) : les secteurs sont les tuiles.
- E est toujours rapporté avec A400 (une égalité à zéro accès n'est pas une réussite).

## D4 — Espaces verts comptés (`data.green`)

Inclus : `leisure=park|garden|playground` sauf `access=private|no|customers|permit` ou
`fee=yes`. Exclus : golfs, vergers/oliveraies/palmeraies agricoles (`landuse=orchard|farmland`),
jardins dans l'emprise d'un hôtel/resort. Exemples : jardin Majorelle (payant) exclu ;
Cyber Parc Arsat Moulay Abdeslam (public) inclus. Entrées : nœuds `entrance` OSM s'ils
existent, sinon nœuds du graphe piéton à ≤ 15 m du contour. Règle détaillée et cas
litigieux : phase 1 (`docs/osm_rules.md`).

## D5 — Parcelles candidates (`data.candidates`)

Règle explicite (les parcelles n'existent pas dans OSM) : `landuse=brownfield|greenfield|grass|vacant`,
`natural=scrub|bare_ground`, `amenity=parking` en surface ; on retire tout ce qui intersecte un
bâtiment, les zones militaires, religieuses, cimetières, eau, voies ferrées, ainsi que les
espaces verts déjà comptés ; surface minimale 500 m² (≈ 8 px). Ce sont des **hypothèses de
travail**, pas des terrains réellement aménageables.

## D6 — Conditionnement c = (b, a)

- `b` : fraction de la surface de la tuile (0,5 % à 5 %), stockée aussi en m².
- `a` : A400 visé **après** aménagement. À l'entraînement, `a` est l'A400 réellement atteint
  par la solution de référence (étiquetage a posteriori) : il est donc toujours atteignable.
- À l'évaluation, on ajoute des cibles volontairement inatteignables (A400 glouton + 10 ou
  25 points) pour vérifier que les sorties restent valides et ne violent pas les contraintes.
- `|â − a|` est rapporté tel quel, avec une colonne « cible atteignable par le glouton ».
  Le glouton n'étant pas optimal, « atteignable » est une **estimation par le bas**, pas
  une borne exacte. Les métriques sont rapportées en global, sur les cibles atteignables
  et sur les non atteignables.

## D7 — Projection de validité

Travail au pixel à l'intérieur de R : on garde les composantes connexes qui touchent le
réseau piéton (≤ 1 px), on les trie par probabilité moyenne et on les ajoute jusqu'au budget
b (la dernière composante est rognée sur ses pixels les plus probables). Les éléments non
modifiables sont conservés. La projection garantit les contraintes géométriques, pas
l'atteinte d'un objectif `a` impossible.

## D8 — Variantes et ablations (`configs/model`, `configs/ablation`)

- A : (b, a) en canaux constants ; B : A + FiLM ; C : B + attention guidée par R + projection.
- Le mécanisme de diversité (MSGAN + régression du code latent, z injecté par AdaIN) est
  **identique** dans A, B et C : la comparaison isole ainsi le conditionnement. Son effet
  est mesuré par l'ablation `no_diversity` sur C.
- Ablations (sur C) : `no_modulation`, `no_attention`, `no_projection`, `no_diversity`.
- L'évaluation rapporte toujours les sorties brutes ET projetées pour toutes les variantes.

## D9 — Découpage train/val/test (`data.split`)

- Partition par quartier **avant** toute création de variantes ou de paires.
- Après le rapport de couverture (phase 1, `docs/coverage.md`) : train = Massira +
  Mabrouka, val = Hivernage, test = Guéliz. Guéliz est la zone la mieux renseignée, donc le
  test le plus informatif. Guéliz et Hivernage, voisins et semblables, ne sont jamais l'un
  en train et l'autre en test.
- Zone tampon de 400 m aux frontières : une tuile qui chevauche deux splits (tampon compris)
  est exclue.
- Repli si un quartier est trop pauvre dans OSM : blocs spatiaux de tuiles contigus à
  l'intérieur des quartiers (`split.strategy=spatial_blocks`).
- Les cartes synthétiques ne servent qu'à l'entraînement et sont rapportées à part ;
  le test final se fait uniquement sur cartes réelles.

## D10 — Jalon J1 (`data.source`)

`data.source` vaut `real`, `synthetic` ou `mixed`. Si le corpus réel n'est pas prêt au jalon
J1 (fin de la semaine 2, date à préciser), on passe `data.source=synthetic` sans modifier
le code.

## D11 — Outillage

- `uv`, dépendances verrouillées (`uv.lock`). Python 3.12 en local, code compatible 3.11
  (images Kaggle) : `requires-python >=3.11,<3.13`, ruff/mypy ciblent 3.11, la CI teste
  3.11 et 3.12. Raison : sous Windows, Smart App Control bloque la DLL SSL du Python 3.11
  fourni par uv (aucun accès HTTPS), alors que le 3.12 fonctionne.
- PyTorch : wheels CUDA 12.8 sous Windows (GTX 1650 locale), CPU sous Linux (CI). Sur Kaggle,
  torch est déjà installé ; le notebook installe le paquet sans réinstaller torch.
- Hydra pour la composition, validée par un schéma typé (`happycity.config.AppConfig`) ;
  la CLI `typer` compose la configuration via l'API `compose` de Hydra.
- PyTorch « nu » plutôt que Lightning : une boucle GAN à deux optimiseurs est plus lisible
  écrite explicitement ; suivi TensorBoard.
- Interface bonus : Streamlit, appelant les mêmes fonctions que la CLI.

## D12 — Source OSM et zones d'étude

- Extrait **Geofabrik daté** (12/11/2026), découpé localement avec pyosmium, plutôt que
  l'API Overpass : reproductible, et les serveurs publics étaient saturés le 13/11/2026.
  Overpass reste disponible (`data.extraction.source=overpass`, avec attente exponentielle).
- Aucune limite OSM exploitable pour les 4 quartiers : zones manuelles de 2048 m autour
  d'un repère OSM (`docs/boundaries.md`).
- Python 3.12 en local (compatible 3.11) : Smart App Control bloque le SSL du Python 3.11
  fourni par uv.

## D13 — Rastérisation

- Surfaces en **fraction de couverture** (suréchantillonnage ×4) plutôt qu'en binaire :
  un bâtiment de 6 m n'est pas perdu à 8 m/px. Réseau en `all_touched`, R au centre du
  pixel (identifiants cohérents avec la projection).
- Tuiles réelles chevauchantes (pas de 512 m) : 36 tuiles au lieu de 16, au prix d'une
  corrélation entre tuiles d'une même zone, que l'on rapportera.
- Tuile exploitable : au moins 30 origines et au moins une parcelle adjacente au réseau.

## D14 — Métriques (docs/metrics.md)

- Points d'accès sur les **tronçons** (point le plus proche à ≤ 15 m du contour), pas
  seulement aux nœuds. Les entrées OSM explicites sont prioritaires. La même règle vaut
  pour les espaces verts existants et générés.
- Module `metrics` autonome (aucun import `happycity` hors de lui-même) ; un contexte
  d'évaluation par tuile (sous-graphe exact à 400 + 100 m).
- E par tuile (4 quadrants) et par quartier (4 tuiles disjointes).
- L'ancienne couche `entrances` (méthode `boundary`, nœuds) ne sert plus qu'au rapport de
  couverture de la phase 1 ; l'A400 utilise les points d'accès sur tronçons.

## D15 — Dépôt GitHub

Dépôt **public** : le notebook Kaggle le clone simplement, sans token.

## D16 — Références et paires (docs/baselines_pairs.md)

- Morceaux de parcelle **compacts** ancrés sur le réseau (pas de bandes ni d'anneaux).
- Glouton = meilleur des critères « gain par pixel » et « gain absolu ».
- Paires : glouton stochastique (k = 3), 8 tirages, solutions de qualité voisine
  (gain ≥ 80 % du meilleur), au plus 3 distinctes ; a = A400 atteint.
- 10 % des cartes synthétiques en `val_synth` pour suivre l'entraînement.
- **Validé par l'utilisateur** : environ 10 % des cartes synthétiques réservées en
  `test_synth`, jamais vues à l'entraînement ni utilisées pour choisir un modèle. Ce test est
  rapporté **séparément** et ne remplace pas le **test final, qui reste réel** (Guéliz). Il
  existe parce que le test réel discrimine mal la qualité du placement (63 % des paires ont
  un budget saturé). Le découpage se fait par carte, avec un hachage stable : val_synth
  n'est pas modifié par l'ajout de test_synth.
- Échelle constante imposée par le code : `rasterize_tile` refuse une tuile ≠ 128 × 8 m.

## D17 — Modèles, projection, diversité (docs/models.md)

- Sortie de G = carte de probabilité d'**ajout** ; la carte « après » est composée de façon
  déterministe, et l'existant n'est jamais modifié.
- Une seule architecture (U-Net + PatchGAN) ; A / B / C ne diffèrent que par le
  conditionnement, l'attention et la projection. Le discriminateur, z, les pertes et la
  diversité sont communs.
- Perte de condition : **A400 doux**, construit à partir de motifs de desserte par pixel
  calculés avec des distances **réseau**. Il surestime l'exact de +0,003 en moyenne et sert
  uniquement à l'entraînement et à la validation ; les résultats utilisent la métrique exacte.
- MSGAN mesuré dans R et borné (ε = 0,05).
- Sélection du checkpoint sur `val_synth`, jamais sur le test.
- L'entraînement n'importe pas la pile géographique (test d'architecture) : il tourne sur
  Kaggle avec `.npz` + torch.
- GTX 1650 : AMP fp16 2,6 fois plus lent → `train.amp=false` en local.
- Smart App Control (Windows) bloque par intermittence des DLL natives, dont osmium. Son
  import est désormais local à `clip_pbf`. Ces blocages expliquent les échecs de tests
  ponctuels non reproductibles.

## D18 — Évaluation (docs/evaluation.md)

- Problèmes : cibles a posteriori et cibles inatteignables (glouton + 0,10 / + 0,25), avec
  une colonne « atteignable par le glouton », qui est une estimation par le bas.
- Métriques exactes uniquement (Dijkstra) ; E toujours avec A400 ; sorties brutes ET
  projetées ; mêmes z pour tous les modèles.
- Référence « glouton ciblé » : arrêt dès que A400 ≥ a. C'est la référence juste pour juger
  la fidélité à la condition.
- Graines : nom d'exécution suffixé (`-s42`) ; agrégation moyenne ± écart-type entre graines.
- `rasterio < 1.4.4` et `pyarrow != 26.0.1` : DLL bloquées par Smart App Control.

## D19 — Démonstration et réponse à la condition (docs/demo.md)

- Un seul module `happycity.demo` pour la CLI, le notebook et Streamlit ; un test interdit à
  l'interface tout autre import de `happycity`.
- Le moteur matplotlib « Agg » est réglé seulement dans la CLI, les tests et Streamlit, pour
  que le notebook affiche ses figures.
- **Constat** : les modèles ignorent a (pente ≈ 0 dans 96 à 98 % des cas, 3 graines). Les
  paires de la phase 4 saturent le budget, si bien que a est redondante avec b. Le test de
  réponse à la condition fait désormais partie de `happycity evaluate`. **Correction
  possible** : paires à a variable pour un budget fixe (glouton ciblé arrêté à des cibles
  tirées entre l'A400 avant et le maximum), puis réentraînement. Elle n'est pas faite : c'est
  à l'utilisateur de décider.

## D20 — Paires v2 à cible variable (validé par l'utilisateur le 20/11/2026)

- Ajout de solutions « glouton ciblé stochastique » arrêtées à 25, 50 et 75 % du gain
  maximal, pour que a varie à budget fixe (écart moyen 0,023 → 0,209).
- Les modèles v1 (qui ignorent a) sont archivés dans `experiments_v1/`, avec leurs paires
  et leur jeu compact. Réentraînement complet sur Kaggle (13 exécutions), puis réévaluation.
- Notebook Kaggle : `RUNS = (variante, ablation, graines)`, une seule version pour tout.

## D21 — Questions tranchées (26/11/2026, par l'utilisateur)

- **Zone Mabrouka** : emplacement **validé** par l'utilisateur (quartier résidentiel le long
  de la voie ferrée, friche centrale, zone industrielle au nord).
- **`landuse=village_green`** : **non compté** comme espace vert public. Les occurrences
  observées sont des oliveraies et vergers (Ménara), pas des espaces aménagés.
- **Dépôt GitHub** : public, https://github.com/soufianebouragaa01/happycity-gen
  (renseigné dans le notebook Kaggle).

## Questions ouvertes

- Date exacte du jalon J1 : laissée à `[JJ/MM/2026]`, à compléter par l'utilisateur.
