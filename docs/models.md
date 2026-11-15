# Modèles A/B/C, projection et diversité (phase 5)

Code : `src/happycity/models/`, `src/happycity/projection/`, `src/happycity/training/`.
Commandes : `happycity make-dataset`, puis `happycity train model=C [ablation=…]`.
Kaggle : `notebooks/kaggle_train.ipynb`. Tests : `tests/test_models.py`,
`tests/test_projection.py`.

## Entrées et sortie

**G(M, R, c = (b, a), z)** :

| Entrée | Contenu |
|---|---|
| M (7 canaux) | bâti résidentiel, autre bâti, réseau piéton, espaces verts existants, obstacles, origines, origines non desservies (comptes en log, ramenés à [0, 1]) |
| R (1 canal) | région modifiable : pixels des parcelles candidates **adjacentes au réseau** |
| c | b (fraction de la tuile, ×20) et a (A400 cible) |
| z | code latent N(0, I) de dimension 8 |

**Sortie** : une carte 128 × 128 de probabilité d'**ajout** d'espace vert.

La carte sémantique « après » est composée de façon déterministe par
`compose_after(M, ajouts)` : seul le canal des espaces verts change, et l'existant n'est
jamais modifié.

## Variantes : une seule architecture, une seule variable change

U-Net à 5 niveaux (128 → 4 px) et PatchGAN 70 × 70 avec normalisation spectrale. Le
discriminateur, le bruit z (AdaIN dans le décodeur), les pertes et la diversité sont
identiques pour A, B et C.

| Variante | Conditionnement c | Attention guidée par R | Projection | Paramètres de G |
|---|---|---|---|---|
| **A** | 2 canaux constants concaténés | — | — | 4,24 M |
| **B** | MLP → FiLM sur chaque bloc (encodeur et décodeur) | — | — | 4,39 M (+3,7 %) |
| **C** | FiLM | portes d'attention sur les 4 connexions de saut | oui | 4,48 M |

- **FiLM** et AdaIN sont initialisés à l'identité : au départ, B se comporte comme un U-Net
  non conditionné par c, puis apprend la modulation.
- **Attention guidée par R** : α = σ(ψ(W_s·saut + W_g·décodeur + W_r·R)). La région
  modifiable décide, avec le décodeur, quelles zones de l'encodeur sont transmises.
- **Projection** : en entraînement, seule sa partie différentiable est appliquée (sortie × R) ;
  en inférence, l'opérateur complet.
- **Ablations de C** : `no_modulation` (concaténation), `no_attention`, `no_projection`,
  `no_diversity`.

## Opérateur de projection (déterministe, non appris)

`happycity.projection.project(prob, R, réseau, budget)` :

1. **restriction à R** : rien en dehors des parcelles adjacentes au réseau. Le bâti, le
   réseau, les obstacles et les espaces verts existants sont non modifiables ;
2. **budget** : au plus b pixels, en gardant les plus probables (ordre stable à égalité) ;
3. **adjacence** : on garde seulement les composantes connexes qui touchent le voisinage du
   réseau (≤ 1 px). Cette étape vient **après** le budget, car tronquer peut détacher un
   morceau de la rue (cas testé).

Garanties testées (10 cartes aléatoires et des cas construits) : validité systématique,
déterminisme et idempotence. L'opérateur renvoie aussi le nombre de pixels retirés à chaque
étape, ce qui sert à mesurer les violations des sorties brutes de A et B.

## Pertes

| Terme | Poids | Définition |
|---|---|---|
| adversariale | 1 | LSGAN ; bruit d'instance σ = 0,1 décroissant jusqu'à 0, sur les masques réels ET générés |
| reconstruction | 100 | L1 dans R avec la solution cible (supervision simulée) |
| violations | 100 | (masse hors de R + dépassement du budget) / budget |
| **condition** | 10 | \|Â400_doux(G) − a\| |
| mode seeking (MSGAN) | 1 | 1 / (‖G(z₁) − G(z₂)‖_R / ‖z₁ − z₂‖ + 0,05) |
| régression latente (BicycleGAN) | 0,5 | ‖E(M, R, G(z)) − z‖₁ |

### A400 doux : la condition a devient différentiable

A400 passe par des plus courts chemins (Dijkstra) et n'est pas dérivable. Le calcul se fait
en deux temps.

**Précalcul, par tuile.** Pour chaque pixel de R, on détermine l'ensemble des origines
**non encore desservies** qu'il desservirait s'il devenait vert. Le calcul utilise les mêmes
conventions que la métrique exacte : point d'accès sur chaque tronçon à ≤ 15 m, distance
**réseau** ≤ 400 m, et la nouvelle fonction `metrics.pixel_service`. Les pixels qui ont le
même ensemble partagent un **motif** (89 motifs par tuile en moyenne).

**Pendant l'entraînement.** On calcule P(origine desservie) = 1 − Π(1 − pᵢ), le produit
portant sur les pixels qui la desservent. On en déduit
Â = (desservies avant + Σ P) / nombre d'origines.

Fidélité, mesurée sur les 6 443 masques cibles :

| | Valeur |
|---|---|
| écart moyen (doux − exact) | +0,003 |
| écart médian | 0 |
| 95e centile | 0,022 |
| cibles à ≤ 1 point | 89,8 % |

L'estimation ne peut que **sur**estimer : chaque pixel apporte son propre point d'accès sur
un tronçon, alors que la métrique exacte n'en prend qu'un par polygone et par tronçon. Les
résultats rapportés utiliseront **toujours** la métrique exacte (phase 6) ; l'A400 doux ne
sert qu'à l'entraînement et à la validation.

## Diversité

- **MSGAN** : l'écart entre deux sorties est mesuré **dans R**. Mesuré sur toute l'image, le
  rapport serait minuscule même pour des propositions entièrement différentes, puisque les
  ajouts occupent 1 à 5 % de la tuile. La perte est **bornée**, faute de quoi elle valait 175
  au premier pas (sorties identiques tant que la modulation par z est l'identité) et
  écrasait tous les autres termes.
- **Régression latente** : un encodeur doit retrouver z à partir de la sortie, ce qui
  empêche G d'ignorer z.
- **Mesure** : 1 − IoU moyen entre 3 propositions projetées par paire de validation.

## Données d'entraînement et validation

`make-dataset` produit un `.npz` par split (6,8 Mo au total, numpy seul). Les augmentations
sont les 8 symétries du carré ; la carte des motifs est transformée avec le reste, ce qui est
testé.

À chaque époque, la validation porte sur `val_synth` et `val` (Hivernage) :

- violations brutes ;
- |Â − a| des sorties brute et projetée ;
- remplissage du budget ;
- diversité.

Le checkpoint `best.pt` est choisi sur **val_synth**, d'après le score
|Â − a| de la sortie retenue + violations. Pour A et B, la sortie retenue est la sortie brute,
et leurs violations sont donc pénalisées. **Le test n'intervient jamais dans la sélection.**

Une exécution enregistre :

- `config.yaml` (configuration résolue) ;
- `meta.json` (commit git, versions, GPU, nombre de paramètres) ;
- `train_log.csv` et `val_log.csv` ;
- `checkpoints/{last,best}.pt`, qui contiennent la configuration, d'où un rechargement sans
  ambiguïté.

## Matériel

| GPU | Temps par pas (lot de 32) |
|---|---|
| GTX 1650 locale, fp32 | ≈ 2,4 s, soit environ 6,5 min par époque |
| GTX 1650 locale, AMP fp16 | ≈ 6,2 s (pas de cœurs tensoriels) |

En local, il faut donc `train.amp=false`, et seuls des essais courts sont réalistes.
L'entraînement complet (7 exécutions × 40 époques) se fait sur Kaggle : T4, où l'AMP est
active.
