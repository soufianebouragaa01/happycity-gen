# Générateur synthétique (`source=synthetic`)

Code : `src/happycity/data/synthetic.py`. Paramètres : `data.synthetic` dans
`configs/data/marrakech.yaml`. Commande : `happycity synthesize --n 600`.

Les cartes synthétiques **complètent** le corpus réel : elles servent **uniquement à
l'entraînement**, sont toujours étiquetées `source=synthetic` et sont rapportées à part.
Le test final se fait sur des cartes réelles.

## Principe

Chaque carte est une tuile de 1024 m, entourée de 500 m de contexte, au **même format**
que l'extraction OSM (`UrbanLayers`). Les origines et les entrées sont calculées par les
mêmes fonctions (`network.py`). La rastérisation et les métriques ne distinguent donc pas
les deux sources.

1. **Trame viaire** : grille tournée d'un angle aléatoire (±45°). Les pas sont tirés dans
   `block_m`, les nœuds bruités (`jitter_m`), et des tronçons supprimés avec la
   probabilité `edge_drop` (impasses, îlots fusionnés). On garde la plus grande
   composante connexe.
2. **Îlots** : faces de la trame, en retrait de 5 m de l'axe des rues.
3. **Occupation** de chaque îlot, tirée selon `p_park`, `p_vacant`, `p_parking`,
   `p_other`, `p_obstacle` et `p_empty` ; les îlots restants sont résidentiels.
   - parc → espace vert public compté ;
   - terrain vacant ou parking de surface → parcelle candidate (même seuil de 500 m²) ;
   - équipement → bâtiment non résidentiel ;
   - obstacle → type cimetière (non modifiable) ;
   - îlot vide → non bâti et non candidat (terrains privés ou non cartographiés, très
     présents dans OSM).
4. **Résidentiel** : découpage en lots carrés (`lot_m`), un bâtiment centré par lot, dont
   l'emprise est tirée dans `building_coverage`.

La reproductibilité est assurée par une graine par carte (`SeedSequence([seed, index])`) :
même index, même carte, sans dépendre des autres cartes.

## Typologies

| Typologie | Poids | Inspirée de | Îlots | Lots | Emprise |
|---|---|---|---|---|---|
| `dense` | 0,40 | Massira | 60–110 m | 24–34 m | 45–70 % |
| `grid` | 0,35 | Guéliz | 80–130 m | 25–40 m | 40–70 % |
| `villas` | 0,25 | Hivernage | 110–180 m | 35–60 m | 15–35 % |

## Calibrage sur les zones réelles

Ce sont des moyennes sur 60 cartes, comparées aux 4 zones réelles. Les effectifs sont
exprimés par km² (`happycity coverage-report`, fichier `calibration.csv`).

| | emprise bâtie | réseau (km/km²) | V | parcs /km² | parcelles /km² | origines /km² |
|---|---|---|---|---|---|---|
| Guéliz (réel) | 0,25 | 21,8 | 0,020 | 5,0 | 2,6 | 551 |
| Hivernage (réel) | 0,09 | 17,5 | 0,022 | 1,4 | 2,9 | 109 |
| Mabrouka (réel) | 0,16 | 16,8 | 0,006 | 2,4 | 1,9 | 499 |
| Massira (réel) | 0,27 | 29,3 | 0,001 | 0,7 | 0,7 | 586 |
| `dense` | 0,29 | 20,9 | 0,007 | 1,2 | 7,6 | 797 |
| `grid` | 0,29 | 19,0 | 0,017 | 2,4 | 8,3 | 686 |
| `villas` | 0,10 | 12,7 | 0,015 | 0,8 | 6,6 | 157 |

Choix assumés :

- **Lots larges** : dans OSM à Marrakech, un « bâtiment » est souvent un bloc de maisons
  mitoyennes. Une première version à petits lots donnait 3 900 origines/km², contre environ
  550 en réalité. Elle a été corrigée.
- **Plus de parcelles candidates que dans le réel** (environ 7/km² contre 1 à 3). OSM
  sous-cartographie les terrains vacants, et un nombre suffisant de parcelles par tuile est
  nécessaire pour que le placement soit un vrai choix (sinon aléatoire ≈ glouton). L'écart
  de distribution réel/synthétique est donc connu, et sera mesuré à l'évaluation
  (résultats rapportés séparément).
- La typologie `villas` a un réseau plus lâche (12,7 contre 17,5 km/km² à l'Hivernage).
  C'est acceptable pour l'entraînement.

## Limites

Géométrie orthogonale simplifiée (pas de rues courbes ni de grandes avenues). Pas de
voie ferrée ni d'oued traversant. Parcs toujours limités à un îlot entier. Pas d'espaces
verts exclus (privés, payants) dans le synthétique.
