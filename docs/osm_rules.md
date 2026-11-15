# Règles d'inclusion OSM

Toutes les valeurs sont dans `configs/data/marrakech.yaml`. Les règles sur les tags sont
dans `src/happycity/data/rules.py` (fonctions pures, testées dans `tests/test_rules.py`) ;
les traitements géométriques sont dans `src/happycity/data/osm.py`
(testés dans `tests/test_osm_build.py`).

## Source des données

- **Extrait Geofabrik** `morocco-latest.osm.pbf`, instantané du **12/11/2026**
  (date notée dans `data/raw/osm/morocco-latest.osm.json` et dans les métadonnées de chaque
  zone). On a choisi un instantané daté plutôt que l'API Overpass : les données sont
  reproductibles, on ne sollicite pas les serveurs publics (tous saturés le 13/11/2026), et
  l'extraction marche hors ligne. Overpass reste utilisable avec `data.extraction.source=overpass`.
- **Découpage** par pyosmium (`src/happycity/data/pbf.py`) : on garde chaque objet qui
  intersecte la zone élargie de 500 m, ainsi que les objets qu'il référence.
- **Graphe piéton** : OSMnx `network_type="walk"`. Le filtre OSMnx lui-même est traduit
  en prédicat Python, ce qui garantit la même définition qu'Overpass. Il exclut notamment
  les autoroutes, les pistes cyclables, `access=private`, et les chaussées dont le trottoir
  est cartographié séparément (`sidewalk=separate`).
  Le graphe est bidirectionnel, projeté en EPSG:32629, et réduit à sa plus grande
  composante connexe.

## Espaces verts comptés (publics ou explicitement accessibles)

Un objet surfacique est **compté** si :

1. il porte `leisure=park`, `leisure=garden` ou `leisure=playground` ;
2. il ne porte **pas** de tag exclu : `leisure=golf_course`,
   `landuse=orchard|farmland|farmyard`, `tourism=hotel|resort` ;
3. `access` ne vaut pas `private`, `no`, `customers` ni `permit` ;
4. `fee` est absent ou vaut `no` (tout autre valeur = payant → exclu) ;
5. son point représentatif n'est **pas contenu** dans un polygone portant un tag exclu
   (jardin dans l'emprise d'un hôtel, d'un golf, d'une oliveraie…) ;
6. il n'est pas listé dans `green.manual_exclude` (cas litigieux, motif écrit) ;
7. sa surface, une fois découpé à la zone élargie, est d'au moins 100 m².

Les objets écartés sont conservés dans la couche `green_excluded` avec leur motif
(audit dans le rapport de couverture).

**Entrées** : si des nœuds OSM `entrance=*` ou `barrier=gate` se trouvent à ≤ 15 m du
contour, on retient le nœud piéton le plus proche de chacun (méthode `osm`). Sinon, on
prend tous les nœuds piétons à ≤ 15 m du polygone (méthode `boundary`). Un espace vert
sans aucun nœud proche n'a pas d'entrée et ne compte pas pour l'accessibilité.

## Parcelles candidates (HYPOTHÉTIQUES)

OSM ne recense pas de « terrains aménageables ». On définit une règle explicite. Les
parcelles obtenues sont des **hypothèses de travail**, pas des terrains réellement
disponibles.

1. Tags retenus : `landuse=brownfield|greenfield|grass|vacant`,
   `natural=scrub|bare_ground`, `amenity=parking` en surface (`parking` absent ou
   `surface` ; `multi-storey` et `underground` exclus).
2. On retire géométriquement les bâtiments, les espaces verts déjà comptés, les obstacles,
   et `landuse=military|cemetery|religious|railway`, `natural=water`.
3. On garde les morceaux d'au moins **500 m²** (≈ 8 px à 8 m/px).

Les espaces libres non étiquetés (« vides » entre bâtiments) ne sont pas détectés : ils
sont trop bruités dans OSM. C'est une limite connue.

## Bâtiments résidentiels et origines

Classement, dans cet ordre :

1. `building` explicitement résidentiel (`residential`, `apartments`, `house`…) → résidentiel ;
2. `building` explicitement non résidentiel (`school`, `hotel`, `mosque`, `retail`…) → autre ;
3. présence d'un tag d'activité (`shop`, `amenity`, `office`, `tourism`, `craft`,
   `healthcare`) → autre ;
4. point représentatif dans un `landuse` non résidentiel ou dans l'emprise d'un hôtel → autre ;
5. dans `landuse=residential` → résidentiel ;
6. sinon (`building=yes` sans indice) → **résidentiel par défaut**. C'est une hypothèse
   (`buildings.default_yes`) ; la part de bâtiments classés ainsi est donnée dans le
   rapport de couverture.

**Origines** : centroïdes des bâtiments résidentiels, rattachés au nœud piéton le plus
proche. Une origine est écartée si ce nœud est à plus de 100 m. C'est une proportion
d'origines, **pas** une population : aucune donnée de population n'est utilisée.

## Obstacles (non modifiables)

`natural=water`, `landuse=military|cemetery|railway|reservoir`,
`waterway=riverbank|canal`, et `railway=rail` (élargi de 5 m de chaque côté).
