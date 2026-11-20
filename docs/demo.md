# Démonstration : CLI, notebook et interface Streamlit (phase 7)

Les trois points d'entrée appellent **les mêmes fonctions** (`src/happycity/demo.py`). Ils ne
contiennent aucune logique propre, et un test vérifie que l'interface n'importe rien d'autre
de `happycity` que `demo` et la configuration.

| Point d'entrée | Commande | Sortie |
|---|---|---|
| CLI | `happycity demo --tile gueliz_11 --budget 0.02 [--target 0.4] [--n 3] [--run C-full-s43]` | figure PNG + tableau CSV dans `experiments/demo/` |
| Interface | `happycity demo --ui` (ou `streamlit run app/streamlit_app.py`) | navigateur, http://localhost:8501 |
| Notebook | `notebooks/demo.ipynb`, livré **exécuté** ; reconstruction : `python scripts/build_demo_notebook.py` | scénarios commentés |

Prérequis : les données traitées (`data/processed/`) et au moins un modèle entraîné
(`experiments/seed<graine>/<exécution>/checkpoints/best.pt`). L'interface demande les extras
`app` (`uv sync --extra app`) ; le notebook, l'extra `notebook`.

## Un scénario

On choisit une tuile, un budget b (1, 2, 3 ou 5 % de la tuile) et une cible a (A400 visé).
`demo.run_scenario` renvoie :

- **n propositions** du modèle : sortie projetée pour C, brute pour A, B et C sans
  projection ;
- la **référence glouton ciblé** (non générative), qui reçoit aussi a ;
- pour l'état initial et chaque option, les métriques **exactes** du module indépendant :
  A400, ΔA400, **E (toujours avec A400)**, V, surface ajoutée, validité et |â − a| ;
- l'indication « **cible atteignable par le glouton** (oui/non) ». Le glouton n'étant pas
  optimal, c'est une estimation par le bas.

L'interface affiche aussi :

- un avertissement de **budget saturé** quand la surface aménageable de la tuile est
  inférieure au budget ;
- la **diversité** des propositions (1 − IoU) ;
- l'avertissement suivant, systématique : *les indicateurs mesurent l'accès piéton à 400 m
  et la végétation, pas le bien-être des habitants ; les propositions ne sont pas des projets
  urbains*.

Un scénario prend 0,3 à 0,5 s une fois le modèle chargé. Le **premier** lancement peut prendre
plusieurs minutes sur cette machine, le temps que Windows analyse les bibliothèques natives.

## Ce que montre la démonstration

1. **Tuiles réelles (Guéliz)** : très peu de parcelles aménageables. Toutes les méthodes
   convergent vers la même solution, et la diversité est nulle.
2. **Tuiles synthétiques** : plusieurs solutions de qualité voisine, une diversité faible mais
   non nulle, et un A400 proche du glouton.
3. **Cible inatteignable** : le modèle C reste valide et s'approche du maximum.
4. **Variante A sans projection** : des propositions **non valides** (ajouts hors des
   parcelles autorisées ou détachés du réseau). La projection est l'ingrédient décisif.
5. **Balayage de la cible a : le modèle ignore a.** L'A400 obtenu reste le même quand la cible
   varie de 0,75 à 1,0 ; le glouton ciblé, lui, la suit. Ce constat, généralisé par le test
   de réponse à la condition, est documenté dans [evaluation.md](evaluation.md#réponse-à-la-condition-a-ajout-de-la-phase-7).

## Fichiers

- `src/happycity/demo.py` : `available_tiles`, `available_runs`, `default_run`,
  `run_scenario`, `scenario_table`, `diversity`, `plot_scenario`, `DISCLAIMER` ;
- `app/streamlit_app.py` : interface ;
- `notebooks/demo.ipynb`, `scripts/build_demo_notebook.py` : notebook ;
- `tests/test_demo.py` : module, CLI, interface (`streamlit.testing`) et règle d'architecture.
