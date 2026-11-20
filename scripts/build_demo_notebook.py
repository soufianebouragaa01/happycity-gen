"""Construit et exécute ``notebooks/demo.ipynb`` (sorties incluses).

Usage : ``python scripts/build_demo_notebook.py`` depuis la racine du dépôt (données,
contextes et checkpoints locaux nécessaires). Le notebook n'appelle que
:mod:`happycity.demo`, comme la commande ``happycity demo`` et l'interface Streamlit.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# HappyCity Gen : démonstration

Scénarios d'**espaces verts** sur des tuiles de 1 024 m × 1 024 m (8 m/px) à Marrakech.
Un modèle génératif conditionnel propose où créer de nouveaux espaces verts, sur les seules
parcelles autorisées (R), sous un **budget b**, en visant une **cible a** d'accessibilité.

Chaque proposition est chiffrée par le module de métriques **indépendant** :
- **A400** : part des habitations à ≤ 400 m d'un espace vert, en distance **réseau** piéton
  (Dijkstra) ;
- **E** : équité entre secteurs de la tuile, toujours lue **avec** A400 ;
- **V** : couverture végétale.

Ce notebook, la commande `happycity demo` et l'interface Streamlit (`happycity demo --ui`)
appellent les **mêmes fonctions** (`happycity.demo`).""",
    ),
    (
        "code",
        """import os
from pathlib import Path

os.chdir(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
import pandas as pd
from happycity import demo
from happycity.config import load_config

pd.set_option("display.precision", 3)
cfg = load_config()
print(demo.DISCLAIMER)""",
    ),
    ("md", "## 1. Tuiles et modèles disponibles"),
    (
        "code",
        """tiles = demo.available_tiles(cfg)
runs = demo.available_runs(cfg)
model = demo.default_run(runs)
print(len(tiles), "tuiles ;", len(runs), "modèles ; modèle par défaut :", model.name)
tiles[tiles["split"] == "test"][["tile_id", "zone", "A400_before", "E_before", "capacity_px",
                                 "greedy_0.01", "greedy_0.02", "greedy_0.05"]]""",
    ),
    (
        "md",
        """## 2. Un scénario sur Guéliz (test réel)

Budget 2 % de la tuile, cible = ce que le glouton atteint avec ce budget. La **référence
glouton ciblé** (non générative) reçoit la même cible.""",
    ),
    (
        "code",
        """row = tiles.set_index("tile_id").loc["gueliz_00"]
s = demo.run_scenario(cfg, model, row["pair_file"], 0.02, float(row["greedy_0.02"]) - 0.005, n=3)
demo.plot_scenario(s)
demo.scenario_table(s)""",
    ),
    (
        "md",
        """Sur les tuiles réelles, il y a peu de parcelles aménageables (ici quelques-unes) : toutes
les méthodes convergent vers la même solution, et la diversité est nulle. C'est la limite
principale du test réel (voir `docs/evaluation.md`).""",
    ),
    ("md", "## 3. Une tuile synthétique : plusieurs solutions possibles"),
    (
        "code",
        """row = tiles.set_index("tile_id").loc["synth_00545"]
s = demo.run_scenario(cfg, model, row["pair_file"], 0.02, 0.95, n=3, seed=1)
demo.plot_scenario(s)
print("diversité des propositions :", round(demo.diversity(s), 3))
demo.scenario_table(s)""",
    ),
    (
        "md",
        """## 4. Une cible inatteignable

On demande plus que ce que le glouton atteint avec ce budget. Le modèle s'en approche au
mieux **sans jamais violer** les contraintes (projection) ; |â − a| est rapporté tel quel.""",
    ),
    (
        "code",
        """row = tiles.set_index("tile_id").loc["synth_00545"]
s = demo.run_scenario(cfg, model, row["pair_file"], 0.01, 1.0, n=3)
print("cible atteignable par le glouton :", "oui" if s.reachable else "non",
      f"(glouton : {s.greedy_max:.1%})")
demo.scenario_table(s)""",
    ),
    (
        "md",
        """## 5. Pourquoi la projection compte : variante A (sans projection)

Même tuile, même budget et même cible avec le modèle **A**. Ses sorties brutes ne passent pas
par l'opérateur de projection : la colonne « valide » montre les violations éventuelles
(ajouts hors des parcelles autorisées, au-delà du budget ou détachés du réseau).""",
    ),
    (
        "code",
        """model_a = next(r for r in runs if r.variant == "A" and r.seed == model.seed)
row = tiles.set_index("tile_id").loc["gueliz_11"]
s = demo.run_scenario(cfg, model_a, row["pair_file"], 0.02, 0.40, n=3)
demo.plot_scenario(s)
demo.scenario_table(s)""",
    ),
    (
        "md",
        """## 6. La cible a pilote-t-elle le résultat ?

On fait varier la cible a sur une même tuile, avec le même budget. Si le modèle utilisait a,
l'A400 obtenu suivrait la diagonale grise, comme le fait (par paliers) le glouton ciblé.""",
    ),
    (
        "code",
        """import matplotlib.pyplot as plt

row = tiles.set_index("tile_id").loc["synth_00545"]
targets = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
rows = []
for a in targets:
    sc = demo.run_scenario(cfg, model, row["pair_file"], 0.03, a, n=3)
    t = demo.scenario_table(sc)
    gen = t[t["option"].str.contains("proposition")]
    rows.append({"cible a": a, "A400 modèle (moyenne)": gen["A400"].mean(),
                 "A400 glouton ciblé": t["A400"].iloc[-1]})
sweep = pd.DataFrame(rows)
ax = sweep.plot(x="cible a", marker="o", figsize=(5, 3.5))
ax.plot([0.75, 1], [0.75, 1], color="#bbb", lw=1, zorder=0)
ax.set_ylabel("A400 atteint (exact)")
sweep""",
    ),
    (
        "md",
        """**Non.** Le modèle donne le même A400 quelle que soit la cible : il **maximise l'accès
avec le budget** et ignore a. Ce n'est pas propre à cette tuile : le test de réponse à la
condition (`happycity evaluate`, `docs/evaluation.md`) donne une pente ≈ 0 pour A, B et C
sur les 3 graines. Cause : les paires d'entraînement saturent toujours le budget, si bien
que a n'apporte aucune information de plus que b.""",
    ),
    (
        "md",
        """## 7. Résultats d'ensemble (phase 6)

Rappel du tableau principal sur le **test synthétique** (moyenne ± écart-type sur 3 graines
pour A, B, C). Détails et limites : `docs/evaluation.md`.""",
    ),
    (
        "code",
        """evaluation = Path(cfg.paths.experiments_dir) / "evaluation"
summary = pd.read_csv(evaluation / "summary_test_synth.csv")
summary[["méthode", "graines", "|â−a| atteignable", "|â−a| non atteignable", "A400 après",
         "E après", "valide", "diversité"]]""",
    ),
    (
        "md",
        """**À retenir**
- Les modèles placent bien les espaces verts (A400 proche du glouton, 2 fois moins d'écart
  que l'aléatoire) mais **ignorent la cible a** : ils maximisent l'accès avec le budget ;
- **C** est le seul valide à 100 % (grâce à la projection), au prix d'une **faible
  diversité** ;
- Ces indicateurs décrivent l'**accès piéton** et la **végétation**, pas le bonheur des
  habitants.""",
    ),
]


def main() -> None:
    """Construit, exécute et écrit le notebook."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, text in CELLS:
        cell = (
            nbformat.v4.new_markdown_cell(text) if kind == "md" else nbformat.v4.new_code_cell(text)
        )
        nb.cells.append(cell)
    NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    ).execute()
    out = ROOT / "notebooks" / "demo.ipynb"
    nbformat.write(nb, out)
    print("écrit :", out)


if __name__ == "__main__":
    main()
