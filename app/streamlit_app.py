"""Interface HappyCity Gen (Streamlit) : explorer des scénarios d'espaces verts.

Lancement : ``happycity demo --ui`` (ou ``streamlit run app/streamlit_app.py``).

Aucune logique propre : l'interface appelle uniquement les fonctions de
:mod:`happycity.demo`, les mêmes que la commande ``happycity demo`` et le notebook.
"""

from __future__ import annotations

import math
import sys

import matplotlib
import streamlit as st

matplotlib.use("Agg")  # rendu hors écran, sûr dans les fils d'exécution de Streamlit

from happycity import demo
from happycity.config import load_config

st.set_page_config(page_title="HappyCity Gen", layout="wide")


@st.cache_resource
def _config(overrides: tuple[str, ...]):  # type: ignore[no-untyped-def]
    return load_config(list(overrides))


@st.cache_data
def _tiles(_cfg, key: str):  # type: ignore[no-untyped-def]
    return demo.available_tiles(_cfg)


cfg = _config(tuple(sys.argv[1:]))
tiles = _tiles(cfg, str(cfg.paths.processed_dir))
runs = demo.available_runs(cfg)

st.title("HappyCity Gen : scénarios d'espaces verts à Marrakech")
st.caption(demo.DISCLAIMER)

if not runs:
    st.error("Aucun modèle entraîné dans `experiments/` (voir le notebook Kaggle).")
    st.stop()

# --------------------------------------------------------------------------- paramètres
with st.sidebar:
    st.header("Scénario")
    names = [r.name for r in runs]
    default = demo.default_run(runs)
    run_name = st.selectbox("Modèle", names, index=names.index(default.name) if default else 0)
    run = next(r for r in runs if r.name == run_name)

    split_labels = {"test": "Guéliz (test réel)", "val": "Hivernage (validation réelle)",
                    "train": "Massira / Mabrouka et synthétique (entraînement)",
                    "test_synth": "Synthétique (test)",
                    "val_synth": "Synthétique (validation)"}  # fmt: skip
    splits = [s for s in split_labels if s in set(tiles["split"])]
    split = st.selectbox("Jeu de tuiles", splits, format_func=lambda s: split_labels[s])
    choices = tiles[tiles["split"] == split]
    tile_id = st.selectbox("Tuile", list(choices["tile_id"]))
    row = choices.set_index("tile_id").loc[tile_id]

    fractions = sorted(float(c.removeprefix("greedy_")) for c in tiles.columns
                       if c.startswith("greedy_"))  # fmt: skip
    b_frac = st.select_slider("Budget b (part de la tuile)", fractions, value=fractions[1],
                              format_func=lambda f: f"{f:.0%}")  # fmt: skip
    greedy_max = float(row[f"greedy_{b_frac:.2f}"])
    default_a = math.floor(greedy_max * 100) / 100  # arrondi vers le bas : reste atteignable
    a_target = st.slider("Cible a (A400 visé)", 0.0, 1.0, value=default_a, step=0.01)
    n = st.slider("Nombre de propositions", 1, 5, 3)
    seed = st.number_input("Graine des codes z", 0, 10_000, 0)
    go = st.button("Générer", type="primary", width="stretch")

# --------------------------------------------------------------------------- contexte
c1, c2, c3, c4 = st.columns(4)
c1.metric("A400 actuel", f"{row['A400_before']:.0%}")
c2.metric(
    "E actuel (avec A400)", "—" if row["E_before"] != row["E_before"] else f"{row['E_before']:.2f}"
)
c3.metric("A400 du glouton avec ce budget", f"{greedy_max:.0%}")
reach = a_target <= greedy_max + 1e-6
c4.metric("Cible atteignable par le glouton", "oui" if reach else "non")
if not reach:
    st.info("Le glouton n'atteint pas cette cible avec ce budget. Le glouton n'étant pas "
            "optimal, c'est une estimation par le bas : le modèle s'en approchera au "
            "mieux.")  # fmt: skip
if int(row["capacity_px"]) < demo.budget_pixels(b_frac, cfg.data):
    st.warning("Budget saturé : la surface aménageable de la tuile est inférieure au budget.")

# --------------------------------------------------------------------------- scénario
if go:
    with st.spinner("Génération et calcul des indicateurs exacts (Dijkstra)…"):
        scenario = demo.run_scenario(cfg, run, str(row["pair_file"]), b_frac, a_target,
                                     n=int(n), seed=int(seed))  # fmt: skip
    st.pyplot(demo.plot_scenario(scenario), width="stretch")
    table = demo.scenario_table(scenario)
    st.dataframe(
        table.style.format({"A400": "{:.1%}", "ΔA400": "{:+.1%}", "E": "{:.2f}", "V": "{:.1%}",
                            "surface ajoutée (m²)": "{:,.0f}", "|â − a|": "{:.3f}"}),
        width="stretch", hide_index=True,
    )  # fmt: skip
    div = demo.diversity(scenario)
    st.caption(f"Diversité des propositions (1 − IoU moyen) : "
               f"{'—' if div != div else f'{div:.3f}'}. E (équité entre secteurs) se lit "
               "toujours avec A400 : une équité élevée peut signifier que tout le monde est "
               "aussi mal desservi.")  # fmt: skip
    st.download_button("Télécharger le tableau (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"{tile_id}_b{b_frac:.2f}_a{a_target:.2f}.csv")  # fmt: skip
else:
    st.write("Choisissez une tuile, un budget et une cible, puis cliquez sur **Générer**.")
