"""Interface en ligne de commande ``happycity``.

Chaque commande accepte des surcharges Hydra en arguments libres, par exemple ::

    happycity show-config model=A data.source=synthetic
    happycity train model=C ablation=no_attention paths=kaggle

L'interface Streamlit et les notebooks appellent les mêmes fonctions que ces commandes.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Annotated

import typer
from omegaconf import DictConfig, OmegaConf

from happycity import __version__
from happycity.config import load_config

# Console Windows (cp1252) redirigée : remplacer les caractères non encodables (✓, →)
# plutôt que de planter APRÈS un calcul long.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(errors="replace")

app = typer.Typer(
    name="happycity",
    help="HappyCity Gen : scénarios d'espaces verts et accessibilité à 400 m (Marrakech).",
    no_args_is_help=True,
    add_completion=False,
)

Overrides = Annotated[
    list[str] | None,
    typer.Argument(help="Surcharges Hydra, par ex. model=A data.source=synthetic."),
]


def _batch_plots() -> None:
    """Figures écrites sur disque, sans fenêtre : la CLI tourne aussi sans écran.

    Réglé ici, et non dans les modules de figures, pour que le notebook et Streamlit
    gardent leur propre moteur d'affichage.
    """
    import matplotlib

    matplotlib.use("Agg")


def _config(overrides: list[str] | None) -> DictConfig:
    _batch_plots()
    try:
        return load_config(overrides or [])
    except Exception as exc:  # erreurs Hydra/OmegaConf : message lisible plutôt qu'une trace
        typer.secho(f"Configuration invalide : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")


@app.command()
def version() -> None:
    """Affiche la version du paquet."""
    typer.echo(__version__)


@app.command("show-config")
def show_config(overrides: Overrides = None) -> None:
    """Affiche la configuration composée et validée."""
    typer.echo(OmegaConf.to_yaml(_config(overrides)))


@app.command()
def extract(
    overrides: Overrides = None,
    district: Annotated[
        list[str] | None, typer.Option("--district", "-d", help="Zone(s) à extraire.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Ré-extraire même si déjà présent.")] = False,
) -> None:
    """Extrait les couches OSM et le graphe piéton des quartiers (réseau requis)."""
    from happycity.data.pipeline import run_extract

    cfg = _config(overrides)
    _logging()
    for path in run_extract(cfg, district, force=force):
        typer.echo(f"✓ {path}")


@app.command()
def synthesize(
    overrides: Overrides = None,
    n: Annotated[int | None, typer.Option(help="Nombre de cartes (défaut : config).")] = None,
    start: Annotated[int, typer.Option(help="Premier indice.")] = 0,
    workers: Annotated[int | None, typer.Option(help="Processus parallèles.")] = None,
) -> None:
    """Génère des cartes synthétiques procédurales (source=synthetic)."""
    from happycity.data.pipeline import run_synthesize

    cfg = _config(overrides)
    paths = run_synthesize(cfg, n, start, workers)
    typer.echo(f"✓ {len(paths)} cartes synthétiques")


@app.command("coverage-report")
def coverage_report(
    overrides: Overrides = None,
    synthetic_sample: Annotated[int, typer.Option(help="Cartes synthétiques analysées.")] = 60,
) -> None:
    """Produit le rapport de couverture OSM / annotations et le calibrage synthétique."""
    from happycity.data.pipeline import run_coverage_report

    cfg = _config(overrides)
    out = run_coverage_report(cfg, synthetic_sample)
    typer.echo(f"✓ rapport : {out / 'coverage.md'}")


@app.command()
def rasterize(
    overrides: Overrides = None,
    workers: Annotated[int | None, typer.Option(help="Processus parallèles.")] = None,
) -> None:
    """Rastérise les couches en tuiles 128×128 + couverture des annotations."""
    from happycity.data.pipeline import run_rasterize

    cfg = _config(overrides)
    out = run_rasterize(cfg, workers)
    typer.echo(f"✓ tuiles : {out / 'tiles.parquet'} ; rapport : {out / 'annotation_coverage.md'}")


@app.command()
def access(
    overrides: Overrides = None,
    workers: Annotated[int | None, typer.Option(help="Processus parallèles.")] = None,
) -> None:
    """Calcule l'état initial (V, A400, E) et le canal unserved_origins de chaque tuile."""
    from happycity.data.pipeline import run_access

    cfg = _config(overrides)
    out = run_access(cfg, workers)
    typer.echo(f"✓ état initial : {out / 'access_before.md'}")


@app.command("make-pairs")
def make_pairs(
    overrides: Overrides = None,
    workers: Annotated[int | None, typer.Option(help="Processus parallèles.")] = None,
) -> None:
    """Génère les paires avant/après (glouton stochastique) et évalue les références."""
    from happycity.data.pipeline import run_make_pairs

    cfg = _config(overrides)
    out = run_make_pairs(cfg, workers)
    typer.echo(f"✓ paires : {out / 'pairs.parquet'} ; rapport : {out / 'pairs_report.md'}")


@app.command("make-dataset")
def make_dataset(
    overrides: Overrides = None,
    workers: Annotated[int | None, typer.Option(help="Processus parallèles.")] = None,
) -> None:
    """Assemble le jeu de données compact d'entraînement (un .npz par split)."""
    from happycity.data.pipeline import run_make_dataset

    cfg = _config(overrides)
    out = run_make_dataset(cfg, workers)
    typer.echo(f"✓ jeu de données : {out} ; rapport : {out / 'dataset_report.md'}")


@app.command()
def train(overrides: Overrides = None) -> None:
    """Entraîne une variante (A, B, C) ou une ablation (par ex. model=C ablation=no_attention)."""
    from happycity.training.trainer import train as run_train

    cfg = _config(overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = run_train(cfg)
    typer.echo(f"✓ exécution {cfg.run_name} : {out}")


@app.command()
def evaluate(
    overrides: Overrides = None,
    runs: Annotated[str, typer.Option(help="Motif des exécutions (dossiers d'experiments).")] = "*",
    splits: Annotated[str, typer.Option(help="Splits évalués, séparés par des virgules.")] = (
        "test_synth,test"
    ),
    workers: Annotated[int | None, typer.Option(help="Processus parallèles.")] = None,
    device: Annotated[str | None, typer.Option(help="cpu ou cuda (défaut : auto).")] = None,
) -> None:
    """Compare les variantes, les ablations et les références (métriques exactes)."""
    from happycity.evaluation.pipeline import run_evaluate

    cfg = _config(overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = run_evaluate(
        cfg, runs, [s.strip() for s in splits.split(",") if s.strip()], workers, device
    )
    typer.echo(f"✓ évaluation : {out / 'report.md'}")


@app.command()
def demo(
    overrides: Overrides = None,
    tile: Annotated[str, typer.Option(help="Tuile, par ex. gueliz_11 ou synth_00545.")] = (
        "gueliz_11"
    ),
    budget: Annotated[float, typer.Option(help="Budget b (fraction de la tuile).")] = 0.02,
    target: Annotated[
        float | None, typer.Option(help="Cible a (défaut : A400 du glouton).")
    ] = None,
    n: Annotated[int, typer.Option(help="Nombre de propositions.")] = 3,
    seed: Annotated[int, typer.Option(help="Graine des codes z.")] = 0,
    run: Annotated[str | None, typer.Option(help="Exécution (défaut : C complet).")] = None,
    ui: Annotated[bool, typer.Option("--ui", help="Lance l'interface Streamlit.")] = False,
) -> None:
    """Scénario d'aménagement sur une tuile (figure + tableau), ou interface Streamlit (--ui)."""
    if ui:
        import subprocess
        import sys
        from pathlib import Path

        app_file = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
        args = [sys.executable, "-m", "streamlit", "run", str(app_file), "--", *(overrides or [])]
        raise typer.Exit(subprocess.call(args))

    from pathlib import Path

    from happycity import demo as d

    cfg = _config(overrides)
    runs = d.available_runs(cfg)
    chosen = next((r for r in runs if r.name == run), None) if run else d.default_run(runs)
    if chosen is None:
        typer.secho(
            "Aucun modèle entraîné trouvé dans experiments/.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    tiles = d.available_tiles(cfg).set_index("tile_id")
    if tile not in tiles.index:
        typer.secho(f"Tuile inconnue : {tile}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    row = tiles.loc[tile].to_dict()
    a = float(target) if target is not None else float(row[f"greedy_{budget:.2f}"])
    scenario = d.run_scenario(cfg, chosen, str(row["pair_file"]), budget, a, n=n, seed=seed)
    out = Path(cfg.paths.experiments_dir) / "demo"
    stem = f"{tile}_b{budget:.2f}_a{a:.2f}_{chosen.name}"
    d.plot_scenario(scenario, out / f"{stem}.png")
    table = d.scenario_table(scenario)
    table.to_csv(out / f"{stem}.csv", index=False)
    typer.echo(table.round(3).to_string(index=False))
    reach = "oui" if scenario.reachable else "non"
    typer.echo(f"cible atteignable par le glouton : {reach} (glouton : {scenario.greedy_max:.1%})"
               f" · diversité : {d.diversity(scenario):.3f}")  # fmt: skip
    typer.echo(d.DISCLAIMER)
    typer.echo(f"✓ figure : {out / f'{stem}.png'}")


if __name__ == "__main__":  # pragma: no cover
    app()
