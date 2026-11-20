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


def _config(overrides: list[str] | None) -> DictConfig:
    try:
        return load_config(overrides or [])
    except Exception as exc:  # erreurs Hydra/OmegaConf : message lisible plutôt qu'une trace
        typer.secho(f"Configuration invalide : {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s : %(message)s")


def _not_implemented(command: str, phase: int) -> None:
    typer.secho(
        f"« {command} » n'est pas encore implémentée (prévue en phase {phase}).",
        fg=typer.colors.YELLOW,
        err=True,
    )
    raise typer.Exit(code=1)


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
def demo(overrides: Overrides = None) -> None:
    """Lance l'interface de démonstration."""
    _config(overrides)
    _not_implemented("demo", 7)


if __name__ == "__main__":  # pragma: no cover
    app()
