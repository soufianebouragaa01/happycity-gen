"""Règles d'architecture : indépendance du module de métriques."""

from __future__ import annotations

import ast
from pathlib import Path

import happycity

PACKAGE_DIR = Path(happycity.__file__).parent
FORBIDDEN_IN_METRICS = (
    "torch",
    "happycity.models",
    "happycity.training",
    "happycity.baselines",
    "happycity.projection",
    "happycity.evaluation",
    "happycity.cli",
    "happycity.data",  # le module de métriques est autonome : tableaux et graphe en entrée
    "happycity.config",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # import relatif : interdit de remonter hors de metrics/
                assert node.level == 1, f"{path}: import relatif hors de metrics/"
            elif node.module:
                modules.add(node.module)
    return modules


def test_metrics_do_not_depend_on_models() -> None:
    files = list((PACKAGE_DIR / "metrics").rglob("*.py"))
    assert files, "le paquet metrics doit exister"
    for path in files:
        for module in _imported_modules(path):
            for forbidden in FORBIDDEN_IN_METRICS:
                is_forbidden = module == forbidden or module.startswith(forbidden + ".")
                assert not is_forbidden, f"{path.name} importe {module}"


def test_training_does_not_need_the_geo_stack() -> None:
    """L'entraînement (Kaggle) n'importe ni geopandas, ni OSMnx, ni rasterio."""
    import subprocess
    import sys

    code = (
        "import sys, happycity.training.trainer, happycity.models.inference, happycity.cli;"
        "heavy = [m for m in ('geopandas', 'osmnx', 'rasterio', 'osmium') if m in sys.modules];"
        "print(heavy); sys.exit(1 if heavy else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
