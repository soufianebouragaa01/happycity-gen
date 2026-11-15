"""Étapes du pipeline de données, appelées par la CLI, les notebooks et l'interface."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from shapely import wkt
from shapely.geometry import Polygon

from happycity.data import coverage
from happycity.data.access_context import build_context, metric_params, with_channel
from happycity.data.dataset import pack_split, save_split, service_patterns, soft_a400
from happycity.data.layers import UrbanLayers
from happycity.data.osm import (
    extraction_area,
    fetch_overpass,
    layers_from_osm,
    load_zone,
    read_osm_xml,
)
from happycity.data.pairs import synthetic_split, tile_pairs
from happycity.data.raster import (
    TileRaster,
    TileSpec,
    rasterize_tile,
    split_conflicts,
    tile_annotations,
    tile_grid,
)
from happycity.data.synthetic import generate_map
from happycity.metrics import TileAccessContext, evaluate_tile, served_origins, unserved_raster

log = logging.getLogger(__name__)
T = TypeVar("T")


def osm_dir(cfg: DictConfig) -> Path:
    """Dossier des zones réelles extraites."""
    return Path(cfg.paths.interim_dir) / "osm"


def synthetic_dir(cfg: DictConfig) -> Path:
    """Dossier des cartes synthétiques."""
    return Path(cfg.paths.interim_dir) / "synthetic"


def coverage_dir(cfg: DictConfig) -> Path:
    """Dossier du rapport de couverture."""
    return Path(cfg.paths.interim_dir) / "coverage"


def split_of(cfg: DictConfig) -> dict[str, str]:
    """Quartier → split (train / val / test)."""
    split = cfg.data.split
    return {name: part for part in ("train", "val", "test") for name in split[part]}


def run_extract(
    cfg: DictConfig, districts: Sequence[str] | None = None, force: bool = False
) -> list[Path]:
    """Extrait les zones demandées (toutes par défaut) ; ignore celles déjà extraites."""
    wanted = list(districts) if districts else [d.name for d in cfg.data.districts]
    known = {d.name: d for d in cfg.data.districts}
    unknown = set(wanted) - set(known)
    if unknown:
        raise ValueError(f"zones inconnues : {sorted(unknown)}")
    todo = [n for n in wanted if force or not (osm_dir(cfg) / n / "meta.json").exists()]
    zones: dict[str, Polygon] = {}
    for name in todo:
        district = known[name]
        if district.boundary_file is None:
            raise ValueError(
                f"{name} : aucune limite OSM exploitable ; renseigner boundary_file "
                "(voir docs/boundaries.md)"
            )
        zones[name] = load_zone(
            Path(cfg.paths.boundaries_dir) / district.boundary_file, cfg.data.crs
        )
    areas = {name: extraction_area(zone, cfg.data) for name, zone in zones.items()}
    extraction = cfg.data.extraction
    raw_osm = Path(cfg.paths.raw_dir) / "osm"

    if extraction.source == "pbf" and areas:
        from happycity.data.pbf import clip_pbf, download_pbf, snapshot_info  # osmium : DLL native

        pbf_path = download_pbf(
            extraction.pbf_url, raw_osm / extraction.pbf_file, extraction.user_agent
        )
        log.info("découpage de %s pour %s…", pbf_path.name, ", ".join(areas))
        clips = clip_pbf(pbf_path, areas, raw_osm / "clips", cfg.data.network.network_type)
        source_meta: dict[str, object] = {"osm_source": "pbf", **snapshot_info(pbf_path)}
    elif extraction.source == "overpass":
        source_meta = {"osm_source": "overpass", "url": extraction.overpass_url}
    elif areas:
        raise ValueError(f"source OSM inconnue : {extraction.source!r}")

    for name, zone in zones.items():
        log.info("construction des couches de %s…", name)
        if extraction.source == "pbf":
            features, graph = read_osm_xml(*clips[name], areas[name], cfg.data)
        else:
            features, graph = fetch_overpass(
                areas[name], cfg.data, Path(cfg.paths.raw_dir) / "osmnx_cache"
            )
        layers = layers_from_osm(name, zone, features, graph, cfg.data)
        layers.meta.update(source_meta, boundary_file=known[name].boundary_file)
        layers.save(osm_dir(cfg) / name)
    return [osm_dir(cfg) / n for n in wanted]


def _synthesize_one(job: tuple[DictConfig, int]) -> Path:
    cfg, index = job
    layers = generate_map(index, cfg.data)
    target = synthetic_dir(cfg) / layers.name
    layers.save(target)
    return target


def _pool_map(fn: Callable[[Any], T], jobs: list[Any], workers: int) -> list[T]:
    """``map`` séquentiel (workers ≤ 1) ou multi-processus, ordre conservé."""
    if workers <= 1 or len(jobs) <= 1:
        return [fn(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs, chunksize=max(1, len(jobs) // (4 * workers))))


def run_synthesize(
    cfg: DictConfig, n: int | None = None, start: int = 0, workers: int | None = None
) -> list[Path]:
    """Génère ``n`` cartes synthétiques (indices ``start…start+n-1``), en parallèle."""
    count = int(cfg.data.synthetic.n_maps) if n is None else n
    jobs = [(cfg, index) for index in range(start, start + count)]
    return _pool_map(_synthesize_one, jobs, int(workers or cfg.data.raster.workers))


def load_zones(directory: Path, limit: int | None = None) -> list[UrbanLayers]:
    """Relit toutes les zones d'un dossier (triées par nom)."""
    if not directory.exists():
        return []
    dirs = sorted(p for p in directory.iterdir() if (p / "meta.json").exists())
    return [UrbanLayers.load(p) for p in dirs[:limit]]


def run_coverage_report(cfg: DictConfig, synthetic_sample: int = 60) -> Path:
    """Rapport de couverture : tableau CSV/Markdown, cartes de contrôle, calibrage synthétique."""
    out = coverage_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    splits = split_of(cfg)
    real = load_zones(osm_dir(cfg))
    sections = ["# Rapport de couverture\n"]

    if real:
        table = coverage.coverage_table(real, cfg.data.coverage)
        table.insert(1, "split", table["zone"].map(splits))
        table.to_csv(out / "coverage_real.csv", index=False)
        gaps = coverage.min_gaps(real, splits)
        gaps["buffer_ok"] = (gaps["split_a"] == gaps["split_b"]) | (
            gaps["gap_m"] >= float(cfg.data.split.border_buffer_m)
        )
        gaps.to_csv(out / "zone_gaps.csv", index=False)
        for layers in real:
            coverage.plot_layers(layers, out / f"zone_{layers.name}.png")
        coverage.plot_overview(real, splits, out / "overview.png")
        sections += [
            "## Zones réelles\n",
            coverage.to_markdown(
                table.set_index("zone").T.reset_index().rename(columns={"index": "indicateur"})
            ),
            "\n## Distances entre zones\n",
            coverage.to_markdown(gaps),
            "\n## Espaces verts comptés\n",
        ]
        for layers in real:
            names = sorted({str(n) for n in layers.green["name"].dropna()})
            sections.append(f"- **{layers.name}** : {', '.join(names) or '(sans nom)'}")
        sections.append("\n## Espaces verts exclus (audit)\n")
        for layers in real:
            excl = layers.green_excluded
            items = [f"{r.name or '(sans nom)'} [{r.reason}]" for r in excl.itertuples()]
            sections.append(f"- **{layers.name}** : {'; '.join(items) or '—'}")

    synth = load_zones(synthetic_dir(cfg), limit=synthetic_sample)
    if synth:
        stable = coverage.coverage_table(synth, cfg.data.coverage)
        stable["typology"] = [s.meta["typology"] for s in synth]
        stable.to_csv(out / "coverage_synthetic_sample.csv", index=False)
        numeric = [
            "built_ratio",
            "walk_density_km_per_km2",
            "V",
            "n_green",
            "n_candidates",
            "candidates_ha",
            "n_origins",
        ]
        calib = stable.groupby("typology")[numeric].mean().round(3).reset_index()
        if real:
            density = table[["zone", *numeric]].copy()
            for col in ("n_green", "n_candidates", "candidates_ha", "n_origins"):
                density[col] = (density[col] / table["area_km2"]).round(2)
            for col in ("n_green", "n_candidates", "candidates_ha", "n_origins"):
                calib[col] = (calib[col] / stable["area_km2"].mean()).round(2)
            calib = pd.concat([density.rename(columns={"zone": "typology"}), calib])
        calib.to_csv(out / "calibration.csv", index=False)
        for typ in sorted(set(stable["typology"])):
            example = next(s for s in synth if s.meta["typology"] == typ)
            coverage.plot_layers(example, out / f"synthetic_{typ}.png")
        sections += [
            "\n## Calibrage synthétique (valeurs par km² pour les effectifs)\n",
            coverage.to_markdown(calib),
        ]

    (out / "coverage.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- rastérisation


def tiles_dir(cfg: DictConfig) -> Path:
    """Dossier des tuiles rastérisées."""
    return Path(cfg.paths.processed_dir) / "tiles"


def _zone_header(directory: Path) -> tuple[str, Polygon]:
    header = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    zone = wkt.loads(header["zone_wkt"])
    assert isinstance(zone, Polygon)
    return str(header["name"]), zone


RasterJob = tuple[DictConfig, Path, str, str, dict[str, tuple[Polygon, str]]]


def _rasterize_zone(job: RasterJob) -> list[dict[str, Any]]:
    cfg, directory, source, split, zones = job
    layers = UrbanLayers.load(directory)
    size_m = float(cfg.data.tile.size_px) * float(cfg.data.tile.m_per_px)
    if source == "osm":
        grid = tile_grid(layers.zone, size_m, float(cfg.data.raster.real_stride_m))
        specs = [
            TileSpec(f"{layers.name}_{i}{j}", source, layers.name, split, x, y, size_m)
            for i, j, x, y in grid
        ]
    else:
        xmin, ymin, _, _ = layers.zone.bounds
        specs = [TileSpec(layers.name, source, layers.name, split, xmin, ymin, size_m)]
    buffer_m = float(cfg.data.split.border_buffer_m)
    rows = []
    for spec in specs:
        raster = rasterize_tile(layers, spec, cfg.data)
        row = tile_annotations(layers, raster, cfg.data)
        conflicts = split_conflicts(spec.geometry, split, zones, buffer_m) if zones else []
        if conflicts:
            reason = "tampon: " + "/".join(conflicts)
            row["flags"] = ", ".join(f for f in (row["flags"], reason) if f)
            row["usable"] = False
        row.update(
            xmin=spec.xmin, ymin=spec.ymin, size_m=spec.size_m, excluded_split=bool(conflicts)
        )
        row["file"] = None
        if not conflicts:
            path = tiles_dir(cfg) / f"{spec.tile_id}.npz"
            raster.save(path)
            row["file"] = path.name
        rows.append(row)
    return rows


def run_rasterize(cfg: DictConfig, workers: int | None = None) -> Path:
    """Rastérise les zones réelles et/ou synthétiques selon ``data.source``.

    Écrit ``tiles/<tile_id>.npz``, l'index ``tiles.parquet`` (une ligne par tuile, avec
    la couverture des annotations) et des figures de contrôle.
    """
    source = str(cfg.data.source)
    splits = split_of(cfg)
    jobs: list[RasterJob] = []
    if source in ("real", "mixed"):
        real_dirs = sorted(p for p in osm_dir(cfg).glob("*") if (p / "meta.json").exists())
        zones: dict[str, tuple[Polygon, str]] = {}
        for d in real_dirs:
            name, zone = _zone_header(d)
            zones[name] = (zone, splits.get(name, "unassigned"))
        jobs += [(cfg, d, "osm", zones[d.name][1], zones) for d in real_dirs]
    if source in ("synthetic", "mixed"):
        synth_split = str(cfg.data.split.synthetic_in[0])
        synth_dirs = sorted(p for p in synthetic_dir(cfg).glob("*") if (p / "meta.json").exists())
        jobs += [(cfg, d, "synthetic", synth_split, {}) for d in synth_dirs]
    if not jobs:
        raise FileNotFoundError("aucune zone à rastériser : lancer extract et/ou synthesize")
    n_workers = int(workers or cfg.data.raster.workers)
    rows = [r for part in _pool_map(_rasterize_zone, jobs, n_workers) for r in part]
    index = pd.DataFrame(rows)
    out = Path(cfg.paths.processed_dir)
    out.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out / "tiles.parquet", index=False)
    index.to_csv(out / "annotation_coverage.csv", index=False)
    _annotation_summary(index, out)
    _tile_figures(cfg, index, out / "figures")
    return out


def _annotation_summary(index: pd.DataFrame, out: Path) -> None:
    """Tableau de couverture des annotations agrégé par source / zone (Markdown)."""
    real = index[index["source"] == "osm"]
    synth = index[index["source"] == "synthetic"]
    agg = {
        "tuiles": ("tile_id", "count"),
        "exploitables": ("usable", "sum"),
        "origines_moy": ("n_origins", "mean"),
        "parcelles_moy": ("n_candidates", "mean"),
        "parcelles_adj_moy": ("n_candidates_adjacent", "mean"),
        "verts_accessibles_moy": ("n_green_with_entrance", "mean"),
        "reseau_px_moy": ("walk_px_share", "mean"),
    }
    parts = ["# Couverture des annotations par tuile", ""]
    if not real.empty:
        by_zone = real.groupby(["split", "zone"]).agg(**agg).round(2).reset_index()
        detail_cols = [
            "tile_id",
            "split",
            "n_origins",
            "n_candidates",
            "n_candidates_adjacent",
            "n_green_with_entrance",
            "usable",
            "flags",
        ]
        parts += ["## Tuiles réelles", "", coverage.to_markdown(by_zone), ""]
        parts += ["### Détail", "", coverage.to_markdown(real[detail_cols]), ""]
    if not synth.empty:
        by_src = synth.assign(zone="synthetic").groupby("zone").agg(**agg).round(2).reset_index()
        flags = synth["flags"].replace("", "—").value_counts().rename_axis("flags").reset_index()
        parts += ["## Tuiles synthétiques", "", coverage.to_markdown(by_src), ""]
        parts += [coverage.to_markdown(flags), ""]
    (out / "annotation_coverage.md").write_text("\n".join(parts), encoding="utf-8")


def _tile_figures(cfg: DictConfig, index: pd.DataFrame, out: Path) -> None:
    """Planches de contrôle visuel : quelques tuiles réelles et synthétiques."""
    saved = index[index["file"].notna()]
    groups = {
        "real": (saved[saved["source"] == "osm"].groupby("zone").head(2), osm_dir(cfg)),
        "synthetic": (saved[saved["source"] == "synthetic"].head(4), synthetic_dir(cfg)),
    }
    for name, (sel, source_dir) in groups.items():
        if sel.empty:
            continue
        tiles = [TileRaster.load(tiles_dir(cfg) / f) for f in sel["file"]]
        layers = {t.spec.zone: UrbanLayers.load(source_dir / t.spec.zone) for t in tiles}
        coverage.plot_tiles(tiles, [layers[t.spec.zone] for t in tiles], out / f"tiles_{name}.png")


# --------------------------------------------------------------------------- accessibilité


def context_dir(cfg: DictConfig) -> Path:
    """Dossier des contextes d'évaluation (un ``.npz`` par tuile)."""
    return Path(cfg.paths.processed_dir) / "context"


AccessJob = tuple[DictConfig, Path, list[str]]


def _access_zone(job: AccessJob) -> list[dict[str, Any]]:
    cfg, zone_dir, files = job
    layers = UrbanLayers.load(zone_dir)
    params = metric_params(cfg.data)
    rows = []
    for name in files:
        tile = TileRaster.load(tiles_dir(cfg) / name)
        ctx = build_context(layers, tile, cfg.data)
        ctx.save(context_dir(cfg) / name)
        served = served_origins(ctx, params)
        scores = evaluate_tile(ctx, params=params)
        with_channel(tile, "unserved_origins", unserved_raster(ctx, served)).save(
            tiles_dir(cfg) / name
        )
        rows.append(
            {
                "tile_id": tile.spec.tile_id,
                "V_before": scores.V,
                "A400_before": scores.A400,
                "E_before": scores.E,
                "n_served_before": scores.n_served,
                "n_valid_sectors": scores.n_valid_sectors,
                "sector_a400_before": list(scores.sector_a400),
                "n_access_points": len(ctx.existing),
                "graph_nodes": ctx.walk.n_nodes,
            }
        )
    return rows


def run_access(cfg: DictConfig, workers: int | None = None) -> Path:
    """État initial de chaque tuile : contexte, V/A400/E avant, canal ``unserved_origins``."""
    out = Path(cfg.paths.processed_dir)
    index = pd.read_parquet(out / "tiles.parquet")
    saved = index[index["file"].notna()]
    jobs: list[AccessJob] = []
    for (source, zone), group in saved.groupby(["source", "zone"]):
        base = osm_dir(cfg) if source == "osm" else synthetic_dir(cfg)
        jobs.append((cfg, base / str(zone), list(group["file"])))
    n_workers = int(workers or cfg.data.raster.workers)
    rows = [r for part in _pool_map(_access_zone, jobs, n_workers) for r in part]
    access = pd.DataFrame(rows)
    keep = [c for c in index.columns if c not in access.columns or c == "tile_id"]
    index = index[keep].merge(access, on="tile_id", how="left")
    index.to_parquet(out / "tiles.parquet", index=False)
    index.drop(columns=["sector_a400_before"]).to_csv(out / "annotation_coverage.csv", index=False)
    _access_summary(cfg, index, out)
    return out


def district_scores(cfg: DictConfig, index: pd.DataFrame, zone: str) -> dict[str, Any]:
    """A400 et E d'une zone réelle sur ses 4 tuiles disjointes (secteurs = tuiles)."""
    params = metric_params(cfg.data)
    disjoint = index[(index["zone"] == zone) & index["tile_id"].str.match(rf"{zone}_[02][02]$")]
    disjoint = disjoint[disjoint["file"].notna()]
    served = disjoint["n_served_before"].fillna(0).to_numpy()
    counts = disjoint["n_origins"].to_numpy()
    valid = counts >= params.min_origins
    rates = served[valid] / counts[valid]
    return {
        "zone": zone,
        "tuiles_disjointes": len(disjoint),
        "origines": int(counts.sum()),
        "A400": round(float(served.sum() / max(counts.sum(), 1)), 3),
        "E_tuiles": round(float(1 - (rates.max() - rates.min())), 3) if valid.sum() >= 2 else None,
        "A400_min_tuile": round(float(rates.min()), 3) if valid.any() else None,
        "A400_max_tuile": round(float(rates.max()), 3) if valid.any() else None,
    }


def _access_summary(cfg: DictConfig, index: pd.DataFrame, out: Path) -> None:
    """Rapport de l'état initial (avant aménagement)."""
    done = index[index["A400_before"].notna()]
    stats = {
        "tuiles": ("tile_id", "count"),
        "A400_moy": ("A400_before", "mean"),
        "A400_med": ("A400_before", "median"),
        "E_moy": ("E_before", "mean"),
        "E_indefini": ("E_before", lambda s: int(s.isna().sum())),
        "V_moy": ("V_before", "mean"),
    }
    parts = ["# État initial : V, A400, E avant aménagement", ""]
    real = done[done["source"] == "osm"]
    if not real.empty:
        per_zone = real.groupby(["split", "zone"]).agg(**stats).round(3).reset_index()
        districts = pd.DataFrame(
            [district_scores(cfg, index, z) for z in sorted(set(real["zone"]))]
        )
        cols = ["tile_id", "split", "n_origins", "A400_before", "E_before", "V_before", "usable"]
        parts += ["## Tuiles réelles (moyennes par zone)", "", coverage.to_markdown(per_zone), ""]
        parts += ["## Quartiers (4 tuiles disjointes ; E entre tuiles)", ""]
        parts += [coverage.to_markdown(districts), ""]
        detail = real[cols].round(3)
        parts += ["### Détail par tuile", "", coverage.to_markdown(detail), ""]
    synth = done[done["source"] == "synthetic"]
    if not synth.empty:
        agg = synth.assign(zone="synthetic").groupby("zone").agg(**stats).round(3).reset_index()
        hist = np.histogram(synth["A400_before"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0001])[0]
        bins = pd.DataFrame(
            {"A400": ["0–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"], "tuiles": hist}
        )
        parts += ["## Tuiles synthétiques", "", coverage.to_markdown(agg), ""]
        parts += [coverage.to_markdown(bins), ""]
    (out / "access_before.md").write_text("\n".join(parts), encoding="utf-8")
    examples = [t for t in ("gueliz_10", "massira_21", "synth_00002") if t in set(done["tile_id"])]
    for tile_id in examples:
        ctx = TileAccessContext.load(context_dir(cfg) / f"{tile_id}.npz")
        served = served_origins(ctx, metric_params(cfg.data))
        title = f"{tile_id} : A400 = {served.mean():.0%} ({served.sum()}/{len(served)} origines)"
        coverage.plot_access(ctx, served, title, out / "figures" / f"access_{tile_id}.png")


# --------------------------------------------------------------------------- paires


def pairs_dir(cfg: DictConfig) -> Path:
    """Dossier des masques cibles (un ``.npz`` par tuile)."""
    return Path(cfg.paths.processed_dir) / "pairs"


PairJob = tuple[DictConfig, list[tuple[str, str]]]


def _pairs_chunk(job: PairJob) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg, items = job
    params = metric_params(cfg.data)
    all_pairs, all_baselines = [], []
    for name, split in items:
        tile = TileRaster.load(tiles_dir(cfg) / name)
        ctx = TileAccessContext.load(context_dir(cfg) / name)
        pairs, masks, baselines = tile_pairs(tile, ctx, split, cfg.data, params)
        stack = np.stack(masks) if masks else np.zeros((0, *tile.candidate_ids.shape), bool)
        out = pairs_dir(cfg) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, masks=np.packbits(stack, axis=-1), shape=np.array(stack.shape))
        for k, row in enumerate(pairs):
            row["mask_index"] = k
            row["pair_file"] = name
        all_pairs += pairs
        all_baselines += baselines
    return all_pairs, all_baselines


def load_pair_masks(cfg: DictConfig, pair_file: str) -> np.ndarray:
    """Masques cibles d'une tuile, ``(P, H, W)`` booléen."""
    with np.load(pairs_dir(cfg) / pair_file) as data:
        shape = tuple(int(x) for x in data["shape"])
        return np.unpackbits(data["masks"], axis=-1)[..., : shape[-1]].astype(bool)


def run_make_pairs(cfg: DictConfig, workers: int | None = None) -> Path:
    """Paires avant/après + références (aléatoire, glouton) pour toutes les tuiles exploitables."""
    out = Path(cfg.paths.processed_dir)
    index = pd.read_parquet(out / "tiles.parquet")
    usable = index[index["usable"].astype(bool) & index["file"].notna()].copy()
    frac = float(cfg.data.pairs.synthetic_val_fraction)
    test_frac = float(cfg.data.pairs.synthetic_test_fraction)
    usable["pair_split"] = [
        synthetic_split(t, frac, test_frac) if src == "synthetic" else sp
        for t, src, sp in zip(usable["tile_id"], usable["source"], usable["split"], strict=True)
    ]
    check_split_integrity(usable, cfg)
    items = list(zip(usable["file"], usable["pair_split"], strict=True))
    n_workers = int(workers or cfg.data.raster.workers)
    chunks = [items[i :: max(n_workers, 1) * 4] for i in range(max(n_workers, 1) * 4)]
    jobs: list[PairJob] = [(cfg, c) for c in chunks if c]
    results = _pool_map(_pairs_chunk, jobs, n_workers)
    pairs = pd.DataFrame([r for p, _ in results for r in p])
    baselines = pd.DataFrame([r for _, b in results for r in b])
    pairs = pairs.sort_values(["tile_id", "b_frac", "solution"]).reset_index(drop=True)
    pairs.insert(
        0,
        "pair_id",
        [
            f"{t}_b{b:.2f}_s{s}"
            for t, b, s in zip(pairs["tile_id"], pairs["b_frac"], pairs["solution"], strict=True)
        ],
    )
    pairs.to_parquet(out / "pairs.parquet", index=False)
    baselines.to_parquet(out / "baselines.parquet", index=False)
    _pairs_report(pairs, baselines, out)
    _pair_figures(cfg, pairs, out / "figures")
    return out


def check_split_integrity(tiles: pd.DataFrame, cfg: DictConfig) -> None:
    """Vérifie le découpage AVANT de créer les variantes (lève ``ValueError`` sinon).

    * une tuile n'appartient qu'à un split ;
    * une tuile réelle est dans le split de son quartier ;
    * aucune tuile synthétique hors de ``train`` / ``val_synth`` / ``test_synth``.
    """
    if tiles["tile_id"].duplicated().any():
        raise ValueError("tuile présente plusieurs fois dans l'index")
    expected = split_of(cfg)
    real = tiles[tiles["source"] == "osm"]
    wrong = real[real["zone"].map(expected) != real["pair_split"]]
    if not wrong.empty:
        raise ValueError(
            f"tuiles réelles hors du split de leur quartier : {list(wrong['tile_id'])}"
        )
    synth = tiles[tiles["source"] == "synthetic"]
    if not set(synth["pair_split"]) <= {"train", "val_synth", "test_synth"}:
        raise ValueError("une tuile synthétique est affectée à val/test réel")


def _pair_figures(cfg: DictConfig, pairs: pd.DataFrame, out: Path) -> None:
    """Exemples de paires : une tuile réelle et une tuile synthétique, budget 2 %."""
    kept = pairs[(pairs["excluded"] == "") & (pairs["b_frac"] == 0.02)]
    for source in ("osm", "synthetic"):
        multi = kept[(kept["source"] == source) & (kept["n_unique_solutions"] >= 2)]
        strong = multi[multi["a_target"] - multi["A400_before"] >= 0.2]
        multi = strong if not strong.empty else multi
        if multi.empty:
            continue
        tile_id = str(multi.sort_values("diversity_1_iou", ascending=False)["tile_id"].iloc[0])
        rows = multi[multi["tile_id"] == tile_id].head(2)
        masks = load_pair_masks(cfg, str(rows["pair_file"].iloc[0]))
        tile = TileRaster.load(tiles_dir(cfg) / str(rows["pair_file"].iloc[0]))
        titles = [
            f"solution {r.solution} : A400 {r.A400_before:.0%} → {r.a_target:.0%}"
            for r in rows.itertuples()
        ]
        coverage.plot_pairs(
            tile, [masks[int(k)] for k in rows["mask_index"]], titles, out / f"pairs_{source}.png"
        )


def _pairs_report(pairs: pd.DataFrame, baselines: pd.DataFrame, out: Path) -> None:
    """Rapport : paires par split, gains d'accès, saturation, diversité, références."""
    kept = pairs[pairs["excluded"] == ""]
    excluded = pairs[pairs["excluded"] != ""]
    by_split = (
        kept.groupby("split")
        .agg(
            tuiles=("tile_id", "nunique"),
            paires=("pair_id", "count"),
            A400_avant=("A400_before", "mean"),
            a_cible=("a_target", "mean"),
            gain_moy=("a_target", lambda s: float((s - kept.loc[s.index, "A400_before"]).mean())),
            satures=("saturated", "mean"),
            diversite=("diversity_1_iou", "mean"),
            solutions_uniques=("n_unique_solutions", "mean"),
        )
        .round(3)
        .reset_index()
    )
    parts = ["# Paires avant/après et références non génératives", ""]
    parts += ["## Paires par split", "", coverage.to_markdown(by_split), ""]
    by_budget = (
        kept.groupby(["b_frac"])
        .agg(
            paires=("pair_id", "count"),
            px_moy=("n_px", "mean"),
            gain_moy=("a_target", lambda s: float((s - kept.loc[s.index, "A400_before"]).mean())),
            satures=("saturated", "mean"),
            diversite=("diversity_1_iou", "mean"),
        )
        .round(3)
        .reset_index()
    )
    parts += ["## Par budget", "", coverage.to_markdown(by_budget), ""]
    per_budget = pairs.drop_duplicates(["tile_id", "b_frac"])
    parts += [
        "Tirages stochastiques écartés (gain < min_relative_gain × meilleur gain) : "
        f"{int(per_budget['low_quality_draws'].sum())} pour {len(per_budget)} tuiles × budgets.",
        "",
    ]
    parts += [
        f"Aménagements exclus (aucun gain d'accès possible) : {len(excluded)} "
        f"(tuile × budget), sur {excluded['tile_id'].nunique()} tuiles.",
        "",
    ]
    real_excl = excluded[excluded["source"] == "osm"]["tile_id"].unique()
    if len(real_excl):
        parts += [
            f"Tuiles réelles où l'accès ne peut pas augmenter : {', '.join(sorted(real_excl))}",
            "",
        ]
    base = baselines.assign(
        group=np.where(
            baselines["source"] == "osm",
            baselines["split"] + " (réel)",
            "synth " + baselines["split"],
        )
    )
    cmp = (
        base.groupby(["group", "b_frac"])
        .agg(
            tuiles=("tile_id", "count"),
            A400_avant=("A400_before", "mean"),
            aleatoire=("random_A400", "mean"),
            glouton=("greedy_A400", "mean"),
            V_aleatoire=("random_V", "mean"),
            V_glouton=("greedy_V", "mean"),
            px_glouton=("greedy_px", "mean"),
        )
        .round(3)
        .reset_index()
    )
    parts += ["## Références au même budget (A400 moyen)", "", coverage.to_markdown(cmp), ""]
    (out / "pairs_report.md").write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------- jeu de données


def dataset_dir(cfg: DictConfig) -> Path:
    """Dossier du jeu de données compact (un ``.npz`` par split)."""
    return Path(cfg.paths.processed_dir) / "dataset"


DatasetJob = tuple[DictConfig, str, list[str]]


def _dataset_tile(job: DatasetJob) -> tuple[str, dict[str, np.ndarray], list[float]]:
    """Motifs de desserte d'une tuile + A400 doux des masques cibles (contrôle)."""
    cfg, name, _ = job
    params = metric_params(cfg.data)
    tile = TileRaster.load(tiles_dir(cfg) / name)
    ctx = TileAccessContext.load(context_dir(cfg) / name)
    patterns = service_patterns(tile, ctx, params)
    soft = [soft_a400(m, patterns) for m in load_pair_masks(cfg, name)]
    return name, patterns, soft


def run_make_dataset(cfg: DictConfig, workers: int | None = None) -> Path:
    """Jeu de données compact par split (entrées, motifs de desserte, paires conservées)."""
    out = dataset_dir(cfg)
    processed = Path(cfg.paths.processed_dir)
    pairs = pd.read_parquet(processed / "pairs.parquet")
    files = sorted(pairs["pair_file"].unique())
    jobs: list[DatasetJob] = [(cfg, f, []) for f in files]
    results = _pool_map(_dataset_tile, jobs, int(workers or cfg.data.raster.workers))
    patterns = {name: p for name, p, _ in results}
    soft = {name: s for name, _, s in results}
    pairs["soft_a400"] = [
        soft[f][int(k)] for f, k in zip(pairs["pair_file"], pairs["mask_index"], strict=True)
    ]
    kept = pairs[pairs["excluded"] == ""].copy()
    summary: dict[str, Any] = {}
    for split, group in kept.groupby("split"):
        names = sorted(group["pair_file"].unique())
        tiles = [(TileRaster.load(tiles_dir(cfg) / n), patterns[n]) for n in names]
        masks = [
            load_pair_masks(cfg, n)[int(k)]
            for n, k in zip(group["pair_file"], group["mask_index"], strict=True)
        ]
        arrays = pack_split(tiles, group.to_dict("records"), masks)
        err = (group["soft_a400"] - group["a_target"]).to_numpy()
        meta = {
            "split": str(split),
            "tiles": len(tiles),
            "pairs": len(group),
            "patterns_mean": float(np.mean([p["patterns"].shape[0] for _, p in tiles])),
            "soft_minus_exact_mean": float(err.mean()),
            "soft_minus_exact_abs_max": float(np.abs(err).max()),
            "soft_exact_within_1pt": float((np.abs(err) <= 0.01).mean()),
        }
        save_split(out / f"{split}.npz", arrays, meta)
        summary[str(split)] = meta
    (out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _dataset_report(kept, summary, out)
    return out


def _dataset_report(kept: pd.DataFrame, summary: dict[str, Any], out: Path) -> None:
    """Rapport : tailles des splits et fidélité de l'A400 doux (proxy de la perte)."""
    table = pd.DataFrame(summary.values()).round(4)
    err = kept["soft_a400"] - kept["a_target"]
    parts = [
        "# Jeu de données d'entraînement",
        "",
        coverage.to_markdown(table),
        "",
        "## Fidélité de l'A400 doux (perte de condition) sur les masques cibles",
        "",
        f"- écart moyen (doux − exact) : {err.mean():+.4f}",
        f"- |écart| médian : {err.abs().median():.4f} ; "
        f"95e centile : {err.abs().quantile(0.95):.4f} ; max : {err.abs().max():.4f}",
        f"- part des cibles à ≤ 1 point : {(err.abs() <= 0.01).mean():.1%}",
        "",
        "L'estimation pixel par pixel ne peut que SURestimer (un point d'accès par pixel et par",
        "tronçon, contre un seul par polygone et par tronçon pour la métrique exacte).",
    ]
    (out / "dataset_report.md").write_text("\n".join(parts), encoding="utf-8")
