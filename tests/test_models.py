"""Jeu de données compact, A400 doux, variantes A/B/C et chaîne d'entraînement (CPU)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from happycity.config import load_config
from happycity.data.access_context import build_context, metric_params, with_channel
from happycity.data.dataset import pack_split, save_split, service_patterns, soft_a400
from happycity.data.layers import UrbanLayers
from happycity.data.pairs import tile_pairs
from happycity.data.raster import TileSpec, rasterize_tile
from happycity.metrics import evaluate_tile, served_origins, unserved_raster
from happycity.models import build_networks, condition_vector
from happycity.models.inference import load_checkpoint, propose
from happycity.training import losses
from happycity.training.data import PairDataset, collate
from happycity.training.trainer import train


@pytest.fixture(scope="module")
def packed(small_synthetic: UrbanLayers, cfg: DictConfig) -> dict[str, Any]:
    xmin, ymin, _, _ = small_synthetic.zone.bounds
    spec = TileSpec(
        small_synthetic.name, "synthetic", small_synthetic.name, "train", xmin, ymin, 1024.0
    )
    tile = rasterize_tile(small_synthetic, spec, cfg.data)
    ctx = build_context(small_synthetic, tile, cfg.data)
    params = metric_params(cfg.data)
    served = served_origins(ctx, params)
    tile = with_channel(tile, "unserved_origins", unserved_raster(ctx, served))  # étape access
    pairs, masks, _ = tile_pairs(tile, ctx, "train", cfg.data, params)
    keep = [i for i, p in enumerate(pairs) if p["excluded"] == ""]
    rows = [{**pairs[i], "pair_id": f"p{i}"} for i in keep]
    patterns = service_patterns(tile, ctx, params)
    arrays = pack_split([(tile, patterns)], rows, [masks[i] for i in keep])
    return {"tile": tile, "ctx": ctx, "params": params, "patterns": patterns,
            "arrays": arrays, "rows": rows, "masks": [masks[i] for i in keep]}  # fmt: skip


@pytest.fixture(scope="module")
def dataset_dir(packed: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dataset")
    for split in ("train", "val_synth"):
        save_split(out / f"{split}.npz", packed["arrays"], {"split": split})
    return out


# --------------------------------------------------------------------------- A400 doux


def test_soft_a400_is_close_to_exact_metric_on_targets(packed: dict[str, Any]) -> None:
    assert packed["rows"], "la scène doit produire des paires"
    for row, mask in zip(packed["rows"], packed["masks"], strict=True):
        exact = evaluate_tile(packed["ctx"], mask, params=packed["params"]).A400
        soft = soft_a400(mask, packed["patterns"])
        assert soft >= exact - 1e-9  # l'estimation pixel par pixel ne sous-estime jamais
        assert soft - exact < 0.05
        assert row["a_target"] == pytest.approx(exact)


def test_soft_a400_of_nothing_is_a400_before(packed: dict[str, Any]) -> None:
    before = evaluate_tile(packed["ctx"], params=packed["params"]).A400
    assert soft_a400(np.zeros((128, 128)), packed["patterns"]) == pytest.approx(before)


def test_torch_soft_a400_matches_numpy(dataset_dir: Path, packed: dict[str, Any]) -> None:
    ds = PairDataset(dataset_dir / "train.npz")
    batch = collate([ds[i] for i in range(min(3, len(ds)))])
    prob = torch.rand(batch["x"].shape[0], 1, 128, 128, generator=torch.Generator().manual_seed(0))
    got = losses.soft_a400(
        prob, batch["pmap"], batch["patterns"], batch["n_origins"], batch["n_base"]
    )
    for i in range(len(got)):
        want = soft_a400(prob[i, 0].numpy(), packed["patterns"])
        assert float(got[i]) == pytest.approx(want, abs=1e-4)
    target = losses.soft_a400(batch["y"], batch["pmap"], batch["patterns"],
                              batch["n_origins"], batch["n_base"])  # fmt: skip
    np.testing.assert_allclose(target.numpy(), [soft_a400(m, packed["patterns"])
                               for m in packed["masks"][: len(got)]], atol=1e-4)  # fmt: skip


def test_augmentation_moves_patterns_with_the_map(dataset_dir: Path) -> None:
    """Une symétrie ne change ni l'A400 doux de la cible ni la région R."""
    plain = PairDataset(dataset_dir / "train.npz")
    aug = PairDataset(dataset_dir / "train.npz", augment=True)
    torch.manual_seed(3)
    for i in range(min(4, len(plain))):
        a, b = plain[i], aug[i]

        def args(it: dict[str, Any]) -> tuple[torch.Tensor, ...]:
            n_o, n_b = float(it["n_origins"]), float(it["n_base"])
            return it["pmap"][None], it["patterns"][None], torch.tensor([n_o]), torch.tensor([n_b])

        sa = losses.soft_a400(a["y"][None], *args(a))
        sb = losses.soft_a400(b["y"][None], *args(b))
        assert float(sa) == pytest.approx(float(sb), abs=1e-5)
        assert float(a["x"][-1].sum()) == pytest.approx(float(b["x"][-1].sum()))
        assert ((b["y"][0] > 0) <= (b["x"][-1] > 0.5)).all()  # la cible reste dans R


# --------------------------------------------------------------------------- variantes


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_variants_shapes_and_projection(variant: str) -> None:
    cfg = load_config([f"model={variant}", "model.ngf=8", "model.ndf=8"])
    nets = build_networks(cfg.model, 8)
    torch.manual_seed(0)
    x = torch.rand(2, 8, 128, 128)
    x[:, -1] = (torch.rand(2, 128, 128) < 0.2).float()
    cond = condition_vector(torch.tensor([0.01, 0.05]), torch.tensor([0.4, 0.9]))
    out = nets.generator(x, cond, nets.generator.sample_z(2))
    assert out.prob.shape == (2, 1, 128, 128)
    assert nets.discriminator(x, cond, out.prob).shape == (2, 1, 14, 14)
    assert nets.encoder is not None
    assert nets.encoder(x, out.prob).shape == (2, 8)
    assert len(out.attention) == (4 if variant == "C" else 0)
    outside = out.prob * (1 - x[:, -1:])
    assert (outside.abs().max() == 0) == (variant == "C")  # C : rien hors de R par construction


@pytest.mark.parametrize("variant", ["A", "B"])
def test_condition_reaches_the_output(variant: str) -> None:
    cfg = load_config([f"model={variant}", "model.ngf=8", "model.dropout=0"])
    gen = build_networks(cfg.model, 8).generator.eval()
    torch.manual_seed(0)
    for p in gen.parameters():  # sortir de l'initialisation identité des modulations
        torch.nn.init.normal_(p, std=0.05)
    x = torch.rand(1, 8, 128, 128)
    z = torch.zeros(1, 8)
    lo = gen(x, condition_vector(torch.tensor([0.01]), torch.tensor([0.3])), z).prob
    hi = gen(x, condition_vector(torch.tensor([0.05]), torch.tensor([0.9])), z).prob
    assert (lo - hi).abs().max() > 1e-4


def test_mode_seeking_is_bounded() -> None:
    p = torch.full((2, 1, 8, 8), 0.5)
    region = torch.ones_like(p)
    loss = losses.mode_seeking(p, p.clone(), torch.zeros(2, 8), torch.ones(2, 8), region)
    assert float(loss) == pytest.approx(1 / losses.MS_EPS)


# --------------------------------------------------------------------------- entraînement


def test_training_smoke_and_inference(dataset_dir: Path, tmp_path: Path) -> None:
    cfg = load_config([
        "train=smoke", "model=C", "model.ngf=8", "model.ndf=8", "train.max_steps=2",
        "train.batch_size=2", f"paths.dataset_dir={dataset_dir.as_posix()}",
        f"paths.experiments_dir={tmp_path.as_posix()}",
    ])  # fmt: skip
    out = train(cfg)
    for name in ("config.yaml", "meta.json", "train_log.csv", "val_log.csv",
                 "checkpoints/last.pt", "checkpoints/best.pt"):  # fmt: skip
        assert (out / name).exists(), name
    nets, state = load_checkpoint(out / "checkpoints" / "best.pt")
    assert state["config"]["model"]["name"] == "C"
    ds = PairDataset(dataset_dir / "val_synth.npz")
    item = ds[0]
    budget = int(item["budget_px"])
    proposals = propose(nets.generator, item["x"].numpy(), item["b_frac"], item["a_target"],
                        budget, n=3, seed=1)  # fmt: skip
    assert len(proposals) == 3
    region = item["x"][-1].numpy() > 0.5
    for prop in proposals:
        mask = prop.deployed(projected=True)
        assert mask.sum() <= budget
        assert not (mask & ~region).any()
    again = propose(nets.generator, item["x"].numpy(), item["b_frac"], item["a_target"],
                    budget, n=3, seed=1)  # fmt: skip
    for a, b in zip(proposals, again, strict=True):
        np.testing.assert_array_equal(a.prob, b.prob)  # mêmes graines → mêmes propositions
