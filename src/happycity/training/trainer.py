"""Entraînement d'une variante (A, B, C) ou d'une ablation, et validation par époque.

Sorties dans ``<experiments_dir>/seed<graine>/<run_name>/`` :

* ``config.yaml`` (configuration résolue) et ``meta.json`` (commit git, versions, paramètres) ;
* ``train_log.csv`` : termes de perte tous les ``log_every`` pas ;
* ``val_log.csv`` : métriques de validation par époque et par split ;
* ``checkpoints/last.pt`` et ``checkpoints/best.pt`` (sélection sur
  ``train.selection_split``, jamais sur le test).

La validation utilise l'A400 doux (rapide), recalculé sur les sorties binarisées. Les
métriques exactes (Dijkstra, module indépendant) sont celles de l'évaluation (phase 6).
"""

from __future__ import annotations

import json
import logging
import math
import platform
import time
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from happycity.models import Networks, build_networks, condition_vector, count_parameters
from happycity.models.inference import REGION_CHANNEL, WALK_CHANNEL, save_checkpoint
from happycity.projection import project
from happycity.training import losses
from happycity.training.data import PairDataset, collate
from happycity.utils.reproducibility import git_commit_hash, seed_everything

log = logging.getLogger(__name__)


def resolve_device(name: str) -> torch.device:
    """``auto`` → CUDA si disponible, sinon CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_dir(cfg: DictConfig) -> Path:
    """Dossier d'une exécution, rangé par graine : ``experiments/seed42/C-full-s42``."""
    return Path(cfg.paths.experiments_dir) / f"seed{int(cfg.train.seed)}" / str(cfg.run_name)


@dataclass
class StepLosses:
    """Termes de perte d'un pas (valeurs Python, pour le journal)."""

    d: float
    g_adv: float
    rec: float
    viol: float
    cond: float
    ms: float
    lr: float
    g_total: float


class Trainer:
    """Entraîneur cGAN : D (LSGAN, bruit d'instance) puis G (+ encodeur latent)."""

    def __init__(self, cfg: DictConfig, in_channels: int, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.nets: Networks = build_networks(cfg.model, in_channels)
        for net in self._modules():
            net.to(device)
        tc = cfg.train
        betas = (float(tc.betas[0]), float(tc.betas[1]))
        g_params = list(self.nets.generator.parameters())
        if self.nets.encoder is not None:
            g_params += list(self.nets.encoder.parameters())
        self.opt_g = torch.optim.Adam(g_params, lr=float(tc.lr_g), betas=betas)
        self.opt_d = torch.optim.Adam(
            self.nets.discriminator.parameters(), lr=float(tc.lr_d), betas=betas
        )
        self.amp = bool(tc.amp) and device.type == "cuda"
        self.scaler_g = GradScaler("cuda", enabled=self.amp)
        self.scaler_d = GradScaler("cuda", enabled=self.amp)
        self.w = tc.loss_weights
        self.div = cfg.model.diversity
        self.step = 0
        self.total_steps: float = 1.0

    def _modules(self) -> Iterator[torch.nn.Module]:
        yield self.nets.generator
        yield self.nets.discriminator
        if self.nets.encoder is not None:
            yield self.nets.encoder

    def _noise(self, mask: torch.Tensor) -> torch.Tensor:
        """Bruit d'instance gaussien, décroissant linéairement jusqu'à 0."""
        sigma = float(self.cfg.train.instance_noise) * max(0.0, 1 - self.step / self.total_steps)
        return mask + sigma * torch.randn_like(mask) if sigma > 0 else mask

    def train_step(self, batch: dict[str, torch.Tensor]) -> StepLosses:
        """Un pas : mise à jour de D, puis de G (et de l'encodeur latent)."""
        b = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        x, y = b["x"], b["y"]
        region = x[:, REGION_CHANNEL:]
        cond = condition_vector(b["b_frac"], b["a_target"])
        gen, disc, enc = self.nets.generator, self.nets.discriminator, self.nets.encoder
        for net in self._modules():
            net.train()
        n = x.shape[0]
        z1 = gen.sample_z(n)
        use_ms = float(self.div.mode_seeking_weight) > 0 and gen.spec.z_injection != "none"
        autocast = torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp)

        with autocast:
            out1 = gen(x, cond, z1)
        p1 = out1.prob.float()

        # --- D
        self.opt_d.zero_grad(set_to_none=True)
        with autocast:
            d_real = disc(x, cond, self._noise(y))
            d_fake = disc(x, cond, self._noise(p1.detach()))
            loss_d = 0.5 * (
                losses.lsgan(d_real.float(), True) + losses.lsgan(d_fake.float(), False)
            )
        self.scaler_d.scale(loss_d).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()

        # --- G (+ E)
        self.opt_g.zero_grad(set_to_none=True)
        with autocast:
            g_adv = losses.lsgan(disc(x, cond, self._noise(p1)).float(), True)
        rec = losses.reconstruction(p1, y, region)
        viol = losses.violation(p1, region, b["budget_px"])
        a_soft = losses.soft_a400(p1, b["pmap"], b["patterns"], b["n_origins"], b["n_base"])
        cond_loss = (a_soft - b["a_target"]).abs().mean()
        ms = torch.zeros((), device=self.device)
        if use_ms:
            z2 = gen.sample_z(n)
            with autocast:
                p2 = gen(x, cond, z2).prob.float()
            ms = losses.mode_seeking(p1, p2, z1, z2, region)
        lr = torch.zeros((), device=self.device)
        if enc is not None:
            with autocast:
                z_hat = enc(x, p1)
            lr = (z_hat.float() - z1).abs().mean()
        total = (
            float(self.w.adversarial) * g_adv
            + float(self.w.reconstruction) * rec
            + float(self.w.violation) * viol
            + float(self.w.condition) * cond_loss
            + float(self.div.mode_seeking_weight) * ms
            + float(self.div.latent_regression_weight) * lr
        )
        self.scaler_g.scale(total).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        self.step += 1
        values = [t.detach().item() for t in (loss_d, g_adv, rec, viol, cond_loss, ms, lr, total)]
        return StepLosses(*values)


def _iou_distance(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(1 - np.logical_and(a, b).sum() / union) if union else 0.0


@torch.no_grad()
def validate(
    generator: torch.nn.Module, dataset: PairDataset, cfg: DictConfig, device: torch.device
) -> dict[str, float]:
    """Métriques de validation (A400 doux recalculé sur les sorties binarisées).

    * ``viol_raw`` : part des pixels proposés (p ≥ seuil) hors de R ;
    * ``over_budget_raw`` : part des propositions brutes dépassant le budget ;
    * ``a_err_raw`` / ``a_err_proj`` : |Â − a| des sorties brute et projetée ;
    * ``fill_proj`` : surface projetée / budget ;
    * ``diversity`` : 1 − IoU moyen entre les ``val_z`` propositions projetées.
    """
    generator.eval()
    thr = float(cfg.model.threshold)
    k_z = max(int(cfg.train.val_z), 1)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate)
    gen_z = torch.Generator().manual_seed(int(cfg.eval.seed))
    names = ("viol_raw", "over_budget_raw", "a_err_raw", "a_err_proj", "fill_proj", "diversity")
    rows: dict[str, list[float]] = {k: [] for k in names}
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        x = b["x"]
        cond = condition_vector(b["b_frac"], b["a_target"])
        n = x.shape[0]
        projected: list[list[np.ndarray]] = [[] for _ in range(n)]
        for _ in range(k_z):
            prob = generator(x, cond, generator.sample_z(n, gen_z)).prob.float()  # type: ignore[operator]
            raw = (prob >= thr).float()
            x_np = x.cpu().numpy()
            masks = []
            prob_np = prob[:, 0].cpu().numpy()
            for i in range(n):
                budget = int(batch["budget_px"][i])
                allowed, walk = x_np[i, REGION_CHANNEL] > 0.5, x_np[i, WALK_CHANNEL] > 0.5
                res = project(prob_np[i], allowed, walk, budget, threshold=thr)
                masks.append(res.mask)
                projected[i].append(res.mask)
                rows["viol_raw"].append(res.removed_outside_r / max(res.proposed_px, 1))
                rows["over_budget_raw"].append(float(res.proposed_px > budget))
                rows["fill_proj"].append(res.n_px / max(budget, 1))
            proj = torch.as_tensor(np.stack(masks)[:, None], dtype=torch.float32, device=device)
            args = (b["pmap"], b["patterns"], b["n_origins"], b["n_base"])
            a_raw = losses.soft_a400(raw * x[:, REGION_CHANNEL:], *args)
            a_proj = losses.soft_a400(proj, *args)
            rows["a_err_raw"] += (a_raw - b["a_target"]).abs().cpu().tolist()
            rows["a_err_proj"] += (a_proj - b["a_target"]).abs().cpu().tolist()
        for masks_i in projected:
            if len(masks_i) > 1:
                rows["diversity"].append(
                    float(np.mean([_iou_distance(a, c) for a, c in combinations(masks_i, 2)]))
                )
    return {k: float(np.mean(v)) if v else float("nan") for k, v in rows.items()}


def selection_score(metrics: dict[str, float], projected: bool) -> float:
    """Score de sélection (plus bas = meilleur) : fidélité à a de la sortie retenue + violations."""
    a_err = metrics["a_err_proj"] if projected else metrics["a_err_raw"]
    viol = 0.0 if projected else metrics["viol_raw"] + metrics["over_budget_raw"]
    return a_err + viol


def _dataset_path(cfg: DictConfig, split: str) -> Path:
    return Path(cfg.paths.dataset_dir) / f"{split}.npz"


def train(cfg: DictConfig) -> Path:
    """Entraîne la variante décrite par ``cfg`` ; renvoie le dossier de l'exécution."""
    tc = cfg.train
    seed_everything(int(tc.seed), deterministic=bool(tc.deterministic))
    device = resolve_device(str(tc.device))
    if device.type == "cuda" and not bool(tc.deterministic):
        torch.backends.cudnn.benchmark = True  # tailles fixes : choix du meilleur algorithme
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    train_set = PairDataset(_dataset_path(cfg, "train"), augment=bool(tc.augment))
    val_sets = {
        s: PairDataset(_dataset_path(cfg, s), max_pairs=tc.val_max_pairs)
        for s in ("val_synth", "val")
        if _dataset_path(cfg, s).exists()
    }
    if str(tc.selection_split) not in val_sets:
        raise FileNotFoundError(f"split de sélection absent : {tc.selection_split}")
    loader = DataLoader(
        train_set,
        batch_size=int(tc.batch_size),
        shuffle=True,
        drop_last=len(train_set) >= int(tc.batch_size),
        num_workers=int(tc.num_workers),
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=int(tc.num_workers) > 0,
        generator=torch.Generator().manual_seed(int(tc.seed)),
    )
    trainer = Trainer(cfg, train_set.in_channels, device)
    epochs = int(tc.epochs)
    max_steps = int(tc.max_steps) if tc.max_steps is not None else None
    trainer.total_steps = float(min(epochs * len(loader), max_steps or math.inf))
    meta = {
        "run_name": str(cfg.run_name),
        "git_commit": git_commit_hash(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "in_channels": train_set.in_channels,
        "input_channels": list(train_set.input_channels),
        "train_pairs": len(train_set),
        "val_pairs": {k: len(v) for k, v in val_sets.items()},
        "params_generator": count_parameters(trainer.nets.generator),
        "params_discriminator": count_parameters(trainer.nets.discriminator),
        "params_encoder": count_parameters(trainer.nets.encoder) if trainer.nets.encoder else 0,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("entraînement %s sur %s : %d paires", cfg.run_name, device, len(train_set))

    projected = bool(cfg.model.projection)
    best = math.inf
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    extra = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "in_channels": train_set.in_channels,
        "meta": meta,
    }
    start = time.time()
    done = False
    for epoch in range(1, epochs + 1):
        for batch in loader:
            losses_step = trainer.train_step(batch)
            if trainer.step % int(tc.log_every) == 0 or trainer.step == 1:
                train_rows.append({"step": trainer.step, "epoch": epoch, **losses_step.__dict__,
                                   "time_s": time.time() - start})  # fmt: skip
                log.info(
                    "pas %d — %s",
                    trainer.step,
                    {k: round(v, 4) for k, v in losses_step.__dict__.items()},
                )
            if max_steps is not None and trainer.step >= max_steps:
                done = True
                break
        for split, ds in val_sets.items():
            metrics = validate(trainer.nets.generator, ds, cfg, device)
            val_rows.append({"epoch": epoch, "step": trainer.step, "split": split, **metrics,
                             "score": selection_score(metrics, projected)})  # fmt: skip
        sel = next(r for r in reversed(val_rows) if r["split"] == str(tc.selection_split))
        state = {**extra, "epoch": epoch, "step": trainer.step, "val": sel}
        save_checkpoint(out / "checkpoints" / "last.pt", trainer.nets, state)
        if sel["score"] < best:
            best = float(sel["score"])
            save_checkpoint(out / "checkpoints" / "best.pt", trainer.nets, state)
        pd.DataFrame(train_rows).to_csv(out / "train_log.csv", index=False)
        pd.DataFrame(val_rows).to_csv(out / "val_log.csv", index=False)
        log.info(
            "époque %d : %s",
            epoch,
            {k: round(v, 4) for k, v in sel.items() if isinstance(v, float)},
        )
        if done:
            break
    return out
