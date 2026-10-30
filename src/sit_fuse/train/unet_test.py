#!/usr/bin/env python3
"""
Train a compact U-Net using SIT-FUSE context-free segment rasters as input
and aligned HMS smoke/fire rasters as targets.

Usage:
    python train_context_free_unet.py unet_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from rasterio.enums import Resampling
from rasterio.warp import reproject
from torch.utils.data import DataLoader, Dataset


LOG = logging.getLogger("context-free-unet")


@dataclass(frozen=True)
class Matchup:
    pair_id: str
    date: str
    product: str
    input_path: Path
    target_path: Path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Configuration must parse to a YAML dictionary.")

    return cfg


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_date_string(value: Any) -> str:
    return pd_to_date(value).strftime("%Y-%m-%d")


def pd_to_date(value: Any) -> dt.datetime:
    text = str(value).strip()

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise ValueError(f"Could not parse date/timestamp: {value}")


def normalize_date_list(values: list[Any]) -> set[str]:
    return {parse_date_string(value) for value in values}


def load_matchups(cfg: dict[str, Any]) -> list[Matchup]:
    """
    Supported input formats:

    1. Matchup CSV:
       pair_id,hms_date,sit_fuse_path,hms_path,product

    2. Matchup YAML:
       matchups:
         - pair_id: smoke_20240615
           hms_date: 2024-06-15
           product: smoke
           sit_fuse_context_free: /path/input.tif
           truth_hms: /path/target.tif
    """
    data_cfg = cfg["data"]
    matchup_path = Path(data_cfg["matchup_file"])

    input_key = data_cfg.get(
        "input_path_column",
        "sit_fuse_context_free",
    )
    target_key = data_cfg.get(
        "target_path_column",
        "truth_hms",
    )
    date_key = data_cfg.get("date_column", "hms_date")
    pair_key = data_cfg.get("pair_id_column", "pair_id")
    product_key = data_cfg.get("product_column", "product")

    if matchup_path.suffix.lower() == ".csv":
        import pandas as pd
        frame = pd.read_csv(matchup_path)
        rows = frame.to_dict(orient="records")
    else:
        doc = load_yaml(matchup_path)
        rows = doc.get("matchups", doc.get("pairs", []))

    if not rows:
        raise ValueError(f"No matchup records found in {matchup_path}")

    records = []

    for row in rows:
        if input_key not in row or target_key not in row:
            raise KeyError(
                f"Matchup missing '{input_key}' or '{target_key}'. "
                f"Available keys: {list(row.keys())}"
            )

        input_path = Path(row[input_key])
        target_path = Path(row[target_key])

        if not input_path.exists():
            raise FileNotFoundError(f"Missing SIT-FUSE input: {input_path}")
        if not target_path.exists():
            raise FileNotFoundError(f"Missing HMS target: {target_path}")

        date_value = row.get(date_key, row.get("date"))
        if date_value is None:
            raise KeyError(
                f"Matchup missing configured date key '{date_key}'."
            )

        records.append(
            Matchup(
                pair_id=str(row.get(pair_key, input_path.stem)),
                date=parse_date_string(date_value),
                product=str(row.get(product_key, cfg["task"]["product"])),
                input_path=input_path,
                target_path=target_path,
            )
        )

    return records


def split_matchups(
    matchups: list[Matchup],
    cfg: dict[str, Any],
) -> tuple[list[Matchup], list[Matchup], list[Matchup]]:
    split_cfg = cfg["splits"]

    holdout_dates = normalize_date_list(
        split_cfg.get("holdout_dates", [])
    )
    validation_dates = normalize_date_list(
        split_cfg.get("validation_dates", [])
    )

    overlap = holdout_dates & validation_dates
    if overlap:
        raise ValueError(
            f"Dates cannot be in both validation and holdout: {sorted(overlap)}"
        )

    train, validation, holdout = [], [], []

    for matchup in matchups:
        if matchup.date in holdout_dates:
            holdout.append(matchup)
        elif matchup.date in validation_dates:
            validation.append(matchup)
        else:
            train.append(matchup)

    if not train:
        raise ValueError("No training matchups remain after date splitting.")

    if not validation:
        raise ValueError(
            "No validation matchups. Add splits.validation_dates."
        )

    LOG.info(
        "Date split: train=%d scenes, validation=%d scenes, holdout=%d scenes",
        len(train),
        len(validation),
        len(holdout),
    )

    return train, validation, holdout


def read_input_and_target(
    matchup: Matchup,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read context-free input and align HMS target to the input raster grid.

    Returns:
        x: float32 [H, W]
        y: int64 [H, W]
        valid: bool [H, W]
    """
    task_cfg = cfg["task"]
    ignore_index = int(task_cfg.get("ignore_index", -1))

    with rasterio.open(matchup.input_path) as src:
        x = src.read(1).astype(np.float32)
        input_profile = src.profile.copy()
        input_transform = src.transform
        input_crs = src.crs
        input_nodata = src.nodata

    if input_nodata is not None:
        input_valid = np.isfinite(x) & (x != input_nodata)
    else:
        input_valid = np.isfinite(x)

    with rasterio.open(matchup.target_path) as target_src:
        y_raw = target_src.read(1)

        y = np.full(
            shape=x.shape,
            fill_value=ignore_index,
            dtype=np.int64,
        )

        reproject(
            source=y_raw,
            destination=y,
            src_transform=target_src.transform,
            src_crs=target_src.crs,
            dst_transform=input_transform,
            dst_crs=input_crs,
            src_nodata=target_src.nodata,
            dst_nodata=ignore_index,
            resampling=Resampling.nearest,
        )

    target_valid = y != ignore_index
    valid = input_valid & target_valid

    input_max = float(task_cfg.get("input_max_value", 0))
    if input_max > 0:
        x = x / input_max
    else:
        valid_x = x[input_valid]
        scale = float(np.nanmax(valid_x)) if valid_x.size else 1.0
        scale = max(scale, 1.0)
        x = x / scale

    x[~input_valid] = 0.0
    y[~valid] = ignore_index

    if task_cfg["mode"] == "binary":
        positive_values = task_cfg.get("positive_values", [1])
        y = np.where(
            y == ignore_index,
            ignore_index,
            np.isin(y, positive_values).astype(np.int64),
        )

    return x.astype(np.float32), y.astype(np.int64), valid


class PatchDataset(Dataset):
    """
    Samples random patches from already-loaded training scenes.

    Rejection sampling encourages patches containing a minimum fraction
    of positive/foreground pixels.
    """

    def __init__(
        self,
        matchups: list[Matchup],
        cfg: dict[str, Any],
        patches_per_epoch: int,
    ):
        self.cfg = cfg
        self.patch_size = int(cfg["training"].get("patch_size", 128))
        self.ignore_index = int(cfg["task"].get("ignore_index", -1))
        self.min_foreground_fraction = float(
            cfg["training"].get("min_foreground_fraction", 0.01)
        )
        self.max_patch_attempts = int(
            cfg["training"].get("max_patch_attempts", 20)
        )

        self.foreground_patch_probability = float(
            cfg["training"].get("foreground_patch_probability", 0.75)
        )

        self.patches_per_epoch = patches_per_epoch

        self.scenes = []
        for matchup in matchups:
            x, y, valid = read_input_and_target(matchup, cfg)
            if x.shape[0] < self.patch_size or x.shape[1] < self.patch_size:
                LOG.warning(
                    "Skipping scene smaller than patch size: %s",
                    matchup.input_path,
                )
                continue

            self.scenes.append(
                {
                    "matchup": matchup,
                    "x": x,
                    "y": y,
                    "valid": valid,
                }
            )

        if not self.scenes:
            raise ValueError("No valid training scenes for patch extraction.")

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        scene = self.scenes[index % len(self.scenes)]
        x = scene["x"]
        y = scene["y"]

        h, w = x.shape
        p = self.patch_size
 
        foreground_yx = np.argwhere(
            (y > 0) & (y != self.ignore_index)
        )

        sample_foreground = (
            len(foreground_yx) > 0
            and np.random.random() < self.foreground_patch_probability
        )

        best = None


        for _ in range(self.max_patch_attempts):
 
            if sample_foreground:
                center_y, center_x = foreground_yx[
                    np.random.randint(len(foreground_yx))
                ]

                y0 = int(
                    np.clip(
                        center_y - np.random.randint(0, p),
                        0,
                        h - p,
                    )
                )
                x0 = int(
                    np.clip(
                        center_x - np.random.randint(0, p),
                        0,
                        w - p,
                    )
                )
            else:
                y0 = np.random.randint(0, h - p + 1)
                x0 = np.random.randint(0, w - p + 1)



            x_patch = x[y0:y0 + p, x0:x0 + p]
            y_patch = y[y0:y0 + p, x0:x0 + p]

            valid = y_patch != self.ignore_index
            foreground = (y_patch > 0) & valid

            valid_count = int(valid.sum())
            foreground_fraction = (
                float(foreground.sum()) / valid_count
                if valid_count > 0
                else 0.0
            )

            best = (x_patch, y_patch)

            if foreground_fraction >= self.min_foreground_fraction:
                break

        x_patch, y_patch = best

        return {
            "x": torch.from_numpy(x_patch[None, ...]),
            "y": torch.from_numpy(y_patch),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """
    Compact U-Net for small segmentation experiments.

    Input:  [B, in_channels, H, W]
    Output: [B, n_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        base_channels: int = 16,
    ):
        super().__init__()

        c = base_channels

        self.enc1 = ConvBlock(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c * 4, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        self.head = nn.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)


class BinaryFocalLoss(nn.Module):
    """
    Numerically stable binary focal loss over logits.

    alpha:
        Relative contribution assigned to the positive/fire class.
        alpha=0.75 means fire receives 3x the base class contribution
        relative to no-fire before the focal modulation.

    gamma:
        Focusing parameter. gamma=0 reduces to alpha-balanced BCE.
        gamma > 0 reduces loss from easy examples.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        pos_weight: float | None = None,
    ):
        super().__init__()

        if not 0.0 < alpha < 1.0:
            raise ValueError("focal alpha must lie strictly between 0 and 1.")

        if gamma < 0:
            raise ValueError("focal gamma must be >= 0.")

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        if pos_weight is None:
            self.register_buffer("pos_weight", torch.tensor(1.0))
        else:
            self.register_buffer(
                "pos_weight",
                torch.tensor(float(pos_weight)),
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits:  [B, 1, H, W]
        targets: [B, 1, H, W], values 0 or 1

        Returns an unreduced loss map with shape [B, 1, H, W].
        """
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
            reduction="none",
        )

        probabilities = torch.sigmoid(logits)

        p_t = (
            probabilities * targets
            + (1.0 - probabilities) * (1.0 - targets)
        )

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        focal_weight = (1.0 - p_t).pow(self.gamma)

        return alpha_t * focal_weight * bce


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable Dice loss for binary segmentation.
    """
    probabilities = torch.sigmoid(logits[:, 0])

    valid = valid_mask.float()

    probabilities = probabilities * valid
    targets = targets.float() * valid

    intersection = (probabilities * targets).sum(dim=(1, 2))
    denominator = (
        probabilities.sum(dim=(1, 2))
        + targets.sum(dim=(1, 2))
    )

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - dice.mean()



def build_loss(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    task_cfg = cfg["task"]
    mode = task_cfg["mode"]
    ignore_index = int(task_cfg.get("ignore_index", -1))

    if mode == "binary":
        loss_name = str(
            task_cfg.get("loss", "focal")
        ).lower()

        if loss_name == "bce":
            pos_weight = float(task_cfg.get("positive_weight", 1.0))

            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], device=device),
                reduction="none",
            )

        if loss_name == "focal":
            return BinaryFocalLoss(
                alpha=float(task_cfg.get("focal_alpha", 0.75)),
                gamma=float(task_cfg.get("focal_gamma", 2.0)),
                pos_weight=task_cfg.get("positive_weight", None),
            )

        raise ValueError(
            f"Unsupported binary loss '{loss_name}'. "
            "Supported values are: bce, focal."
        )


    class_weights = task_cfg.get("class_weights")
    weight_tensor = None

    if class_weights is not None:
        weight_tensor = torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )

    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        ignore_index=ignore_index,
        reduction="none",
    )


def masked_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    ignore_index: int,
    cfg: dict[str, Any],
) -> torch.Tensor:
    """
    logits: [B, 1, H, W]
    target: [B, H, W], values 0/1/ignore_index
    """
    valid = target != ignore_index

    if valid.sum() == 0:
        return logits.sum() * 0.0

    target_float = target.clamp(min=0).float().unsqueeze(1)
    loss_map = criterion(logits, target_float)

    valid_4d = valid.unsqueeze(1)

    focal_or_bce_loss = loss_map[valid_4d].mean()

    dice_weight = float(
        cfg["task"].get("dice_loss_weight", 0.0)
    )

    if dice_weight <= 0:
        return focal_or_bce_loss

    dice_loss = soft_dice_loss(
        logits=logits,
        targets=target,
        valid_mask=valid,
    )

    return (
        (1.0 - dice_weight) * focal_or_bce_loss
        + dice_weight * dice_loss
    )


def masked_multiclass_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    ignore_index: int,
) -> torch.Tensor:
    """
    logits: [B, C, H, W]
    target: [B, H, W]
    """
    loss_map = criterion(logits, target)
    valid = target != ignore_index

    if valid.sum() == 0:
        return logits.sum() * 0.0

    return loss_map[valid].mean()


@torch.no_grad()
def binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int,
    threshold: float,
) -> dict[str, float]:
    valid = target != ignore_index

    if valid.sum() == 0:
        return {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "f1": np.nan,
            "iou": np.nan,
        }

    pred = (torch.sigmoid(logits[:, 0]) >= threshold).long()
    truth = target.long()

    pred = pred[valid]
    truth = truth[valid]

    tp = int(((pred == 1) & (truth == 1)).sum().item())
    tn = int(((pred == 0) & (truth == 0)).sum().item())
    fp = int(((pred == 1) & (truth == 0)).sum().item())
    fn = int(((pred == 0) & (truth == 1)).sum().item())

    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "f1": f1,
        "iou": iou,
    }


@torch.no_grad()
def multiclass_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    ignore_index: int,
) -> dict[str, Any]:
    pred = torch.argmax(logits, dim=1)
    valid = target != ignore_index

    confusion = torch.zeros(
        (n_classes, n_classes),
        dtype=torch.int64,
        device=target.device,
    )

    if valid.sum() == 0:
        return {
            "confusion_matrix": confusion.cpu().tolist(),
            "macro_iou": np.nan,
        }

    truth_flat = target[valid]
    pred_flat = pred[valid]

    for true_class in range(n_classes):
        for pred_class in range(n_classes):
            confusion[true_class, pred_class] = (
                ((truth_flat == true_class) & (pred_flat == pred_class))
                .sum()
            )

    ious = []

    for c in range(n_classes):
        tp = confusion[c, c].float()
        fp = confusion[:, c].sum().float() - tp
        fn = confusion[c, :].sum().float() - tp
        denom = tp + fp + fn
        ious.append(
            (tp / denom).item() if denom > 0 else np.nan
        )

    return {
        "confusion_matrix": confusion.cpu().tolist(),
        "class_iou": ious,
        "macro_iou": float(np.nanmean(ious)),
    }


@torch.no_grad()
def predict_scene_tiled(
    model: nn.Module,
    x: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """
    Predict one scene in overlapping tiles and average logits in overlaps.
    """
    patch_size = int(cfg["training"].get("patch_size", 128))
    stride = int(cfg["validation"].get("tile_stride", patch_size // 2))

    h, w = x.shape

    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)

    padded = np.pad(
        x,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0,
    )

    hp, wp = padded.shape
    n_outputs = 1 if cfg["task"]["mode"] == "binary" else int(
        cfg["task"]["num_classes"]
    )

    logits_sum = torch.zeros(
        (n_outputs, hp, wp),
        dtype=torch.float32,
        device=device,
    )
    count = torch.zeros(
        (1, hp, wp),
        dtype=torch.float32,
        device=device,
    )

    y_starts = list(range(0, max(hp - patch_size + 1, 1), stride))
    x_starts = list(range(0, max(wp - patch_size + 1, 1), stride))

    if y_starts[-1] != hp - patch_size:
        y_starts.append(hp - patch_size)

    if x_starts[-1] != wp - patch_size:
        x_starts.append(wp - patch_size)

    model.eval()

    for y0 in y_starts:
        for x0 in x_starts:
            patch = padded[y0:y0 + patch_size, x0:x0 + patch_size]

            tensor = torch.from_numpy(
                patch[None, None, ...]
            ).float().to(device)

            logits = model(tensor)[0]

            logits_sum[:, y0:y0 + patch_size, x0:x0 + patch_size] += logits
            count[:, y0:y0 + patch_size, x0:x0 + patch_size] += 1.0

    logits = logits_sum / count.clamp(min=1.0)
    return logits[:, :h, :w]


@torch.no_grad()
def evaluate_scenes(
    model: nn.Module,
    matchups: list[Matchup],
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    task_cfg = cfg["task"]
    ignore_index = int(task_cfg.get("ignore_index", -1))

    total_loss = 0.0
    n_scenes = 0

    if task_cfg["mode"] == "binary":
        tp = tn = fp = fn = 0
    else:
        n_classes = int(task_cfg["num_classes"])
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    criterion = build_loss(cfg, device)

    for matchup in matchups:
        x, y, _ = read_input_and_target(matchup, cfg)

        logits = predict_scene_tiled(model, x, cfg, device)
        target = torch.from_numpy(y[None, ...]).long().to(device)

        if task_cfg["mode"] == "binary":
            loss = masked_binary_loss(
                logits[None, ...],
                target,
                criterion,
                ignore_index,
                cfg
            )

            metrics = binary_metrics(
                logits[None, ...],
                target,
                ignore_index,
                threshold=float(task_cfg.get("threshold", 0.5)),
            )

            tp += metrics["tp"]
            tn += metrics["tn"]
            fp += metrics["fp"]
            fn += metrics["fn"]

        else:
            loss = masked_multiclass_loss(
                logits[None, ...],
                target,
                criterion,
                ignore_index,
            )

            metrics = multiclass_metrics(
                logits[None, ...],
                target,
                n_classes=n_classes,
                ignore_index=ignore_index,
            )

            confusion += np.asarray(metrics["confusion_matrix"])

        total_loss += float(loss.item())
        n_scenes += 1

    result = {
        "loss": total_loss / max(n_scenes, 1),
        "n_scenes": n_scenes,
    }

    if task_cfg["mode"] == "binary":
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan

        result.update({
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "f1": f1,
            "iou": iou,
        })

    else:
        class_iou = []

        for c in range(n_classes):
            tp_c = confusion[c, c]
            fp_c = confusion[:, c].sum() - tp_c
            fn_c = confusion[c, :].sum() - tp_c
            denom = tp_c + fp_c + fn_c
            class_iou.append(tp_c / denom if denom else np.nan)

        result.update({
            "confusion_matrix": confusion.tolist(),
            "class_iou": class_iou,
            "macro_iou": float(np.nanmean(class_iou)),
        })

    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
) -> float:
    model.train()

    task_cfg = cfg["task"]
    ignore_index = int(task_cfg.get("ignore_index", -1))

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].float().to(device)
        y = batch["y"].long().to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        if task_cfg["mode"] == "binary":
            loss = masked_binary_loss(
                logits,
                y,
                criterion,
                ignore_index,
                cfg,
            )
        else:
            loss = masked_multiclass_loss(
                logits,
                y,
                criterion,
                ignore_index,
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(cfg["training"].get("gradient_clip_norm", 1.0)),
        )

        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def write_prediction_raster(
    reference_path: Path,
    output_path: Path,
    prediction: np.ndarray,
    nodata,
) -> None:
    """
    Write one georeferenced prediction raster on the SIT-FUSE input grid.
    """
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        count=1,
        height=prediction.shape[0],
        width=prediction.shape[1],
        dtype=str(prediction.dtype),
        nodata=nodata,
        compress="lzw",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction, 1)

@torch.no_grad()
def predict_and_write_scene(
    model: nn.Module,
    matchup: Matchup,
    cfg: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Predict a single SIT-FUSE context-free scene and write GeoTIFF outputs.

    Binary task outputs:
      - class mask: 0/1
      - fire/smoke probability: float32 in [0, 1]

    Multiclass task outputs:
      - predicted class mask: integer class IDs
      - optional per-class probability stack: one band per class
    """
    task_cfg = cfg["task"]
    ignore_index = int(task_cfg.get("ignore_index", -1))
    write_probabilities = bool(
        cfg.get("prediction", {}).get("write_probabilities", True)
    )

    x, y, valid = read_input_and_target(matchup, cfg)

    logits = predict_scene_tiled(
        model=model,
        x=x,
        cfg=cfg,
        device=device,
    )

    scene_dir = output_dir / matchup.product / matchup.date
    scene_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"{matchup.pair_id}_"
        f"{matchup.product}_"
        f"{matchup.date.replace('-', '')}"
    )

    valid_mask = valid.astype(bool)

    if task_cfg["mode"] == "binary":
        threshold = float(task_cfg.get("threshold", 0.5))

        probability = torch.sigmoid(logits[0]).cpu().numpy().astype(np.float32)
        prediction = (probability >= threshold).astype(np.int16)

        prediction[~valid_mask] = ignore_index
        probability[~valid_mask] = np.nan

        class_path = scene_dir / f"{stem}_prediction.tif"
        probability_path = scene_dir / f"{stem}_probability.tif"

        write_prediction_raster(
            reference_path=matchup.input_path,
            output_path=class_path,
            prediction=prediction,
            nodata=ignore_index,
        )

        if write_probabilities:
            write_prediction_raster(
                reference_path=matchup.input_path,
                output_path=probability_path,
                prediction=probability,
                nodata=np.nan,
            )

        return {
            "pair_id": matchup.pair_id,
            "date": matchup.date,
            "product": matchup.product,
            "input_path": str(matchup.input_path),
            "target_path": str(matchup.target_path),
            "prediction_path": str(class_path),
            "probability_path": str(probability_path)
            if write_probabilities else "",
        }

    n_classes = int(task_cfg["num_classes"])

    probabilities = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
    prediction = np.argmax(probabilities, axis=0).astype(np.int16)

    prediction[~valid_mask] = ignore_index
    probabilities[:, ~valid_mask] = np.nan

    class_path = scene_dir / f"{stem}_prediction.tif"

    write_prediction_raster(
        reference_path=matchup.input_path,
        output_path=class_path,
        prediction=prediction,
        nodata=ignore_index,
    )

    probability_path = ""

    if write_probabilities:
        probability_path = scene_dir / f"{stem}_class_probabilities.tif"

        with rasterio.open(matchup.input_path) as src:
            profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            count=n_classes,
            dtype="float32",
            nodata=np.nan,
            compress="lzw",
        )

        with rasterio.open(probability_path, "w", **profile) as dst:
            for class_index in range(n_classes):
                dst.write(probabilities[class_index], class_index + 1)
                dst.set_band_description(
                    class_index + 1,
                    f"class_{class_index}_probability",
                )

    return {
        "pair_id": matchup.pair_id,
        "date": matchup.date,
        "product": matchup.product,
        "input_path": str(matchup.input_path),
        "target_path": str(matchup.target_path),
        "prediction_path": str(class_path),
        "probability_path": str(probability_path),
    }


@torch.no_grad()
def generate_prediction_rasters(
    model: nn.Module,
    matchups: list[Matchup],
    cfg: dict[str, Any],
    device: torch.device,
    split_name: str,
) -> list[dict[str, Any]]:
    """
    Generate prediction GeoTIFFs for every scene in a train, validation,
    or holdout split.
    """
    prediction_cfg = cfg.get("prediction", {})

    if not prediction_cfg.get("enabled", True):
        LOG.info("Prediction-raster generation disabled.")
        return []

    root_dir = Path(
        prediction_cfg.get(
            "output_dir",
            Path(cfg["output_dir"]) / "predictions",
        )
    )

    split_dir = root_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for i, matchup in enumerate(matchups, start=1):
        LOG.info(
            "Writing %s prediction %d/%d: %s",
            split_name,
            i,
            len(matchups),
            matchup.input_path.name,
        )

        record = predict_and_write_scene(
            model=model,
            matchup=matchup,
            cfg=cfg,
            device=device,
            output_dir=split_dir,
        )

        record["split"] = split_name
        records.append(record)

    if records:
        manifest_path = split_dir / "prediction_manifest.csv"

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(records[0].keys()),
            )
            writer.writeheader()
            writer.writerows(records)

        LOG.info("Wrote prediction manifest: %s", manifest_path)

    return records


def run(cfg: dict[str, Any]) -> None:
    seed = int(cfg.get("random_state", 42))
    seed_everything(seed)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = cfg["training"].get(
        "device",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(device_name)

    matchups = load_matchups(cfg)
    product = cfg["task"]["product"].lower()

    matchups = [
        matchup
        for matchup in matchups
        if matchup.product.lower() == product
    ]

    if not matchups:
        raise ValueError(f"No matchups available for product='{product}'.")

    train_matchups, val_matchups, holdout_matchups = split_matchups(
        matchups,
        cfg,
    )

    dataset = PatchDataset(
        matchups=train_matchups,
        cfg=cfg,
        patches_per_epoch=int(
            cfg["training"].get("patches_per_epoch", 2000)
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 16)),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(
            cfg["training"].get("num_workers", 4)
        ) > 0,
    )

    n_outputs = 1 if cfg["task"]["mode"] == "binary" else int(
        cfg["task"]["num_classes"]
    )

    model = SmallUNet(
        in_channels=1,
        n_classes=n_outputs,
        base_channels=int(cfg["model"].get("base_channels", 16)),
    ).to(device)

    criterion = build_loss(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    epochs = int(cfg["training"].get("epochs", 50))
    best_score = -np.inf
    history = []

    for epoch in tqdm(range(1, epochs + 1)):
        print(len(loader), "HERE EPOCH")
        train_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            cfg=cfg,
            device=device,
        )

        validation = evaluate_scenes(
            model=model,
            matchups=val_matchups,
            cfg=cfg,
            device=device,
        )

        if cfg["task"]["mode"] == "binary":
            score = validation["iou"]
        else:
            score = validation["macro_iou"]

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(history_row)

        LOG.info(
            "Epoch %03d | train_loss=%.5f | validation=%s",
            epoch,
            train_loss,
            validation,
        )

        if np.isfinite(score) and score > best_score:
            best_score = score

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "validation": validation,
                },
                output_dir / "best_model.pt",
            )

    pd_history = __import__("pandas").DataFrame(history)
    pd_history.to_csv(output_dir / "history.csv", index=False)

    checkpoint = torch.load(
        output_dir / "best_model.pt",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    final_validation = evaluate_scenes(
        model=model,
        matchups=val_matchups,
        cfg=cfg,
        device=device,
    )

    final_holdout = {}
    if holdout_matchups:
        final_holdout = evaluate_scenes(
            model=model,
            matchups=holdout_matchups,
            cfg=cfg,
            device=device,
        )


    prediction_records = []
 
    prediction_records.extend(
        generate_prediction_rasters(
            model=model,
            matchups=train_matchups,
            cfg=cfg,
            device=device,
            split_name="train",
        )
    )

    prediction_records.extend(
        generate_prediction_rasters(
            model=model,
            matchups=val_matchups,
            cfg=cfg,
            device=device,
            split_name="validation",
        )
    )

    if holdout_matchups:
        prediction_records.extend(
            generate_prediction_rasters(
                model=model,
                matchups=holdout_matchups,
                cfg=cfg,
                device=device,
                split_name="holdout",
            )
        )

    summary = {
        "device": str(device),
        "n_train_scenes": len(train_matchups),
        "n_validation_scenes": len(val_matchups),
        "n_holdout_scenes": len(holdout_matchups),
        "best_validation_score": best_score,
        "validation": final_validation,
        "holdout": final_holdout,
    }
    save_json(output_dir / "summary.json", summary)

    LOG.info("Training complete. Summary: %s", summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a compact U-Net from context-free SIT-FUSE segments."
    )
    parser.add_argument("yaml_config", type=Path)
    args = parser.parse_args()

    cfg = load_yaml(args.yaml_config)
    run(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()




