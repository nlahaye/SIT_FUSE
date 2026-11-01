"""Trains the RTDBN IIC clustering head on the frozen, pretrained RTDBN encoder."""
import argparse
import os

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import entropy as scipy_entropy
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.strategies import DDPStrategy

from learnergy.models.temporal.rtdbn import RTDBN

from sit_fuse.models.deep_cluster.rtdbn_dc import RTDBN_DC
from sit_fuse.datasets.rtdbn_data_utils import set_all_seeds, build_rtdbn_splits
import yaml


def read_yaml(fpath):
    with open(fpath, "r") as f:
        return yaml.safe_load(f)


def _load_frozen_encoder(yml_conf, encoder_dir):
    """Rebuilds RTDBN from config, loads trained weights, freezes, and eval()s."""
    n_visible = int(yml_conf["rtdbn"]["n_visible"])
    n_hidden = tuple(yml_conf["rtdbn"]["n_hidden"])
    gibbs_steps = tuple(yml_conf["rtdbn"]["gibbs_steps"])
    temp = tuple(yml_conf["rtdbn"]["temp"])
    model_type = tuple(yml_conf["rtdbn"]["model_type"])

    learning_rate = tuple(yml_conf["encoder"]["training"]["learning_rate"])
    momentum = tuple(yml_conf["encoder"]["training"]["momentum"])
    decay = tuple(yml_conf["encoder"]["training"]["weight_decay"])
    accelerator = yml_conf["encoder"]["training"]["accelerator"]

    rtdbn = RTDBN(
        model=model_type,
        n_visible=n_visible,
        n_hidden=n_hidden,
        steps=gibbs_steps,
        learning_rate=learning_rate,
        momentum=momentum,
        decay=decay,
        temperature=temp,
        use_gpu=(accelerator == "gpu"),
    )

    ckpt_path = os.path.join(encoder_dir, "rtdbn.ckpt")
    rtdbn.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    for param in rtdbn.parameters():
        param.requires_grad = False
    for model in rtdbn.models:
        for param in model.parameters():
            param.requires_grad = False
    rtdbn.eval()

    return rtdbn


def train_RTDBN_DC(yml_conf):
    out_dir = yml_conf["output"]["out_dir"]
    encoder_dir = os.path.join(out_dir, "encoder")
    save_dir = os.path.join(out_dir, "full_model")
    os.makedirs(save_dir, exist_ok=True)

    num_workers = int(yml_conf["data"]["num_loader_workers"])

    num_classes = yml_conf["cluster"]["num_classes"]
    lr = yml_conf["cluster"]["training"]["learning_rate"]
    batch_size = yml_conf["cluster"]["training"]["batch_size"]
    max_epochs = yml_conf["cluster"]["training"]["epochs"]
    accelerator = yml_conf["cluster"]["training"]["accelerator"]
    devices = yml_conf["cluster"]["training"]["devices"]
    precision = yml_conf["cluster"]["training"]["precision"]
    gradient_clip = yml_conf["cluster"]["training"]["gradient_clip_val"]
    noise_stdev = float(yml_conf["cluster"]["gauss_noise_stdev"][0])
    lamb = float(yml_conf["cluster"]["lambda"])

    set_all_seeds(42)

    train_ds, val_ds, test_ds = build_rtdbn_splits(
        yml_conf, scaler_in_path=os.path.join(encoder_dir, "scaler.pkl")
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    rtdbn = _load_frozen_encoder(yml_conf, encoder_dir)

    model = RTDBN_DC(
        rtdbn, num_classes=num_classes, lr=lr, noise_stdev=noise_stdev, lamb=lamb
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)
    checkpoint_steps = 100
    if max_epochs < checkpoint_steps:
        checkpoint_steps = 1
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="deep_cluster",
        enable_version_counter=False,
        every_n_train_steps=checkpoint_steps,
        save_on_train_epoch_end=False,
    )

    # See pretrain_rtdbn.py's matching comment: DDPStrategy defaults to the
    # NCCL backend on GPU, unavailable on Windows and unnecessary here since
    # every config uses devices=1.
    strategy = DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto"

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[lr_monitor, model_summary, checkpoint_callback],
        gradient_clip_val=gradient_clip,
    )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(save_dir, "deep_cluster.ckpt"))
    print(f"RTDBN_DC (IIC head) saved to {save_dir}/deep_cluster.ckpt")

    _quality_check(model, train_ds, val_ds, test_ds, num_classes, num_workers)


def _get_embeddings_and_iic(model, dataset, num_workers, eval_batch_size=1024):
    """Runs the frozen encoder + trained IIC head, returning (embeddings, iic_assignments)."""
    loader = DataLoader(
        dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers
    )
    all_embeddings = []
    all_iic_assignments = []
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            emb = model._encode(x)
            all_embeddings.append(emb)
            iic_out = model.mlp_head(emb)[0]
            all_iic_assignments.append(torch.argmax(iic_out, dim=1))

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    iic_assignments = torch.cat(all_iic_assignments, dim=0).numpy()
    return embeddings, iic_assignments


def _consistent_silhouette(embeddings, assignments, cap=20000, seed=42):
    """Silhouette score, applying the same cap/subsample policy to both splits."""
    n = len(embeddings)
    if n > cap:
        return silhouette_score(embeddings, assignments, sample_size=cap, random_state=seed)
    return silhouette_score(embeddings, assignments, random_state=seed)


def _quality_check(model, train_ds, val_ds, test_ds, num_classes, num_workers):
    """k-means/silhouette sanity check on frozen embeddings, plus IIC cluster usage."""
    print("\nRunning k-means / IIC-usage sanity check on frozen embeddings...")
    model.eval()

    train_val_embeddings, iic_assignments = _get_embeddings_and_iic(
        model, ConcatDataset([train_ds, val_ds]), num_workers
    )

    unique_iic, counts_iic = np.unique(iic_assignments, return_counts=True)
    print(f"\nIIC head cluster usage: {len(unique_iic)}/{num_classes} clusters used")
    for c, n in zip(unique_iic, counts_iic):
        print(f"  IIC cluster {c}: {n} sequences ({100 * n / len(iic_assignments):.1f}%)")

    km = KMeans(n_clusters=num_classes, n_init=20, random_state=42)
    # Fit on a capped subsample for large splits, then predict on all points.
    fit_cap = 20000
    if len(train_val_embeddings) > fit_cap:
        rng = np.random.default_rng(42)
        fit_idx = rng.choice(len(train_val_embeddings), size=fit_cap, replace=False)
        km.fit(train_val_embeddings[fit_idx])
        km_assignments = km.predict(train_val_embeddings)
    else:
        km_assignments = km.fit_predict(train_val_embeddings)
    unique_km, counts_km = np.unique(km_assignments, return_counts=True)

    if len(unique_km) > 1:
        km_sil = _consistent_silhouette(train_val_embeddings, km_assignments)
    else:
        km_sil = -1.0

    probs = counts_km / counts_km.sum()
    km_entropy = scipy_entropy(probs) / np.log(num_classes)

    print("\nk-means baseline on train+val embeddings (for reference against "
          "the paper's reported ~0.50 silhouette / ~0.96 entropy):")
    print(f"  Clusters used:      {len(unique_km)}/{num_classes}")
    print(f"  Silhouette score:   {km_sil:.4f}")
    print(f"  Normalized entropy: {km_entropy:.4f}")
    for c, n in zip(unique_km, counts_km):
        print(f"  k-means cluster {c}: {n} sequences ({100 * n / len(km_assignments):.1f}%)")
    norms = np.linalg.norm(train_val_embeddings, axis=1)
    print(f"  Embedding L2 norm: mean={norms.mean():.3f} std={norms.std():.3f} "
          f"min={norms.min():.3f} max={norms.max():.3f}")

    if len(unique_iic) <= 2:
        print(
            "\nNOTE: IIC head collapsed to <=2 active cluster(s) -- a known "
            "failure mode, not a bug. k-means numbers above are the signal here."
        )

    if test_ds is None:
        print("\nNo held-out test split configured -- skipping generalization check.")
        return

    test_embeddings, _ = _get_embeddings_and_iic(
        model, test_ds, num_workers
    )
    test_assignments = km.predict(test_embeddings)
    unique_test, counts_test = np.unique(test_assignments, return_counts=True)

    if len(unique_test) > 1:
        test_sil = _consistent_silhouette(test_embeddings, test_assignments)
    else:
        test_sil = -1.0

    test_probs = counts_test / counts_test.sum()
    test_entropy = scipy_entropy(test_probs) / np.log(num_classes)

    print(f"\nHeld-out test split ({len(test_embeddings)} sequences, "
          f"centroids from train+val -- no leakage):")
    print(f"  Clusters used:      {len(unique_test)}/{num_classes}")
    print(f"  Silhouette score:   {test_sil:.4f}")
    print(f"  Normalized entropy: {test_entropy:.4f}")
    for c, n in zip(unique_test, counts_test):
        print(f"  k-means cluster {c}: {n} sequences ({100 * n / len(test_assignments):.1f}%)")
    print(f"  Train+val vs test silhouette gap: {abs(km_sil - test_sil):.4f} "
          f"({'similar' if abs(km_sil - test_sil) < 0.05 else 'notable gap'})")


def load_trained_rtdbn_dc(yml_conf):
    """Loads a frozen RTDBN encoder + trained RTDBN_DC head for inference/FLOPs use.

    Rebuilds RTDBN_DC with the same hyperparameters used at training time, then
    loads the state_dict saved by train_RTDBN_DC's final torch.save() call (deep_cluster.ckpt
    is a raw state_dict, not a PyTorch Lightning checkpoint, despite the ModelCheckpoint
    callback also writing to that same path during training - the final manual save overwrites it).
    """
    out_dir = yml_conf["output"]["out_dir"]
    encoder_dir = os.path.join(out_dir, "encoder")
    full_model_dir = os.path.join(out_dir, "full_model")

    rtdbn = _load_frozen_encoder(yml_conf, encoder_dir)

    num_classes = yml_conf["cluster"]["num_classes"]
    lr = yml_conf["cluster"]["training"]["learning_rate"]
    noise_stdev = float(yml_conf["cluster"]["gauss_noise_stdev"][0])
    lamb = float(yml_conf["cluster"]["lambda"])

    model = RTDBN_DC(rtdbn, num_classes=num_classes, lr=lr, noise_stdev=noise_stdev, lamb=lamb)

    ckpt_path = os.path.join(full_model_dir, "deep_cluster.ckpt")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model


def run_dc_outside(yml_fpath):
    yml_conf = read_yaml(yml_fpath)
    train_RTDBN_DC(yml_conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--yaml", help="YAML config file for RTDBN clustering (same file used for pretrain_rtdbn.py)."
    )
    args = parser.parse_args()
    run_dc_outside(args.yaml)
