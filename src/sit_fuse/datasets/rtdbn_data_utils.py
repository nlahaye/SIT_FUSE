"""Shared data-loading utilities for the RTDBN encoder and clustering scripts."""
import glob
import os

import numpy as np
import torch
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sit_fuse.datasets.sf_temporal_dataset import SFTemporalDataset


def set_all_seeds(seed: int = 42) -> None:
    """Seeds numpy and torch RNGs for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_episode_files(data_dir, val_percent, test_percent=0.0, seed=42):
    """Splits sorted episode files into train/val/(test) via a seeded permutation."""
    all_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npy")))
    if len(all_files) == 0:
        raise ValueError(f"No episode_*.npy files found in {data_dir}")

    np.random.seed(seed)
    indices = np.random.permutation(len(all_files))

    n_test = int(len(all_files) * test_percent)
    n_val = int(len(all_files) * val_percent)
    n_train = len(all_files) - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"val_percent ({val_percent}) + test_percent ({test_percent}) "
            f"leaves no training episodes out of {len(all_files)} files."
        )

    train_files = [all_files[i] for i in indices[:n_train]]
    val_files = [all_files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [all_files[i] for i in indices[n_train + n_val:]]

    return train_files, val_files, test_files


def fit_and_save_scaler(train_files, scaler_out_path=None, model_type="variance_gaussian"):
    """Fits a scaler on train files only, and persists it if a path is given.

    RTVarianceGaussianRBM's learned per-feature sigma needs zero-mean/unit-
    variance input to actually differentiate across channels -- MinMax-to-
    [0,1] left sigma converging to a near-uniform band regardless of true
    per-channel variance, which tanked R^2 despite a good-looking MSE (see
    SIT-FUSE-RTRBM's notes/MSE_VS_R2_MODEL_COMPARISON.md and
    notes/STANDARDSCALER_FIX_RESOLUTION.md, task #70). Every RTDBN config in
    this pipeline trains "variance_gaussian" as its first (and currently
    only) layer, so this defaults to StandardScaler; Bernoulli/fixed-variance
    Gaussian layers, if ever configured here, still want MinMax.
    """
    scaler_cls = StandardScaler if model_type == "variance_gaussian" else MinMaxScaler
    scaler = scaler_cls()
    for fpath in train_files:
        scaler.partial_fit(np.load(fpath).astype(np.float32))

    if scaler_out_path is not None:
        os.makedirs(os.path.dirname(scaler_out_path), exist_ok=True)
        dump(scaler, scaler_out_path)

    return scaler


def load_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"No scaler found at {scaler_path}. Run pretrain_rtdbn.py first "
            f"-- it fits and saves the scaler that downstream stages reuse."
        )
    return load(scaler_path)


def build_temporal_dataset(files, scaler, seq_len, fill_value, do_shuffle):
    """Builds a boundary-preserving SFTemporalDataset from episode files."""
    ds = SFTemporalDataset()
    ds.read_and_preprocess_data(
        filenames=files,
        seq_len=seq_len,
        scaler=scaler,
        scale=True,
        train_scaler=False,
        do_shuffle=do_shuffle,
        fill_value=fill_value,
    )
    return ds


def build_rtdbn_splits(yml_conf, scaler_out_path=None, scaler_in_path=None):
    """Builds train/val/test SFTemporalDatasets for the RTDBN pipeline."""
    data_dir = yml_conf["data"]["data_dir"]
    seq_len = int(yml_conf["data"]["seq_len"])
    val_percent = float(yml_conf["data"]["val_percent"])
    test_percent = float(yml_conf["data"].get("test_percent", 0.0))
    fill_value = yml_conf["data"]["fill_value"]
    seed = int(yml_conf["data"].get("split_seed", 42))

    train_files, val_files, test_files = split_episode_files(
        data_dir, val_percent, test_percent, seed=seed
    )

    # First layer's model type determines the scaler convention (see
    # fit_and_save_scaler's docstring); later stacked layers, if any, train
    # on the first layer's hidden output, not this raw-data scaler.
    model_type = yml_conf.get("rtdbn", {}).get("model_type", ["variance_gaussian"])[0]

    if scaler_in_path is not None:
        scaler = load_scaler(scaler_in_path)
    else:
        scaler = fit_and_save_scaler(
            train_files, scaler_out_path=scaler_out_path, model_type=model_type
        )

    train_ds = build_temporal_dataset(
        train_files, scaler, seq_len, fill_value, do_shuffle=True
    )
    val_ds = build_temporal_dataset(
        val_files, scaler, seq_len, fill_value, do_shuffle=False
    )
    test_ds = None
    if len(test_files) > 0:
        test_ds = build_temporal_dataset(
            test_files, scaler, seq_len, fill_value, do_shuffle=False
        )

    return train_ds, val_ds, test_ds
