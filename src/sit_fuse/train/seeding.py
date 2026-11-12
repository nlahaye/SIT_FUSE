"""
Run-level seeding for SIT-FUSE training stages.

Seed resolution (first found wins):
  <stage section>.training.seed   e.g. encoder.training.seed, cluster.training.seed, cluster.heir.training.seed
  seed                            top-level key (written by sit_fuse.train.seed_variants)
If none is set, nothing is seeded (previous behaviour).

Seeds Python, NumPy and torch (CPU and CUDA) via pytorch_lightning.seed_everything(workers=True),
so model initialisation, shuffling, dropout and Gibbs sampling differ between seed variants and
repeat for the same seed. Sample selection has its own seed (data.sampling.seed); see
seed_variants.py for how the two are combined.
"""
import os
import random

import numpy as np

_STAGE_KEYS = {
    "encoder": ("encoder", "training"),
    "cluster": ("cluster", "training"),
    "heir": ("cluster", "heir", "training"),
}


def _get(conf, keys):
    cur = conf
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def resolve_seed(yml_conf, stage=None):
    if stage in _STAGE_KEYS:
        sect = _get(yml_conf, _STAGE_KEYS[stage])
        if isinstance(sect, dict) and sect.get("seed") is not None:
            return int(sect["seed"])
    if yml_conf.get("seed") is not None:
        return int(yml_conf["seed"])
    return None


def apply_seed(yml_conf, stage=None, offset=0):
    """Seed everything for this stage; returns the seed used (or None). `offset` separates stages."""
    seed = resolve_seed(yml_conf, stage)
    if seed is None:
        return None
    seed = int(seed) + int(offset)
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed, workers=True)
    except Exception:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
    print("SEED", stage, seed)
    return seed
