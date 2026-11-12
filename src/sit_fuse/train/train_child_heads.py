"""
Train the child heads of the hierarchical decoder (Heir_DC), each strictly on its parent's samples.

This is a stand-alone stage that runs after the main model (encoder + deep-cluster head) is trained.
The main model is loaded read-only and frozen (eval mode throughout), and nothing under encoder/ or
full_model/ is written. It can therefore run on the main model while seed variants are trained, and
again on every seed variant, each with its own config.

For every parent cluster p of the main model:
  1. Every training sample is assigned to a parent once, in fp32 and eval mode, with the same
     argmax the inference path uses (parent = argmax of the deep-cluster head).
  2. Child head p is trained only on the samples assigned to p. The loader serves only those rows,
     and Heir_DC.rows_are_key stops forward() from re-deriving membership under autocast or train
     mode. Its validation split is drawn from the same parent subset.
  3. Parents with fewer than min_samples samples get no head; inference then keeps parent * num_child,
     as before.
  4. Check: on the parent's validation rows, full-model inference (Heir_DC.forward(train=False)) must
     route every row to child head p (fine // num_child == p). The pass rate is recorded per parent.

The output is heir_fc.ckpt in the format Heir_DC / generate_output already read (state_dict["1"][key]
plus state_dict["labels"]), written to output.heir_model_dir (default full_model_heir). It also writes
child_heads_manifest.csv / .json, with per-parent sample counts, losses, child-cluster usage and the
consistency check, and parent_assignment.npy, the parent id of every training sample.

Config (cluster.heir.training; all optional except those used before):
  min_samples: 1000            parents with fewer samples get no child head
  max_samples_per_head: null   random cap on a parent's samples (seeded)
  val_percent: null            within-parent validation share (default data.val_percent)
  limit_train_batches: null    per-epoch batch cap (null = all batches of that parent; the legacy stage used 100 of the full set)
  epochs, batch_size, learning_rate (fallback cluster.training.learning_rate), weight_decay
  assign_batch_size: 8192      batch size of the parent-assignment pass
  number_heads: 1
  overwrite: false             true = retrain heads already in heir_fc.ckpt
  seed: null                   falls back to the top-level seed (sit_fuse.train.seeding)
  logger: none                 none | csv | wandb (wandb opens one run per parent)

Usage:
  python -m sit_fuse.train.train_child_heads -y model.yaml [--keys 3,17] [--overwrite] [--heir_model_dir DIR]
"""
if __name__ == '__main__':
    from sit_fuse.compute_resources import limit_env_from_argv
    limit_env_from_argv()

import argparse
import json
import os

import numpy as np
import torch


def _training_conf(yml_conf):
    tr = dict(yml_conf["cluster"]["heir"].get("training", {}) or {})
    tr.setdefault("min_samples", 1000)
    tr.setdefault("max_samples_per_head", None)
    tr.setdefault("val_percent", None)
    tr.setdefault("limit_train_batches", None)
    tr.setdefault("epochs", 30)
    tr.setdefault("batch_size", 1000)
    tr.setdefault("learning_rate", yml_conf["cluster"].get("training", {}).get("learning_rate", 1e-4))
    tr.setdefault("weight_decay", 0.0)
    tr.setdefault("assign_batch_size", 8192)
    tr.setdefault("number_heads", 1)
    tr.setdefault("overwrite", False)
    tr.setdefault("logger", "none")
    tr.setdefault("accelerator", "auto")
    tr.setdefault("devices", 1)
    tr.setdefault("precision", "32-true")
    if tr.get("val_percent") is None:
        tr["val_percent"] = float(yml_conf["data"].get("val_percent", 0.1))
    return tr


# --------------------------------------------------------------------------- #
# Parent assignment
# --------------------------------------------------------------------------- #
@torch.no_grad()
def assign_parents(parent_model, data, batch_size=8192, device=None):
    """
    Parent cluster of every sample: argmax over the deep-cluster head output, fp32, eval mode.
    Returns an int64 array (N,) for per-sample outputs, or (N, P) for per-pixel (tiled) outputs.
    """
    was_training = parent_model.training
    parent_model.eval()
    if device is None:
        try:
            device = next(parent_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    out = []
    n = data.shape[0]
    for i in range(0, n, int(batch_size)):
        x = torch.as_tensor(np.asarray(data[i:i + int(batch_size)]), dtype=torch.float32).to(device)
        y = parent_model.forward(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        if y.ndim == 2:
            lab = torch.argmax(y, dim=1)
        elif y.ndim == 4:            # (B, K, H, W) -> (B, H*W)
            lab = torch.argmax(y, dim=1).flatten(start_dim=1)
        elif y.ndim == 3:            # (B, H*W, K) as in some segmentors
            lab = torch.argmax(y, dim=-1)
        else:
            lab = torch.argmax(y.reshape(y.shape[0], -1), dim=1)
        out.append(lab.cpu().numpy().astype(np.int64))
    if was_training:
        parent_model.train()
    return np.concatenate(out, axis=0) if out else np.empty(0, dtype=np.int64)


def parent_index_sets(parents):
    """{str(parent): sample indices}. Per-pixel labels: samples (tiles) containing that parent."""
    sets = {}
    if parents.ndim == 1:
        order = np.argsort(parents, kind="stable")
        uniq, starts, counts = np.unique(parents[order], return_index=True, return_counts=True)
        for u, st, c in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
            sets[str(int(u))] = np.sort(order[st:st + c])
    else:
        for u in np.unique(parents).tolist():
            sets[str(int(u))] = np.nonzero((parents == u).any(axis=1))[0]
    return dict(sorted(sets.items(), key=lambda kv: int(kv[0])))


def split_parent_subset(idx, val_percent, max_samples, seed):
    """Shuffle a parent's sample indices (seeded), cap them, and split train / val within the parent."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(np.asarray(idx))
    if max_samples is not None and len(idx) > int(max_samples):
        idx = idx[:int(max_samples)]
    n_val = int(round(len(idx) * float(val_percent)))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def make_datamodule(train_x, val_x, batch_size, num_workers=0, seed=0):
    """Train / val loaders over one parent's rows (shuffled, seeded training batches)."""
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from sit_fuse.datasets.simple_dataset import SimpleDataset, batch_collate

    def _ds(x):
        return SimpleDataset(x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x)))

    class _ParentDataModule(pl.LightningDataModule):
        def train_dataloader(self):
            g = torch.Generator()
            g.manual_seed(int(seed))
            bs = max(1, min(int(batch_size), len(train_x)))
            return DataLoader(_ds(train_x), batch_size=bs, shuffle=True, drop_last=len(train_x) > bs,
                              num_workers=int(num_workers), collate_fn=batch_collate, generator=g,
                              pin_memory=torch.cuda.is_available(), persistent_workers=int(num_workers) > 0)

        def val_dataloader(self):
            if len(val_x) == 0:
                return []
            bs = max(1, min(int(batch_size), len(val_x)))
            return DataLoader(_ds(val_x), batch_size=bs, shuffle=False, drop_last=False,
                              num_workers=0, collate_fn=batch_collate)

    return _ParentDataModule()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _freeze_except(model, key):
    for p in model.pretrained_model.parameters():
        p.requires_grad = False
    enc = getattr(model, "encoder", None)
    if isinstance(enc, torch.nn.Module):
        for p in enc.parameters():
            p.requires_grad = False
    for k, head in model.clust_tree["1"].items():
        if head is None:
            continue
        for p in head.parameters():
            p.requires_grad = (k == key)


@torch.no_grad()
def _check_routing(model, x, key, num_child, batch_size=4096):
    """Share of rows that full inference routes to child head `key` (fine // num_child == key), and child usage."""
    model.eval()
    dev = next(model.clust_tree["1"][key].parameters()).device
    ok, n, usage = 0, 0, np.zeros(num_child, dtype=np.int64)
    rows_are_key = model.rows_are_key
    model.rows_are_key = False
    try:
        for i in range(0, x.shape[0], batch_size):
            xb = torch.as_tensor(np.asarray(x[i:i + batch_size]), dtype=torch.float32).to(dev)
            out = model.forward(xb, train=False)
            if out is None:
                continue
            fine = out[1].reshape(-1).detach().cpu().numpy().astype(np.int64)
            ok += int((fine // num_child == int(key)).sum())
            n += fine.size
            usage += np.bincount(fine % num_child, minlength=num_child)[:num_child]
    finally:
        model.rows_are_key = rows_are_key
    return (ok / n if n else float("nan")), usage


def _save(model, lab_full, path):
    from sit_fuse.models.deep_cluster.heir_dc import get_state_dict
    tmp = path + ".tmp"
    torch.save(get_state_dict(model.clust_tree, lab_full), tmp)
    os.replace(tmp, path)


def _make_trainer(tr, save_dir, key, yml_conf):
    import pytorch_lightning as pl
    from sit_fuse.compute_resources import get_strategy
    logger = False
    kind = str(tr.get("logger", "none")).lower()
    if kind == "csv":
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(save_dir, name="child_heads", version=f"parent_{key}")
    elif kind == "wandb":
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(project=yml_conf["logger"].get("project"), save_dir=save_dir, name=f"child_{key}")
    kw = dict(default_root_dir=save_dir, accelerator=tr["accelerator"], devices=tr["devices"],
              strategy=get_strategy(tr["devices"], tr["accelerator"]), precision=tr["precision"],
              max_epochs=int(tr["epochs"]), gradient_clip_val=tr.get("gradient_clip_val",
              yml_conf.get("encoder", {}).get("training", {}).get("gradient_clip_val")),
              logger=logger, enable_checkpointing=False, enable_model_summary=False,
              enable_progress_bar=bool(tr.get("progress_bar", True)), num_sanity_val_steps=0)
    if tr.get("limit_train_batches"):
        kw["limit_train_batches"] = tr["limit_train_batches"]
    if tr.get("max_steps_per_head"):
        kw["max_steps"] = int(tr["max_steps_per_head"])
    return pl.Trainer(**kw)


def train_child_heads(model, data, yml_conf, heir_dir, keys=None, overwrite=None, seed=None, n_visible=None):
    """
    Core loop (model = Heir_DC with a frozen pretrained_model; data = training samples, N x F).
    Returns the manifest rows. Kept free of config-file handling so other use-cases and tests can call it.
    """
    from sit_fuse.models.deep_cluster.multi_prototypes import MultiPrototypes
    tr = _training_conf(yml_conf)
    if overwrite is None:
        overwrite = bool(tr["overwrite"])
    seed = 0 if seed is None else int(seed)
    num_child = int(model.num_classes)
    os.makedirs(heir_dir, exist_ok=True)
    ckpt = os.path.join(heir_dir, "heir_fc.ckpt")

    dev = torch.device("cuda") if (torch.cuda.is_available() and str(tr["accelerator"]) in ("gpu", "cuda", "auto")) else torch.device("cpu")
    model.pretrained_model.to(dev)
    if isinstance(getattr(model, "encoder", None), torch.nn.Module):
        model.encoder.to(dev)
    parents = assign_parents(model.pretrained_model, data, tr["assign_batch_size"], dev)
    np.save(os.path.join(heir_dir, "parent_assignment.npy"), parents)
    sets = parent_index_sets(parents)
    per_sample = parents.ndim == 1
    if not per_sample:
        print("Per-pixel parent labels: each tile trains every parent it contains; rows are filtered in forward().")
    lab_full = {k: torch.from_numpy(v) for k, v in sets.items()}

    # Resume: keep heads already trained for the same parents, unless overwrite.
    prev = model.clust_tree.get("1", {})
    model.clust_tree["1"] = {k: (prev.get(k) if (not overwrite and prev.get(k) is not None) else None) for k in sets}
    model.lab_full = lab_full
    model.rows_are_key = per_sample
    model.keep_parent_eval = True
    model.lr = float(tr["learning_rate"])
    model.weight_decay = float(tr["weight_decay"])
    if n_visible is None:
        raise ValueError("n_visible (child-head input size) is required")

    rows = []
    for c, (key, idx) in enumerate(sets.items()):
        row = dict(parent=int(key), n_parent=int(len(idx)), n_train=0, n_val=0, status="", train_loss=np.nan,
                   val_loss=np.nan, routing_ok=np.nan, children_used=0, child_counts="")
        if keys is not None and key not in keys:
            row["status"] = "not_selected"
            rows.append(row)
            continue
        if len(idx) < int(tr["min_samples"]):
            model.clust_tree["1"][key] = None
            row["status"] = "too_few_samples"
            rows.append(row)
            continue
        if model.clust_tree["1"].get(key) is not None:
            row["status"] = "kept_existing"
            rows.append(row)
            continue
        print(f"CHILD HEAD {c + 1} / {len(sets)}: parent {key}, {len(idx)} samples")
        tr_idx, va_idx = split_parent_subset(idx, tr["val_percent"], tr["max_samples_per_head"], [seed, int(key)])
        model.rng = np.random.default_rng([seed, int(key), 1])
        head = MultiPrototypes(n_visible, num_child, int(tr["number_heads"]))
        model.clust_tree["1"][key] = head
        model.module_list.append(head)
        _freeze_except(model, key)
        model.key = key
        dm = make_datamodule(data[tr_idx], data[va_idx], tr["batch_size"], tr.get("num_loader_workers", 0), seed + int(key))
        trainer = _make_trainer(tr, heir_dir, key, yml_conf)
        trainer.fit(model, dm)
        m = trainer.callback_metrics
        row.update(n_train=int(len(tr_idx)), n_val=int(len(va_idx)), status="trained",
                   train_loss=float(m["train_loss"]) if "train_loss" in m else np.nan,
                   val_loss=float(m["val_loss"]) if "val_loss" in m else np.nan)
        chk = data[va_idx] if len(va_idx) else data[tr_idx[:min(len(tr_idx), 4096)]]
        model.to(dev)
        if per_sample:
            ok, usage = _check_routing(model, chk, key, num_child)
            row.update(routing_ok=ok, children_used=int((usage > 0).sum()), child_counts=";".join(map(str, usage.tolist())))
            if ok < 1.0:
                print(f"WARNING parent {key}: only {ok:.4f} of its rows are routed to its child head at inference")
        head.eval()
        for p in head.parameters():
            p.requires_grad = False
        _save(model, lab_full, ckpt)
        rows.append(row)

    _save(model, lab_full, ckpt)
    _write_manifest(rows, heir_dir, dict(seed=seed, num_child=num_child, n_samples=int(len(parents)),
                                         n_parents=len(sets), per_sample=per_sample, min_samples=int(tr["min_samples"])))
    return rows


def _write_manifest(rows, heir_dir, meta):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(heir_dir, "child_heads_manifest.csv"), index=False)
    tr = df[df.status == "trained"] if len(df) else df
    meta = dict(meta, n_trained=int(len(tr)), n_too_few=int((df.status == "too_few_samples").sum()) if len(df) else 0,
                min_routing_ok=float(tr.routing_ok.min()) if len(tr) else None)
    with open(os.path.join(heir_dir, "child_heads_manifest.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    print("CHILD HEADS", json.dumps(meta, default=float))


def run(yml_conf, keys=None, overwrite=None, heir_model_dir=None):
    from sit_fuse.compute_resources import apply_compute_limits, num_local_processes
    from sit_fuse.datasets.dataset_utils import get_train_dataset_sf
    from sit_fuse.models.deep_cluster.heir_dc import Heir_DC
    from sit_fuse.train.seeding import apply_seed
    from sit_fuse.train import train_heirarchichal_deep_cluster as thdc

    tr = _training_conf(yml_conf)
    apply_compute_limits(yml_conf, default_num_threads=5,
                         num_processes=num_local_processes(tr.get("devices", 1), tr.get("accelerator", "auto")))
    seed = apply_seed(yml_conf, "heir")
    if heir_model_dir:
        yml_conf.setdefault("output", {})["heir_model_dir"] = heir_model_dir
    save_dir = thdc.model_save_dir(yml_conf)
    heir_dir = thdc.heir_model_dir(yml_conf)
    ckpt_main = os.path.join(save_dir, "full_model", "deep_cluster.ckpt")
    if not os.path.exists(ckpt_main):
        raise FileNotFoundError(f"main model checkpoint not found: {ckpt_main}")

    out_dir = yml_conf["output"]["out_dir"]
    sampling = yml_conf["data"].get("sampling") or {}
    if not os.path.exists(os.path.join(out_dir, "train_data.npy")) and sampling.get("seed") is None:
        print("WARNING: no cached training sample (data.sampling.cache_train) and no data.sampling.seed: "
              "the child heads see a different random sample than the main model was trained on.")
    dataset = get_train_dataset_sf(yml_conf)

    encoder, encoder_type, _, _ = thdc.load_frozen_encoder(yml_conf, dataset, tr["accelerator"])
    heir_ckpt = os.path.join(heir_dir, "heir_fc.ckpt")
    model = Heir_DC(None, pretrained_model_path=ckpt_main, num_classes=int(yml_conf["cluster"]["heir"]["num_classes"]),
                    yml_conf=yml_conf, lr=float(tr["learning_rate"]), encoder_type=encoder_type,
                    number_heads=int(tr["number_heads"]), min_samples=int(tr["min_samples"]), encoder=encoder,
                    clust_tree_ckpt=heir_ckpt if os.path.exists(heir_ckpt) else None, generate_labels=False)
    n_visible = thdc.head_input_size(yml_conf, model, encoder)
    data = dataset.data_full
    data = data.numpy() if torch.is_tensor(data) else data
    return train_child_heads(model, data, yml_conf, heir_dir, keys=keys, overwrite=overwrite,
                             seed=seed if seed is not None else 0, n_visible=n_visible)


def main(argv=None):
    from sit_fuse.utils import read_yaml
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--keys", default=None, help="comma-separated parent ids to (re)train; default all")
    ap.add_argument("--overwrite", action="store_true", help="retrain heads already in heir_fc.ckpt")
    ap.add_argument("--heir_model_dir", default=None, help="override output.heir_model_dir")
    ap.add_argument("-n", "--num-threads", default=None)
    a = ap.parse_args(argv)
    if a.num_threads is not None:
        os.environ["SIT_FUSE_NUM_THREADS"] = str(a.num_threads)
    keys = set(k.strip() for k in a.keys.split(",")) if a.keys else None
    run(read_yaml(a.yaml), keys=keys, overwrite=True if a.overwrite else None, heir_model_dir=a.heir_model_dir)


if __name__ == "__main__":
    main()
