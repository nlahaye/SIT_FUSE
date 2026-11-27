"""RT-DBN encoder over pixel + neighbourhood temporal stacks.

Input rows come from SFDataset (tile: False) on a ``temporal_stack`` scene, i.e. the same
(2p+1) x (2p+1) pixel + neighbourhood windows the DBN sees, flattened (wy, wx, C) with
C = L * Cb time-major (Cb = bands + age channel). They are reshaped to sequences
(N, L, (2p+1)^2 * Cb): one visible vector per day holding the whole window for that day.

The learnergy RTDBN maps (N, L, V) -> (N, L, H) layer by layer; the per-pixel embedding is
the mean over time (architecture default, as in RTDBN_DC), or the last step / both.

Config (model YAML):
  encoder_type: rtdbn
  data.reader_type: temporal_stack, data.reader_kwargs.temporal_days: L, data.pixel_padding: p
  rtdbn: {model_type, n_hidden, gibbs_steps, temp, n_visible (per step, optional check),
          seq_len (default reader_kwargs.temporal_days), embedding: mean | last | mean_last,
          warmup_epochs}
  encoder.training: learning_rate, momentum, weight_decay, nesterov_accel, epochs (per layer)
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from sit_fuse.models.encoders.rtdbn_pl import RTDBN_PL


def seq_len_from_conf(yml_conf) -> int:
    r = yml_conf.get("rtdbn", {}) or {}
    if r.get("seq_len"):
        return int(r["seq_len"])
    return int(((yml_conf.get("data", {}) or {}).get("reader_kwargs", {}) or {}).get("temporal_days", 8))


def window_from_conf(yml_conf) -> int:
    p = int((yml_conf.get("data", {}) or {}).get("pixel_padding", 1))
    return (2 * p + 1) ** 2


def to_sequence(x: torch.Tensor, seq_len: int, window: int) -> torch.Tensor:
    """(N, window * L * Cb) or (N, wy, wx, L*Cb) -> (N, L, window * Cb)."""
    n = x.shape[0]
    x = x.reshape(n, -1)
    f = x.shape[1]
    if f % (window * seq_len):
        raise ValueError(f"RT-DBN input width {f} is not window ({window}) x seq_len ({seq_len}) x channels")
    cb = f // (window * seq_len)
    return x.view(n, window, seq_len, cb).permute(0, 2, 1, 3).reshape(n, seq_len, window * cb)


class RTDBNSeqEncoder(nn.Module):
    def __init__(self, rtdbn, seq_len: int, window: int, embedding: str = "mean"):
        super().__init__()
        self.rtdbn = rtdbn
        self.models = rtdbn.models
        self.seq_len = int(seq_len)
        self.window = int(window)
        if embedding not in ("mean", "last", "mean_last"):
            raise ValueError(f"rtdbn.embedding must be mean | last | mean_last, got {embedding}")
        self.embedding = embedding
        n_hidden = self.models[-1].n_hidden
        self.out_dim = int(n_hidden * (2 if embedding == "mean_last" else 1))

    def sequence(self, x):
        return to_sequence(x, self.seq_len, self.window)

    def hidden(self, x):
        h = self.sequence(x)
        for m in self.models:
            h = m.forward(h)                       # (N, L, H_i)
        return h

    def forward(self, x):
        h = self.hidden(x)
        if self.embedding == "last":
            return h[:, -1]
        if self.embedding == "mean_last":
            return torch.cat([h.mean(dim=1), h[:, -1]], dim=1)
        return h.mean(dim=1)


class RTDBN_SF_PL(RTDBN_PL):
    """RTDBN_PL on flat SFDataModule batches; earlier layers are applied to the sequence."""

    def __init__(self, model, save_dir, previous_layers=None, seq_len=8, window=9, **kw):
        super().__init__(model, save_dir, None, **kw)
        self.prev = nn.ModuleList(previous_layers) if previous_layers else None
        self.seq_len = int(seq_len)
        self.window = int(window)

    def _seq(self, batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        h = to_sequence(x.float(), self.seq_len, self.window)
        if self.prev is not None:
            with torch.no_grad():
                for m in self.prev:
                    h = m.forward(h)
        return (h, None)

    def training_step(self, batch, batch_idx):
        return super().training_step(self._seq(batch), batch_idx)

    def validation_step(self, batch, batch_idx):
        return super().validation_step(self._seq(batch), batch_idx)

    def on_validation_epoch_end(self):
        # the parent saves the single layer as rtdbn.ckpt; the full RTDBN is saved by the caller
        pass


def build_rtdbn(yml_conf, n_visible: int):
    from learnergy.models.temporal.rtdbn import RTDBN
    r = yml_conf["rtdbn"]
    tr = yml_conf["encoder"]["training"]
    acc = tr.get("accelerator", "gpu")
    return RTDBN(model=tuple(r["model_type"]), n_visible=int(n_visible), n_hidden=tuple(r["n_hidden"]),
                 steps=tuple(r["gibbs_steps"]), learning_rate=tuple(tr["learning_rate"]),
                 momentum=tuple(tr["momentum"]), decay=tuple(tr["weight_decay"]),
                 temperature=tuple(r["temp"]), use_gpu=(acc == "gpu"))


def step_visible(yml_conf, n_features: int) -> int:
    L, w = seq_len_from_conf(yml_conf), window_from_conf(yml_conf)
    if n_features % (L * w):
        raise ValueError(f"input width {n_features} is not divisible by seq_len {L} x window {w}")
    v = n_features // L
    want = (yml_conf.get("rtdbn", {}) or {}).get("n_visible")
    if want is not None and int(want) != v:
        raise ValueError(f"rtdbn.n_visible = {want} but the temporal stack gives {v} per step")
    return v


def encoder_dir(yml_conf) -> str:
    save_dir = yml_conf["output"]["out_dir"]
    if yml_conf.get("logger", {}).get("use_wandb"):
        save_dir = os.path.join(save_dir, yml_conf["logger"]["log_out_dir"])
    return os.path.join(save_dir, "encoder")


def load_rtdbn_encoder(yml_conf, n_features: int, ckpt_dir: str | None = None, freeze: bool = True):
    """Frozen RTDBNSeqEncoder from <encoder_dir>/rtdbn.ckpt."""
    rtdbn = build_rtdbn(yml_conf, step_visible(yml_conf, n_features))
    path = os.path.join(ckpt_dir or encoder_dir(yml_conf), "rtdbn.ckpt")
    rtdbn.load_state_dict(torch.load(path, map_location="cpu"))
    enc = RTDBNSeqEncoder(rtdbn, seq_len_from_conf(yml_conf), window_from_conf(yml_conf),
                          (yml_conf.get("rtdbn", {}) or {}).get("embedding", "mean"))
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
    return enc


def pretrain_RTDBN(yml_conf, dataset):
    """Layer-wise RT-DBN pretraining on SFDataModule rows; writes <encoder_dir>/rtdbn.ckpt."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
    from sit_fuse.compute_resources import get_strategy, training_checkpoint

    dataset.setup()
    rtdbn = build_rtdbn(yml_conf, step_visible(yml_conf, int(dataset.n_visible)))
    L, w = seq_len_from_conf(yml_conf), window_from_conf(yml_conf)
    r = yml_conf["rtdbn"]
    tr = yml_conf["encoder"]["training"]
    epochs = tr["epochs"] if isinstance(tr["epochs"], (list, tuple)) else [tr["epochs"]] * len(rtdbn.models)
    lr, mom, dec = tr["learning_rate"], tr["momentum"], tr["weight_decay"]
    nest = tr.get("nesterov_accel", [False] * len(rtdbn.models))
    save_dir = encoder_dir(yml_conf)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(rtdbn.models)):
        prev = list(rtdbn.models[:i]) if i > 0 else None
        model = RTDBN_SF_PL(rtdbn.models[i], save_dir, prev, seq_len=L, window=w,
                            learning_rate=float(lr[i]), momentum=float(mom[i]),
                            nesterov_accel=bool(nest[i]) and float(mom[i]) > 0, decay=float(dec[i]),
                            warmup_epochs=int(r.get("warmup_epochs", 15)))
        trainer = pl.Trainer(default_root_dir=save_dir, accelerator=tr["accelerator"], devices=tr["devices"],
                             strategy=get_strategy(tr["devices"], tr["accelerator"]),
                             precision=tr.get("precision", "32"), max_epochs=int(epochs[i]),
                             callbacks=[LearningRateMonitor(logging_interval="step"), ModelSummary(max_depth=2),
                                        training_checkpoint(save_dir, f"encoder_rtdbn_{i}", int(epochs[i]))],
                             gradient_clip_val=tr.get("gradient_clip_val"), logger=False)
        trainer.fit(model, dataset)
        rtdbn.models[i] = model.model
        torch.save(rtdbn.state_dict(), os.path.join(save_dir, "rtdbn.ckpt"))
    torch.save(rtdbn.state_dict(), os.path.join(save_dir, "rtdbn.ckpt"))
    return rtdbn
