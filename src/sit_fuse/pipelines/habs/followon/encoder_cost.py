"""Parameter counts and FLOPs per output pixel of a trained SIT-FUSE encoder (+ cluster heads).

One prediction batch from the first scene of the window is pushed through the encoder embedding
(hab_head_export._embed) and through the full model; FLOPs come from calflops (forward only).
Writes <out>/<name>.flops.json, merged into the Rep_Taxonomy resource summary.

  python -m sit_fuse.pipelines.habs.followon.encoder_cost -y HAB.yaml --instrument viirs \
      --model-config train_model_VIIRS.yaml --model-dir WEIGHTS --name conv_dbn \
      --start 2025-07-01 --end 2025-07-07 --out DIR
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os

LOG = logging.getLogger("encoder_cost")


def _params(m) -> int:
    return int(sum(p.numel() for p in m.parameters())) if m is not None else 0


def _encoder_part(model):
    """Encoder module of a DC / Heir_DC model (the first pretrained_model / encoder attribute found)."""
    m = model
    for _ in range(3):
        for k in ("pretrained_model", "encoder"):
            sub = getattr(m, k, None)
            if sub is not None and hasattr(sub, "parameters"):
                return sub
        m = getattr(m, "model", None) or getattr(m, "dc", None)
        if m is None:
            break
    return None


def _flops(fn_module, x):
    try:
        from calflops import calculate_flops
    except Exception as e:                       # noqa: BLE001
        LOG.warning("calflops unavailable (%s); FLOPs not counted", e)
        return None, None
    try:
        flops, macs, _ = calculate_flops(model=fn_module, args=[x], output_as_string=False,
                                         print_results=False, print_detailed=False)
        return float(flops), float(macs)
    except Exception as e:                       # noqa: BLE001
        LOG.warning("calflops failed (%s)", e)
        return None, None


def main(argv=None):
    import numpy as np
    import torch
    import yaml
    from sit_fuse.pipelines.habs import hab_inference_pipeline as hip
    from sit_fuse.pipelines.habs.uq.hab_head_export import _embed
    from sit_fuse.compute_resources import apply_compute_limits
    from sit_fuse.datasets.dataset_utils import get_prediction_dataset
    from sit_fuse.inference.generate_output import prep_model

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("--instrument", required=True)
    p.add_argument("--key", default="no_trop")
    p.add_argument("--model-config")
    p.add_argument("--model-dir")
    p.add_argument("--entries-from")
    p.add_argument("--name", required=True)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--batch", type=int, default=64, help="samples (pixels or tiles) in the counted batch")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(a.yaml) as f:
        yml = yaml.safe_load(f)
    inf = copy.deepcopy(yml.get("inference", {}) or {})
    if a.start:
        inf["start_date"] = a.start
        inf.pop("date_windows", None)
    if a.end:
        inf["end_date"] = a.end
    cfg_path = a.model_config or hip.model_config_path(yml, a.instrument, a.key)
    sf_yaml = hip.merge_compute_conf(yml, hip._read_yaml(cfg_path))
    apply_compute_limits(sf_yaml)
    out_dir = hip.instrument_out_dir(yml, a.instrument, a.key, sf_yaml)
    sf_yaml["output"]["out_dir"] = a.model_dir or hip.instrument_model_dir(yml, a.instrument, a.key, out_dir)
    hip.apply_data_overrides(sf_yaml, inf)
    src = sf_yaml
    if a.entries_from:
        src = copy.deepcopy(sf_yaml)
        other = hip._read_yaml(a.entries_from)
        for k in ("files_train", "files_test"):
            src["data"][k] = other["data"].get(k) or []
    tiled = bool(sf_yaml["data"].get("tile", False))
    data = None
    for e in hip.scene_entries(src, inf):
        data = get_prediction_dataset(sf_yaml, e)[0]
        if data is not None and data.data_full is not None:
            break
    if data is None or data.data_full is None:
        raise SystemExit("no readable scene in the window")
    model = prep_model(sf_yaml)
    model.eval()
    dev = hip._device(model)
    x = torch.as_tensor(np.asarray(data.data_full[:a.batch]), dtype=torch.float32, device=dev)
    pix = int(x.shape[0]) * (int(x.shape[2]) * int(x.shape[3]) if tiled and x.ndim == 4 else 1)

    class _Enc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, xx):
            return _embed(self.m, xx)

    with torch.no_grad():
        f_enc, m_enc = _flops(_Enc(model), x)
        f_all, m_all = _flops(model, x)
        e = _embed(model, x)
    enc = _encoder_part(model)
    rec = {"arm": a.name, "model_config": cfg_path, "model_dir": sf_yaml["output"]["out_dir"], "tiled": tiled,
           "input_shape": list(x.shape), "pixels_in_batch": pix, "embed_dim": int(e.reshape(e.shape[0], -1).shape[1]),
           "params_total": _params(model), "params_encoder": _params(enc) if enc is not None else None,
           "flops_encoder_batch": f_enc, "flops_total_batch": f_all,
           "flops_per_pixel": (f_enc / pix) if f_enc else None, "macs_per_pixel": (m_enc / pix) if m_enc else None,
           "flops_total_per_pixel": (f_all / pix) if f_all else None, "macs_total_per_pixel": (m_all / pix) if m_all else None}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.name}.flops.json"), "w") as f:
        json.dump(rec, f, indent=1)
    LOG.info("%s", rec)


if __name__ == "__main__":
    main()
