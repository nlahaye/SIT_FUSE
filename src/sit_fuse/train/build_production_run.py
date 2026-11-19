#!/usr/bin/env python
"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Apply trained UQ pipelines (TEMPO / GOES smoke, fire) to a new period with no reference data, and plan the whole run
for sit_fuse.local_launcher: head export (GPU) -> inference (model x shard) -> geo products -> [fusion] -> overlay
animations.

    python build_production_run.py production_run.yaml            # write configs + plan, print the launcher command
    python build_production_run.py production_run.yaml --run [-- --dry-run | --only 'infer*' ...]
    python -m sit_fuse.local_launcher status <output_root>/launch/production_plan.yaml

For every pipeline (an evaluated UQ config) it writes, under <output_root>/<pipeline>/:
  scenes.yaml     unlabeled scene file: one record per input in [start, end) (and in bbox) with scene_id, head_outputs,
                  sit_fuse_input, sit_fuse_timestamp, date, product, classifier_split=production
  uq_config.yaml  the evaluated UQ config with output_dir -> <output_root>/<pipeline>, the trained model, calibrators,
                  thresholds and excluded clusters reused (inference.training_summary_path -> the training run), scene
                  file -> scenes.yaml, geo export reading the training run's area_interval_calibration.json
and, per distinct head export (sensor x SIT-FUSE model x head settings; shared by smoke and fire when they use the
same encoder), <output_root>/configs/export_<name>_<hash>_<shard>.yaml for export_head_outputs.py with `files:` set
to the inputs whose head outputs do not exist yet. The head export settings come from the pipeline's original export
config (export_config), else from the tags of the training head outputs, so they match training (inference checks).
With a `fusion:` section ({name: {config: <evaluation fusion config>, pipelines: {tempo: <pipe>, goes: <pipe>},
shards, cpus, grids}}) it writes <output_root>/configs/fusion_<name>.yaml (sources -> the production geo dirs,
fusion.models_from -> the evaluation fusion output_dir, references {}, output.splits [production]) and adds
fusion_apply (shards, after both geo exports) -> fusion_merge jobs; fused rasters land in <output_root>/fusion_<name>/.
With an `animation:` section it also writes <output_root>/configs/overlay_animation.yaml (make_overlay_animations.py)
with the product geo dirs (and fused products, sensor: fused) filled in.

The plan reuses plan_uq_jobs.build_plan for inference and geo export (stages [infer, geo]; no evaluation, since there
is no reference), adds the head-export jobs before inference and the animation jobs after the geo export. Rerunning
is safe: existing head outputs are not re-exported, inference reuses its scene checkpoints and launcher markers skip
finished jobs. Thresholds: per-group thresholds (calibration.threshold_groups) apply when the new scenes fall in a
calibrated group (season groups follow the date; set split_stratum for event groups); otherwise the pooled threshold.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from uq_viz.overlay import file_bounds_4326, time_from_name, utc

TRAIN_DIR = Path(__file__).resolve().parent
DEFAULT_EXCLUDE = [".RGB.tif", ".UV_RGB.tif", ".head_outputs.tif", ".clust.", "full_geo", ".proba"]


def _read(p):
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(d, sort_keys=False, width=200))
    return p


def _abs(base, p):
    if p is None:
        return None
    p = Path(str(p)).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def list_inputs(sc, t0, t1, bbox, base):
    """Inputs of one sensor in [t0, t1) (and intersecting bbox [W, E, S, N])."""
    dirs = sc["inputs"] if isinstance(sc["inputs"], list) else [sc["inputs"]]
    pat = sc.get("pattern", "*.tif")
    rec = bool(sc.get("recursive", False))
    excl = sc.get("exclude", DEFAULT_EXCLUDE)
    cand = []
    for d in dirs:
        for p in sorted(glob.glob(str(Path(_abs(base, d)) / ("**" if rec else "") / pat), recursive=rec)):
            n = Path(p).name
            if any(e in n for e in excl):
                continue
            t = time_from_name(n)
            if t is None or not (t0 <= t < t1):
                continue
            cand.append((p, t))
    cand.sort(key=lambda x: (x[1], x[0]))
    every = sc.get("every_minutes")         # optional subsampling: first input per <every_minutes> UTC bin
    if every:
        off = pd.Timedelta(minutes=float(sc.get("offset_minutes", 0)))
        bins = {}
        for p, t in cand:
            bins.setdefault((t - off).floor(f"{int(every)}min"), []).append((p, t))
        groups = list(bins.values())
    else:
        groups = [[c] for c in cand]
    out = []
    for g in groups:
        for p, t in g:                       # first input of the bin that intersects the bbox
            if bbox is not None:
                b = file_bounds_4326(p)
                if b is None or not (b[0] < bbox[1] and b[2] > bbox[0] and b[1] < bbox[3] and b[3] > bbox[2]):
                    continue
            out.append((p, t))
            break
    return out


def head_settings(pipe, uq, base):
    """sit_fuse_yaml + head_export + georef for the pipeline: its export config, else the training head-output tags."""
    if pipe.get("export_config"):
        ex = _read(_abs(base, pipe["export_config"]))
        return {"sit_fuse_yaml": ex["sit_fuse_yaml"], "head_export": ex.get("head_export") or {},
                "georef": {k: v for k, v in (ex.get("georef") or {}).items()
                           if k not in ("reference_geotiffs", "subset_inds")}}
    import rasterio
    sf = (uq.get("inference") or {}).get("scene_file") or uq["data"]["matchup_file"]
    sf = _abs(Path(pipe["_uq_path"]).parent, sf) if not os.path.isabs(sf) else sf
    recs = _read(sf).get("matchups") or []
    for r in recs:
        h = r.get("head_outputs")
        if h and Path(h).exists():
            with rasterio.open(h) as s:
                t = s.tags()
            return {"sit_fuse_yaml": pipe.get("sit_fuse_yaml") or t["sit_fuse_yaml"],
                    "head_export": {"top_k_parent": int(t["top_k_parent"]), "top_k_child": int(t["top_k_child"]),
                                    "n_parent_ranks": int(t["n_parent_ranks"]), "batch_size": 10000},
                    "georef": {"use_input_files": "auto", "latlon_order": "auto", "n_gcps": 20}}
    raise FileNotFoundError(f"{pipe['_name']}: set export_config (the export YAML used for training); no readable "
                            f"head outputs in {sf}")


def production_uq_config(uq, uq_path, out_dir, scene_file):
    """Evaluated UQ config -> production config (new outputs, trained model and calibration reused)."""
    base = Path(uq_path).parent
    old_out = Path(_abs(base, uq["output_dir"]))
    c = copy.deepcopy(uq)
    inf = c.setdefault("inference", {})
    ev = c.setdefault("evaluation", {})
    gx = c.setdefault("geo_export", {}) or {}
    c["geo_export"] = gx
    tsum = inf.get("training_summary_path") or old_out / "training" / "training_summary.json"
    old_eval = gx.get("evaluation_dir") or ev.get("output_dir") or old_out / "evaluation"
    for k in ("matchup_file",):
        if c.get("data", {}).get(k):
            c["data"][k] = _abs(base, c["data"][k])
    c["output_dir"] = str(out_dir)
    inf.pop("output_dir", None)                    # one directory per model under <output_dir>/inference
    inf.update({"training_summary_path": _abs(base, tsum), "scene_file": str(scene_file), "splits": None,
                "reference_mask_areas": False})
    tmpl = str(Path(out_dir) / "inference" / "{model}")
    ev.update({"inference_dir_template": tmpl, "output_dir": str(Path(out_dir) / "evaluation")})
    gx.update({"inference_dir_template": tmpl, "output_dir": str(Path(out_dir) / "geo"),
               "scene_file": str(scene_file), "evaluation_dir": _abs(base, old_eval)})
    return c, Path(_abs(base, tsum)), Path(_abs(base, old_eval))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml_config", type=Path)
    ap.add_argument("--run", action="store_true", help="run the plan with sit_fuse.local_launcher")
    ap.add_argument("--reexport", action="store_true", help="export head outputs even when they exist")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--animation", dest="animation", action="store_true", default=None,
                   help="add the animation jobs (overrides animation.enabled)")
    g.add_argument("--no-animation", dest="animation", action="store_false",
                   help="write the animation config but add no animation jobs")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:len(argv) - len(extra) - (1 if "--" in argv else 0)])
    cfg = _read(a.yaml_config)
    base = a.yaml_config.resolve().parent
    root = Path(_abs(base, cfg["output_root"]))
    t0, t1 = utc(cfg["start"]), utc(cfg["end"])
    bbox = [float(v) for v in cfg["bbox"]] if cfg.get("bbox") else None
    cdir = root / "configs"
    sensors = cfg["sensors"]
    pipes = cfg["pipelines"]
    ex_cfg = cfg.get("export") or {}
    n_ex_shards = int(ex_cfg.get("shards", 1))

    inputs = {}
    for s, sc in sensors.items():
        inputs[s] = list_inputs(sc, t0, t1, bbox, base)
        print(f"{s}: {len(inputs[s])} inputs in [{t0:%Y-%m-%d %H:%M}, {t1:%Y-%m-%d %H:%M}) UTC"
              + (f" in bbox {bbox}" if bbox else "")
              + (f", first per {sc['every_minutes']} min" if sc.get("every_minutes") else ""))

    exports = {}                    # key -> {name, settings, sensor, pipes}
    planner_pipes, summary, scene_hash = {}, [], {}
    for name, p in pipes.items():
        p = dict(p, _name=name, _uq_path=_abs(base, p["uq_config"]))
        uq = _read(p["_uq_path"])
        s = p["sensor"]
        hs = head_settings(p, uq, base)
        key = json.dumps([s, hs], sort_keys=True)
        e = exports.setdefault(key, {"name": f"{s}_{len([x for x in exports.values() if x['sensor'] == s])}",
                                     "settings": hs, "sensor": s, "pipes": []})
        e["pipes"].append(name)
        hdir = root / "heads" / e["name"]
        product = str(p.get("product") or uq["data"].get("product") or "").lower()
        recs = []
        for f, t in inputs[s]:
            stem = Path(f).name[:-4] if f.endswith(".tif") else Path(f).stem
            recs.append({"scene_id": f"{product}_{stem}", "pair_id": f"{product}_{stem}", "product": product,
                         "sit_fuse_input": f, "sit_fuse_context_free": f,
                         "head_outputs": str(hdir / f"{Path(f).name}.head_outputs.tif"),
                         "sit_fuse_timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "date": t.strftime("%Y-%m-%d"),
                         "split_date": t.strftime("%Y-%m-%d"), "split_group": t.strftime("%Y-%m-%d"),
                         "classifier_split": "production", "encoder_split": "production"})
            ss = p.get("split_stratum", cfg.get("split_stratum"))
            if ss:                                  # event / stratum key for per-group thresholds
                recs[-1]["split_stratum"] = str(ss)
        pdir = root / name
        sf = _write(pdir / "scenes.yaml", {"matchups": recs})
        scene_hash[name] = hashlib.sha1("\n".join(r["scene_id"] for r in recs).encode()).hexdigest()[:10]
        pc, tsum, old_eval = production_uq_config(uq, p["_uq_path"], pdir, sf)
        if p.get("models"):
            pc["evaluation"]["models"] = list(p["models"])
        if not tsum.exists():
            print(f"WARNING {name}: training summary {tsum} not found")
        uqp = _write(pdir / "uq_config.yaml", pc)
        planner_pipes[name] = {"uq_config": str(uqp), **({"models": p["models"]} if p.get("models") else {}),
                               **({"inference_shards": p["inference_shards"]} if p.get("inference_shards") else {})}
        summary.append(f"  {name}: {len(recs)} scenes, product {product}, heads/{e['name']}, models "
                       f"{pc['evaluation'].get('models')}, area calibration {old_eval}")

    # head-export configs (only inputs without head outputs)
    ex_jobs, tok_ready = [], []
    lcfg = dict(cfg.get("launcher") or {})
    state = Path(_abs(base, lcfg.get("state_dir", root / "launch")))
    lcfg["state_dir"] = str(state)
    (state / "tokens").mkdir(parents=True, exist_ok=True)
    for e in exports.values():
        hdir = root / "heads" / e["name"]
        todo = [f for f, _ in inputs[e["sensor"]]
                if a.reexport or not (hdir / f"{Path(f).name}.head_outputs.tif").exists()]
        toks = [str(state / "tokens" / f"heads_{p}") for p in e["pipes"]]
        if not todo:
            tok_ready += toks
            print(f"  heads/{e['name']}: all {len(inputs[e['sensor']])} head outputs exist")
            continue
        h = hashlib.sha1("\n".join(todo).encode()).hexdigest()[:8]
        shards = [todo[i::n_ex_shards] for i in range(min(n_ex_shards, len(todo)))]
        paths = []
        for i, fs in enumerate(shards):
            d = {"sit_fuse_yaml": e["settings"]["sit_fuse_yaml"], "output_dir": str(hdir), "files": fs,
                 "matchup_file": None, "head_export": e["settings"]["head_export"],
                 "georef": dict(e["settings"]["georef"], reference_geotiffs=None, subset_inds=None)}
            paths.append(str(_write(cdir / f"export_{e['name']}_{h}_{i}.yaml", d)))
        ex_jobs.append({"name": f"heads_{e['name']}", "cmd": "cd {td} && {py} export_head_outputs.py {ecfg}",
                        "cases": [{"ecfg": pth, "shard": i} for i, pth in enumerate(paths)], "id_keys": ["shard"],
                        "cpus": int(ex_cfg.get("cpus", 4)), "gpu": ex_cfg.get("gpu", 1), "provides": toks})
        print(f"  heads/{e['name']}: {len(todo)} inputs to export in {len(paths)} job(s) (pipelines {e['pipes']})")
    for t in tok_ready:
        Path(t).touch()

    # planner (inference + geo) reused from plan_uq_jobs
    from plan_uq_jobs import build_plan
    pcfg = {"python": cfg.get("python", sys.executable), "launcher": lcfg, "stages": ["infer", "geo"],
            "resume": cfg.get("resume", True), "resources": cfg.get("resources") or {},
            "shards": cfg.get("shards") or {}, "pipelines": planner_pipes}
    _write(cdir / "uq_parallel.production.yaml", pcfg)
    plan, report = build_plan(pcfg, cdir)
    py, td = shlex.quote(pcfg["python"]), shlex.quote(str(TRAIN_DIR))
    names = {j["name"] for j in plan["jobs"]}
    for j in plan["jobs"]:
        if j["name"] in ("infer", "infer_merge", "geo"):    # new scene set -> new command -> the launcher reruns it
            j["cmd"] += " # scenes {sh}"
            for c in j["cases"]:
                c["sh"] = scene_hash[c["pipe"]]
        if j["name"] == "geo":                     # no evaluation stage: geo waits for the merged inference
            j["after"] = [x for x in ("infer_merge", "infer") if x in names][:1]
            if not j["after"]:
                del j["after"]
        if j["name"] == "infer":
            j["needs"] = list(j.get("needs") or []) + [str(state / "tokens" / "heads_{pipe}")]
    for j in ex_jobs:
        j["cmd"] = j["cmd"].replace("{td}", td).replace("{py}", py)

    # fusion: apply the fusion fitted on the evaluation periods (frozen weights, recalibration, thresholds)
    fu_jobs, fused_prods = [], {}
    for fname, fs in (cfg.get("fusion") or {}).items():
        ecfg_path = _abs(base, fs["config"])
        fcfg = copy.deepcopy(_read(ecfg_path))
        fmap = fs.get("pipelines") or {}
        if isinstance(fmap, list):              # [tempo_smoke, goes_smoke] -> by sensor
            fmap = {pipes[x]["sensor"]: x for x in fmap}
        for src, sc in (fcfg.get("sources") or {}).items():
            pipe = fmap.get(src)
            if pipe not in pipes:
                raise ValueError(f"fusion {fname}: source {src!r} needs a production pipeline (fusion.{fname}.pipelines)")
            pm = pipes[pipe].get("models")
            if pm and sc.get("model", "hist_gradient_boosting") not in pm:
                raise ValueError(f"fusion {fname}: source {src} model {sc.get('model')} not in pipeline {pipe} models {pm}")
            sc["geo_dir"] = str(root / pipe / "geo")
            sc["manifest"] = str(root / pipe / "geo" / "geo_products_manifest.json")
        fit_dir = _abs(Path(ecfg_path).parent, fs.get("models_from") or fcfg["output_dir"])
        if not (Path(fit_dir) / "fusion_models.pkl").exists() and not str(fit_dir).endswith(".pkl"):
            print(f"WARNING fusion {fname}: {fit_dir}/fusion_models.pkl not found (fit the evaluation fusion first)")
        fout = root / f"fusion_{fname}"
        fcfg.update({"output_dir": str(fout), "references": {},
                     "output": dict(fcfg.get("output") or {}, splits=["production"])})
        fcfg["fusion"] = dict(fcfg.get("fusion") or {}, models_from=str(fit_dir))
        if fs.get("grids"):
            fcfg["grids"] = {g: v for g, v in fcfg["grids"].items() if g in fs["grids"]}
            fcfg["fusion"]["grids"] = [g for g in fcfg["fusion"].get("grids", fs["grids"]) if g in fs["grids"]]
        fpath = _write(cdir / f"fusion_{fname}.yaml", fcfg)
        nfs, fcpus = int(fs.get("shards", 1)), int(fs.get("cpus", 2))
        used = sorted(set(fmap[x] for x in fcfg["sources"]))
        sh = hashlib.sha1(" ".join(scene_hash[x] for x in used).encode()).hexdigest()[:10]
        sh += " cfg " + hashlib.sha1(Path(fpath).read_bytes()).hexdigest()[:8]   # config edits -> rerun
        fu = f"cd {td} && {py} fuse_multiscale_products.py -c {shlex.quote(str(fpath))}"
        fu_jobs.append({"name": f"fusion_apply_{fname}", "cmd": fu + f" --stage apply --shard {{shard}}/{nfs} -n {fcpus}"
                        f" # scenes {sh}", "cases": [{"shard": i} for i in range(nfs)], "id_keys": ["shard"],
                        "cpus": fcpus, "needs": [str(state / "tokens" / f"geo_{x}") for x in used]})
        ftok = str(state / "tokens" / f"fusion_{fname}")
        fu_jobs.append({"name": f"fusion_merge_{fname}", "cmd": fu + f" --stage merge --n-shards {nfs} -n 1 # scenes {sh}",
                        "cases": [{}], "id_keys": [], "cpus": 1, "after": [f"fusion_apply_{fname}"], "provides": [ftok]})
        grid = (cfg.get("animation") or {}).get("fusion_grid", fs.get("animation_grid", "fine"))
        fprod = str(fcfg.get("product", "smoke")).lower()
        fused_prods[f"fused_{fprod}" if len(cfg["fusion"]) == 1 else f"fused_{fname}"] = {
            "geo_dir": str(fout / grid), "model": "fused", "product": fprod, "style": fprod, "sensor": "fused",
            "_tok": ftok}
        summary.append(f"  fusion {fname}: sources {fmap} -> {fout} (weights from {fit_dir}, {nfs} shard(s))")

    # overlay animations
    an_jobs = []
    an = cfg.get("animation")
    an_on = bool(an) and (a.animation if a.animation is not None else bool(an.get("enabled", True)))
    if an:
        model = an.get("model", "hist_gradient_boosting")
        prods = {}
        for name, p in pipes.items():
            pc = _read(root / name / "uq_config.yaml")
            prod = str(p.get("product") or pc["data"].get("product", "smoke")).lower()
            prods[f"{p['sensor']}_{prod}"] = {
                "geo_dir": str(root / name / "geo"), "manifest": str(root / name / "geo" / "geo_products_manifest.json"),
                "model": (an.get("models") or {}).get(name, model), "product": pc["data"].get("product", "smoke"),
                "style": prod,
                "sensor": p["sensor"], "_tok": str(state / "tokens" / f"geo_{name}")}
        prods.update(fused_prods)
        obs = copy.deepcopy(an.get("observations") or {})
        if "goes" in sensors and "goes" in obs and not obs["goes"].get("dir"):
            obs["goes"]["dir"] = sensors["goes"]["inputs"]
            obs["goes"].setdefault("pattern", sensors["goes"].get("pattern", "OR_ABI-*.tif"))
        for o in obs.values():
            if o.get("dir"):
                o["dir"] = _abs(base, o["dir"]) if not isinstance(o["dir"], list) else [_abs(base, x) for x in o["dir"]]
        acfg = {k: v for k, v in an.items() if k not in ("observations", "models", "model", "products", "shards",
                                                         "include", "cpus", "fusion_grid")}
        acfg.update({"output_dir": _abs(base, an.get("output_dir", root / "animations")),
                     "time_range": an.get("time_range") or [str(cfg["start"]), str(cfg["end"])],
                     "bbox": an.get("bbox") or cfg.get("bbox"), "observations": obs,
                     "products": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                  for k, v in prods.items() if not an.get("include") or k in an["include"]}})
        if not acfg["bbox"]:
            raise ValueError("animation needs a bbox (animation.bbox or bbox)")
        acfg.pop("enabled", None)
        fb = (acfg.get("frames") or {}).get("base", "tempo")
        ev = (sensors.get("goes") or {}).get("every_minutes")
        if ev and fb == "tempo" and float((acfg.get("frames") or {}).get("goes_window_minutes", 10)) < float(ev) / 2:
            print(f"WARNING animation: GOES inputs are subsampled to {ev} min but frames.base is tempo with a "
                  f"goes_window_minutes below {float(ev) / 2:g}; GOES / fused overlays will be missing on most "
                  f"frames (use frames: {{base: goes, goes_every_minutes: {ev}}})")
        apath = _write(cdir / "overlay_animation.yaml", acfg)
        n_an = int(an.get("shards", 1))
        needs = [v["_tok"] for k, v in prods.items() if not an.get("include") or k in an["include"]]
        acpus = int(an.get("cpus", 4))
        if not an_on:
            print(f"  animation: disabled; config written to {apath} (enable with animation.enabled: true or "
                  f"--animation and rebuild; finished jobs are skipped)")
    if an and an_on:
        an_jobs.append({"name": "animate", "cmd": f"cd {td} && {py} make_overlay_animations.py -c {shlex.quote(str(apath))}"
                        f" --shard {{shard}}/{n_an} -n {acpus} --workers {acpus}"
                        f" # scenes {hashlib.sha1(' '.join(sorted(scene_hash.values())).encode()).hexdigest()[:10]}"
                        f" cfg {hashlib.sha1(Path(apath).read_bytes()).hexdigest()[:8]}",
                        "cases": [{"shard": i} for i in range(n_an)], "id_keys": ["shard"],
                        "cpus": acpus, "needs": needs})
    plan["jobs"] = ex_jobs + plan["jobs"] + fu_jobs + an_jobs
    out = Path(_abs(base, cfg.get("plan_file", state / "production_plan.yaml")))
    _write(out, plan)
    print("pipelines:\n" + "\n".join(summary))
    n = sum(len(j["cases"]) for j in plan["jobs"])
    run = [sys.executable, "-m", "sit_fuse.local_launcher", "run", str(out)] + extra
    print(f"wrote {out}: {len(plan['jobs'])} job specs, {n} jobs\n  " + " ".join(map(shlex.quote, run)))
    if a.run:
        env = dict(os.environ, PYTHONPATH=plan["launcher"]["env"]["PYTHONPATH"])
        sys.exit(subprocess.call(run, env=env))


if __name__ == "__main__":
    main()
