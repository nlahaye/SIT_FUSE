"""
Copyright [2022-26], by the California Institute of Technology and Chapman University.
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.

Write (and optionally run) a sit_fuse.local_launcher plan for the UQ pipeline from feature-table generation on, so
the independent pieces run side by side on the local CPUs with the right order between them:

    python plan_uq_jobs.py uq_parallel.yaml            # writes the plan (plan_file), prints the launcher command
    python plan_uq_jobs.py uq_parallel.yaml --run [-- --dry-run | --only 'infer*' | --rerun-failed ...]
    python -m sit_fuse.local_launcher status <plan_file>

Per pipeline (one UQ config, e.g. tempo_smoke / goes_fire):

    features  (shard i/N, scenes)  -> features_merge -> train (--workers)
    infer     (model x shard)      -> infer_merge (model) -> evaluate (model) -> compare (--compare-only)
    geo       (model, --update-manifest; after its evaluate, which supplies the area calibration)
    visuals   (after all geo jobs + compare of the pipeline)
Per fusion (smoke / fire): fusion_fit (after the geo jobs of its pipelines) -> fusion_apply (shard) -> fusion_merge
  [-> fusion_visuals with fusion.<name>.visuals: <make_fusion_visuals.py config>]
joint_visuals: after every visuals and fusion_merge job.
extra_jobs: raw sit_fuse.local_launcher job specs added to the same plan (e.g. matchup preparation before a pipeline,
  production runs or animations after it), with three extra keys resolved here:
    before: [pipe, ...]     the pipeline's first planned stage (features / train / infer ...) waits for this job
    after_geo: [pipe, ...]  waits for the pipeline's geo jobs (no wait when that geo export is already finished)
    after_fusion: [name]    waits for that fusion's merge job (no wait when it is not in the plan)
  `after` (other extra jobs or planner job names), cases / grid, cpus, gpu, creates, provides, needs pass through.

Resume (resume: true, the default): stages whose outputs already exist are left out of the plan, so a run started
without the launcher (or with a deleted state_dir) continues where it stopped. Finished = upstream finished and outputs
at least as new as upstream outputs: features (build_manifest.json + pixel_features.parquet), train
(training_summary.json), infer per model (inference_manifest.json, written after all scenes), evaluate per model
(eval_summary.json with bootstrap_groups), compare (model_comparison.csv), geo per model (entry in
geo_products_manifest.json whose listed files all still exist). A model whose inference was interrupted is
re-planned; its finished scenes are reused from the per-scene checkpoints (scene_results/) when those exist. Fusion and visuals rely on the launcher markers.
Resume also re-plans features / inference / geo made at another scene cadence than the config asks for
(data.every_minutes, inference.every_minutes, geo_every_minutes), and training whose model files are missing.
A per-pipeline `rerun_from: features|train|infer|evaluate|geo` re-plans that stage and everything after it.
An explicit inference.every_minutes / geo_export.every_minutes in a UQ config is appended to the infer, evaluate and
geo commands as a shell comment (`# every 30 min`), so changing the cadence also reruns jobs the launcher marked done
at the old cadence in the same state_dir (inference reuses the scene checkpoints of the kept scenes).
--no-resume plans everything (the scripts still reuse matching scene checkpoints; add --overwrite to the inference
commands to recompute them).

Every script keeps its single-process default; shards / workers reproduce the serial outputs (see
CHANGES_UQ_PARALLEL.md). Finished jobs are skipped through the launcher markers, so the plan can be regenerated and
rerun at any time; `stages:` restricts the plan (dependencies on omitted stages are assumed satisfied).
See uq_parallel.example.yaml for the YAML keys.
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

STAGES = ("features", "train", "infer", "evaluate", "geo", "visuals", "fusion", "joint_visuals")
DEFAULT_CPUS = {"features": 1, "features_merge": 1, "train": 2, "infer": 1, "infer_merge": 1, "evaluate": 1,
                "compare": 1, "geo": 1, "visuals": 1, "fusion_fit": 2, "fusion_apply": 1, "fusion_merge": 1,
                "joint_visuals": 1}


def _read(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _models(pipe_spec, uq_cfg):
    """Explicit pipeline `models`, else the UQ config's evaluation.models, else inference.model."""
    m = pipe_spec.get("models") or (uq_cfg.get("evaluation") or {}).get("models") \
        or [(uq_cfg.get("inference") or {}).get("model")]
    m = [x for x in m if x]
    if not m:
        raise ValueError(f"no models for {pipe_spec.get('uq_config')}: set evaluation.models or pipeline models")
    return m


def _mtime(p):
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return None


def _newer(p, *deps):
    """p exists and is at least as new as every existing dep."""
    t = _mtime(p)
    return t is not None and all(_mtime(d) is None or t >= _mtime(d) for d in deps)


def _cad(v):
    """Cadence value as compared by resume: None / 0 / '' = every scene."""
    try:
        return float(v) if v not in (None, "", False) and float(v) > 0 else None
    except (TypeError, ValueError):
        return None


def _jload(p):
    import json
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def _training_files(summary):
    """Artifacts a training_summary.json points at (models may live in a --recalibrate source dir)."""
    fs = list((summary.get("model_paths") or {}).values())
    fs += [summary.get(k) for k in ("context_encoder_path", "calibrators_path") if summary.get(k)]
    for v in (summary.get("bootstrap_manifest") or {}).values():
        fs += list(v or [])
    return [str(f) for f in fs if f]


RERUN_ORDER = ("features", "train", "infer", "evaluate", "geo")


def finished(uq, models, geo_every=None, rerun_from=None):
    """
    Stages of one UQ config whose outputs already exist (resume after a run without the launcher, or after the
    launcher state was deleted). A stage counts as finished only when its upstream stage is finished and its
    outputs are at least as new as the upstream outputs, so a rerun upstream invalidates everything below it.
    Also re-planned:
      - features / infer / geo made at another scene cadence than the config now asks for (data.every_minutes,
        inference.every_minutes, geo every_minutes; a full-cadence geo export covers an hourly request);
      - train when a model, encoder, calibrator or bootstrap file named in training_summary.json is missing;
      - rerun_from: <stage> and everything after it.
    Returns {"features": bool, "train": bool, "infer": {m: bool}, "evaluate": {m: bool}, "compare": bool,
    "geo": {m: bool}, "notes": [str]}.
    """
    out = Path(uq["output_dir"])
    data = uq.get("data") or {}
    inf, ev, gx = uq.get("inference") or {}, uq.get("evaluation") or {}, uq.get("geo_export") or {}
    notes = []
    if rerun_from and rerun_from not in RERUN_ORDER:
        raise ValueError(f"rerun_from {rerun_from!r}: choose from {RERUN_ORDER}")
    forced = set(RERUN_ORDER[RERUN_ORDER.index(rerun_from):]) if rerun_from else set()

    bm = out / "build_manifest.json"
    bman = _jload(bm) if bm.exists() else None
    feat = Path((bman or {}).get("pixel_features") or out / "pixel_features.parquet")
    d = {"features": bman is not None and feat.exists()}
    if d["features"] and "every_minutes" in bman and _cad(bman["every_minutes"]) != _cad(data.get("every_minutes")):
        notes.append(f"features built with every_minutes={bman['every_minutes']}, config wants "
                     f"{data.get('every_minutes') or 'every scene'}")
        d["features"] = False
    if bman is not None:
        notes.append(f"feature table: {bman.get('n_scenes', '?')} scenes, {bman.get('n_rows', '?')} rows"
                     + (f", {len(bman['skipped_scenes'])} skipped" if bman.get("skipped_scenes") else ""))
    d["features"] = d["features"] and "features" not in forced

    tsum = Path(inf.get("training_summary_path", out / "training" / "training_summary.json"))
    d["train"] = d["features"] and _newer(tsum, bm, feat) and "train" not in forced
    if tsum.exists():
        summ = _jload(tsum) or {}
        miss = [f for f in _training_files(summ) if not Path(f).exists()]
        src = summ.get("recalibrated_from")
        notes.append(f"training: {tsum}" + (f" (recalibrated; models in {src})" if src else ""))
        if miss:
            notes.append(f"training files missing ({len(miss)}), e.g. {miss[0]}")
            d["train"] = False
    else:
        notes.append(f"training: none yet ({tsum})")

    edir = Path(ev.get("output_dir", out / "evaluation"))
    gman = Path(gx.get("output_dir", out / "geo")) / "geo_products_manifest.json"
    gent = (_jload(gman) or {}) if gman.exists() else {}
    want_inf = _cad(inf.get("every_minutes", data.get("every_minutes")))
    want_geo = _cad(geo_every if geo_every is not None else gx.get("every_minutes"))

    def _geo_ok(e):                       # a deleted raster / vector file re-plans the model's geo job
        if not isinstance(e, dict) or not e.get("rasters"):
            return False
        fs = list(e["rasters"]) + [f for k in ("polygons", "scene_footprints", "pixels") for f in e.get(k) or []]
        have = _cad(e.get("every_minutes"))
        return all(Path(f).exists() for f in fs) and (have is None or have == want_geo)
    geo_models = {m for m, e in gent.items() if _geo_ok(e)}
    d["infer"], d["evaluate"], d["geo"], evs = {}, {}, {}, []
    for m in models:
        man = Path(inf["output_dir"]) / "inference_manifest.json" if inf.get("output_dir") else \
            out / "inference" / m / "inference_manifest.json"
        ok = d["train"] and _newer(man, tsum) and "infer" not in forced   # written only after every scene
        if ok:
            im = _jload(man) or {}
            have = _cad(im.get("every_minutes"))
            if have != want_inf:
                notes.append(f"{m}: inference at every_minutes={im.get('every_minutes')}, config wants "
                             f"{want_inf or 'every scene'}")
                ok = False
        d["infer"][m] = ok
        es = edir / m / "eval_summary.json"
        ok = ok and _newer(es, man) and "evaluate" not in forced
        if ok:                                                    # older evaluations lack the pairing groups
            ok = "bootstrap_groups" in (_jload(es) or {})
        d["evaluate"][m] = ok
        evs.append(es)
        d["geo"][m] = ok and m in geo_models and _newer(gman, es) and "geo" not in forced
    d["compare"] = all(d["evaluate"].values()) and _newer(edir / "model_comparison.csv", *evs) \
        and "evaluate" not in forced
    d["notes"] = notes
    return d


def _check_fusion_sources(fname, fcfg_path, after_uq):
    """Warn when a fusion source (tempo / goes) reads a geo dir other than its pipeline's geo_export.output_dir."""
    msgs = []
    try:
        fcfg = _read(fcfg_path)
    except OSError as e:
        return [f"  fusion {fname}: cannot read {fcfg_path}: {e}"]
    for key, src in (fcfg.get("sources") or {}).items():
        pipe = next((p for p in after_uq if p.startswith(str(key))), None)
        if not pipe or not src.get("geo_dir"):
            continue
        uq = _read(after_uq[pipe])
        want = Path((uq.get("geo_export") or {}).get("output_dir", Path(uq["output_dir"]) / "geo"))
        if Path(src["geo_dir"]).resolve() != want.resolve():
            msgs.append(f"  WARNING fusion {fname}: sources.{key}.geo_dir = {src['geo_dir']} but pipeline {pipe} "
                        f"writes its geo products to {want}")
    return msgs


def extra_jobs(specs, J, pipes, tok, ab, train_dir, py, stages):
    """
    extra_jobs entries -> launcher job specs. `before: [pipe]` adds an `after: {name, on: [pipe]}` join to every
    planned stage job of those pipelines (cases of the extra job get pipe=<pipe>, so other pipelines do not wait);
    after_geo / after_fusion become `needs` on the planner's geo / fusion tokens, kept only for planned jobs.
    {train} and {py} in cmd expand to train_dir and python.
    """
    planned = {}
    for j in J:
        for c in j["cases"]:
            planned.setdefault(j["name"], set()).add(c.get("pipe") or c.get("fusion"))
    out, names = [], {j["name"] for j in J}
    for sp in specs:
        sp = dict(sp)
        name = sp["name"]
        if name in names:
            raise ValueError(f"extra_jobs: {name!r} clashes with a planner job name")
        names.add(name)
        if sp.pop("enabled", True) is False:
            continue
        before = list(sp.pop("before", None) or [])
        bad = [p for p in before + list(sp.get("after_geo") or []) if p not in pipes]
        if bad:
            raise ValueError(f"extra_jobs {name}: {bad} are not pipelines")
        if before:
            if len(before) > 1 and (sp.get("cases") or sp.get("grid")):
                raise ValueError(f"extra_jobs {name}: before several pipelines needs no cases / grid")
            base = sp.get("cases") or [{}]
            sp["cases"] = [dict(c, pipe=p) for p in before for c in base]
            sp.setdefault("id_keys", sorted({k for c in sp["cases"] for k in c if k != "pipe"}) or ["pipe"])
            for j in J:                               # every stage job of those pipelines (transitive anyway)
                if j["name"] in ("features", "features_merge", "train", "infer", "infer_merge", "evaluate",
                                 "compare", "geo") and any(c.get("pipe") in before for c in j["cases"]):
                    j["after"] = list(j.get("after") or []) + [{"name": name, "on": ["pipe"]}]
        needs = list(sp.get("needs") or [])
        needs += [tok("geo", p) for p in sp.pop("after_geo", None) or [] if p in planned.get("geo", ())]
        needs += [tok("fusion", f) for f in sp.pop("after_fusion", None) or []
                  if f"fusion_merge_{f}" in planned]
        if needs:
            sp["needs"] = needs
        sp["cmd"] = str(sp["cmd"]).replace("{train}", shlex.quote(train_dir)).replace("{py}", shlex.quote(py))
        sp.setdefault("cpus", 1)
        out.append(sp)
    return out


def build_plan(cfg, base_dir, resume=None):
    def ab(p):
        return str(p if os.path.isabs(str(p)) else (base_dir / p).resolve())

    stages = set(cfg.get("stages") or STAGES)
    bad = stages - set(STAGES)
    if bad:
        raise ValueError(f"unknown stages {sorted(bad)}; choose from {STAGES}")
    res = dict(DEFAULT_CPUS, **(cfg.get("resources") or {}))
    train_workers = int(res.pop("train_workers", res["train"]))
    shards = dict({"features": 1, "inference": 1, "fusion": 1}, **(cfg.get("shards") or {}))
    launcher = dict(cfg.get("launcher") or {})
    state = ab(launcher.setdefault("state_dir", "launch_state_uq"))
    launcher["state_dir"] = state
    train_dir = ab(cfg.get("train_dir") or Path(__file__).resolve().parent)
    py = cfg.get("python", sys.executable)
    q = shlex.quote

    def tok(kind, name):                 # virtual paths joining stages across pipelines (provides / needs)
        return f"{state}/tokens/{kind}_{name}"

    def cmd(script, *args):
        return f"cd {q(train_dir)} && {q(py)} {script} " + " ".join(args)

    pipes, fusions = cfg.get("pipelines") or {}, cfg.get("fusion") or {}
    resume = cfg.get("resume", True) if resume is None else resume
    pc, inf_cases, model_cases, vis_cases = [], [], [], []
    done = {}                                    # (stage, pipe[, model]) finished per existing outputs
    report = []
    for name, sp in pipes.items():
        uq = ab(sp["uq_config"])
        ge = sp.get("geo_every_minutes")         # e.g. 60: hourly geo products from 5-min GOES inference
        c = {"pipe": name, "uq": uq, "geo_extra": (f"--every-minutes {ge:g}" + (
            f" --offset-minutes {float(sp['geo_offset_minutes']):g}" if sp.get("geo_offset_minutes") else ""))
            if ge else ""}
        uq_cfg = _read(uq)
        # explicit scene cadence of the UQ config -> a shell comment on the infer / evaluate / geo commands, so a
        # cadence change gives new commands and the launcher reruns jobs it marked done at the old cadence (the
        # inference reuses its scene checkpoints). Unset cadence: commands unchanged.
        ie = _cad((uq_cfg.get("inference") or {}).get("every_minutes"))
        gev = _cad(ge if ge else (uq_cfg.get("geo_export") or {}).get("every_minutes"))
        c["inf_cad"] = f" # every {ie:g} min" if ie else ""
        c["geo_cad"] = " # " + ", ".join(x for x in (f"geo every {gev:g} min" if gev and not ge else "",
                                                      f"inference every {ie:g} min" if ie else "") if x) \
            if (gev and not ge) or ie else ""
        pc.append(c)
        models = _models(sp, uq_cfg)
        if resume:
            f = finished(uq_cfg, models, geo_every=ge, rerun_from=sp.get("rerun_from"))
            done.update({("features", name): f["features"], ("train", name): f["train"],
                         ("compare", name): f["compare"]})
            for st in ("infer", "evaluate", "geo"):
                done.update({(st, name, m): v for m, v in f[st].items()})
            report.append(f"  {name} ({uq_cfg['output_dir']}): features={'done' if f['features'] else 'run'} "
                          f"train={'done' if f['train'] else 'run'} " + " ".join(
                              f"{m}[infer={'done' if f['infer'][m] else 'run'},"
                              f"eval={'done' if f['evaluate'][m] else 'run'},"
                              f"geo={'done' if f['geo'][m] else 'run'}]" for m in models))
            report += [f"      {n}" for n in f["notes"]]
        for m in models:
            model_cases.append(dict(c, model=m))
            inf_cases += [dict(c, model=m, shard=i) for i in range(int(sp.get("inference_shards",
                                                                             shards["inference"])))]
        if sp.get("visuals_config"):
            vis_cases.append(dict(c, vis=ab(sp["visuals_config"]), products=" ".join(
                f"--product {q(p)}" for p in (sp.get("visual_products") or []))))
    fshards = {name: int(sp.get("features_shards", shards["features"])) for name, sp in pipes.items()}

    J = []

    skip_key = {"features": "features", "features_merge": "features", "train": "train", "infer": "infer",
                "infer_merge": "infer", "evaluate": "evaluate", "compare": "compare", "geo": "geo"}

    def job(name, stage, cases, command, after=(), cpus=None, id_keys=("pipe",), **kw):
        k = skip_key.get(name)
        if k:                                    # drop work whose outputs already exist (resume)
            cases = [c for c in cases if not done.get((k, c["pipe"]) + ((c["model"],) if "model" in c else ()))]
        if stage not in stages or not cases:
            return
        cpus = int(cpus or res[name])
        J.append(dict(name=name, cmd=command.replace("{cpus}", str(cpus)), cases=cases, cpus=cpus,
                      id_keys=list(id_keys), after=list(after), **kw))

    nf = "{nf}"
    feat_cases = [dict(c, nf=fshards[c["pipe"]], shard=i) for c in pc for i in range(fshards[c["pipe"]])]
    job("features", "features", feat_cases,
        cmd("build_uq_feature_table.py", "{uq}", "--shard {shard}/" + nf, "-n {cpus}"), id_keys=("pipe", "shard"))
    job("features_merge", "features", [dict(c, nf=fshards[c["pipe"]]) for c in pc],
        cmd("build_uq_feature_table.py", "{uq}", "--merge --n-shards " + nf, "-n {cpus}"), after=["features"])
    job("train", "train", pc,
        cmd("simple_classifiers_uq.py", "{uq}", f"-n {{cpus}} --workers {train_workers}"), after=["features_merge"])
    ni = {}
    for c in inf_cases:
        ni[(c["pipe"], c["model"])] = ni.get((c["pipe"], c["model"]), 0) + 1
    job("infer", "infer", [dict(c, ni=ni[(c["pipe"], c["model"])]) for c in inf_cases],
        cmd("classifier_inference_uq.py", "{uq}", "--model {model} --shard {shard}/{ni}", "-n {cpus}{inf_cad}"),
        after=["train"], id_keys=("pipe", "model", "shard"))
    job("infer_merge", "infer", [dict(c, ni=ni[(c["pipe"], c["model"])]) for c in model_cases],
        cmd("classifier_inference_uq.py", "{uq}", "--model {model} --merge --n-shards {ni}", "-n {cpus}{inf_cad}"),
        after=["infer"], id_keys=("pipe", "model"))
    job("evaluate", "evaluate", model_cases, cmd("evaluate_uq.py", "{uq}", "--models {model}", "-n {cpus}{inf_cad}"),
        after=["infer_merge"], id_keys=("pipe", "model"))
    job("compare", "evaluate", pc, cmd("evaluate_uq.py", "{uq}", "--compare-only", "-n {cpus}{inf_cad}"),
        after=["evaluate"])
    job("geo", "geo", model_cases,
        cmd("export_geo_products.py", "{uq}", "--models {model} --update-manifest {geo_extra}", "-n {cpus}{geo_cad}"),
        after=["evaluate"], id_keys=("pipe", "model"), provides=[tok("geo", "{pipe}")])
    job("visuals", "visuals", vis_cases, cmd("make_uq_visuals.py", "-c {vis}", "{products}"),
        after=["geo", "compare"], provides=[tok("visuals", "all")])

    fc = []
    for name, sp in fusions.items():
        after = sp.get("after") or []
        unknown = [p for p in after if p not in pipes]
        if unknown:
            raise ValueError(f"fusion {name}: after {unknown} are not pipelines")
        geo_planned = {c["pipe"] for j in J if j["name"] == "geo" for c in j["cases"]}
        report += _check_fusion_sources(name, ab(sp["config"]), {p: ab(pipes[p]["uq_config"]) for p in after})
        fc.append({"fusion": name, "fcfg": ab(sp["config"]), "nfs": int(sp.get("shards", shards["fusion"])),
                   "_needs": [tok("geo", p) for p in after if p in geo_planned]})   # finished geo: no wait
    # needs are per job spec, so one fit spec per fusion (each waits only on its own pipelines' geo jobs)
    sp_vis = {n: sp.get("visuals") for n, sp in fusions.items()}
    for c in fc:
        needs = c.pop("_needs") if "geo" in stages else []
        job(f"fusion_fit_{c['fusion']}", "fusion", [c], cmd("fuse_multiscale_products.py", "-c {fcfg}",
            "--stage fit", "-n {cpus}"), cpus=res["fusion_fit"], id_keys=("fusion",), needs=needs)
    if "fusion" in stages:
        for c in fc:
            job(f"fusion_apply_{c['fusion']}", "fusion", [dict(c, shard=i) for i in range(c["nfs"])],
                cmd("fuse_multiscale_products.py", "-c {fcfg}", "--stage apply --shard {shard}/{nfs}", "-n {cpus}"),
                after=[f"fusion_fit_{c['fusion']}"], cpus=res["fusion_apply"], id_keys=("fusion", "shard"))
            job(f"fusion_merge_{c['fusion']}", "fusion", [c],
                cmd("fuse_multiscale_products.py", "-c {fcfg}", "--stage merge --n-shards {nfs}", "-n {cpus}"),
                after=[f"fusion_apply_{c['fusion']}"], cpus=res["fusion_merge"], id_keys=("fusion",),
                provides=[tok("fusion", "all"), tok("fusion", c["fusion"])])
            if sp_vis.get(c["fusion"]):              # fusion.<name>.visuals: make_fusion_visuals.py config
                job(f"fusion_visuals_{c['fusion']}", "fusion", [dict(c, fvis=ab(sp_vis[c["fusion"]]))],
                    cmd("make_fusion_visuals.py", "-c {fvis}"), after=[f"fusion_merge_{c['fusion']}"],
                    cpus=res.get("fusion_visuals", 1), id_keys=("fusion",))
    jv = cfg.get("joint_visuals")
    if jv:
        jv = jv if isinstance(jv, dict) else {"config": jv}
        needs = ([tok("visuals", "all")] if vis_cases and "visuals" in stages else []) + \
                ([tok("fusion", "all")] if fc and "fusion" in stages else [])
        job("joint_visuals", "joint_visuals", [{"jcfg": ab(jv["config"])}], cmd("make_joint_visuals.py", "-c {jcfg}"),
            id_keys=(), needs=needs)

    J += extra_jobs(cfg.get("extra_jobs") or [], J, pipes, tok, ab, train_dir, py, stages)
    names = {j["name"] for j in J}
    for j in J:                          # drop dependencies on omitted stages (assumed done)
        j["after"] = [a for a in j.get("after") or [] if (a if isinstance(a, str) else a["name"]) in names]
        if not j["after"]:
            del j["after"]
    env = dict(launcher.pop("env", None) or {})
    src = str(Path(train_dir).parents[1])          # .../SIT_FUSE/src, so `sit_fuse.*` imports resolve
    env.setdefault("PYTHONPATH", f"{src}:{train_dir}:" + os.environ.get("PYTHONPATH", ""))
    launcher["env"] = env
    return {"launcher": launcher, "jobs": J}, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaml_config", type=Path)
    ap.add_argument("--run", action="store_true", help="run the plan with sit_fuse.local_launcher")
    ap.add_argument("--no-resume", action="store_true",
                    help="plan every stage even when its outputs already exist (default: YAML resume, true)")
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []     # after --: passed to `local_launcher run`
    a = ap.parse_args(argv[:len(argv) - len(extra) - (1 if "--" in argv else 0)])
    cfg = _read(a.yaml_config)
    base = a.yaml_config.resolve().parent
    plan, report = build_plan(cfg, base, False if a.no_resume else None)
    if report:
        print("existing outputs (done = skipped; launcher markers handle the rest):\n" + "\n".join(report))
    out = Path(cfg.get("plan_file") or base / f"{a.yaml_config.stem}.plan.yaml")
    out = out if out.is_absolute() else base / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(plan, sort_keys=False, width=200))
    try:                                                   # cases / grid expanded as the launcher does
        from sit_fuse.local_launcher import _expand_params
        n = sum(len(list(_expand_params(j))) for j in plan["jobs"])
    except Exception:  # noqa: BLE001
        n = sum(len(j.get("cases") or [0]) for j in plan["jobs"])
    run = [sys.executable, "-m", "sit_fuse.local_launcher", "run", str(out)] + extra
    print(f"wrote {out}: {len(plan['jobs'])} job specs, {n} jobs\n  " + " ".join(map(shlex.quote, run)))
    if a.run:
        env = dict(os.environ, PYTHONPATH=plan["launcher"]["env"]["PYTHONPATH"])
        sys.exit(subprocess.call(run, env=env))


if __name__ == "__main__":
    main()
