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
joint_visuals: after every visuals and fusion_merge job.

Resume (resume: true, the default): stages whose outputs already exist are left out of the plan, so a run started
without the launcher (or with a deleted state_dir) continues where it stopped. Finished = upstream finished and outputs
at least as new as upstream outputs: features (build_manifest.json + pixel_features.parquet), train
(training_summary.json), infer per model (inference_manifest.json, written after all scenes), evaluate per model
(eval_summary.json with bootstrap_groups), compare (model_comparison.csv), geo per model (entry in
geo_products_manifest.json). A model whose inference was interrupted is re-planned; its finished scenes are reused
from the per-scene checkpoints (scene_results/) when those exist. Fusion and visuals rely on the launcher markers.
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


def finished(uq, models):
    """
    Stages of one UQ config whose outputs already exist (resume after a run without the launcher, or after the
    launcher state was deleted). A stage counts as finished only when its upstream stage is finished and its
    outputs are at least as new as the upstream outputs, so a rerun upstream invalidates everything below it.
    Returns {"features": bool, "train": bool, "infer": {m: bool}, "evaluate": {m: bool}, "compare": bool,
    "geo": {m: bool}}.
    """
    import json
    out = Path(uq["output_dir"])
    inf, ev, gx = uq.get("inference") or {}, uq.get("evaluation") or {}, uq.get("geo_export") or {}
    bm, feat = out / "build_manifest.json", out / "pixel_features.parquet"
    tsum = Path(inf.get("training_summary_path", out / "training" / "training_summary.json"))
    d = {"features": bm.exists() and feat.exists()}
    d["train"] = d["features"] and _newer(tsum, bm, feat)
    edir = Path(ev.get("output_dir", out / "evaluation"))
    gman = Path(gx.get("output_dir", out / "geo")) / "geo_products_manifest.json"
    try:
        geo_models = set(json.loads(gman.read_text())) if gman.exists() else set()
    except Exception:  # noqa: BLE001
        geo_models = set()
    d["infer"], d["evaluate"], d["geo"], evs = {}, {}, {}, []
    for m in models:
        man = Path(inf["output_dir"]) / "inference_manifest.json" if inf.get("output_dir") else \
            out / "inference" / m / "inference_manifest.json"
        d["infer"][m] = d["train"] and _newer(man, tsum)          # written only after every scene finished
        es = edir / m / "eval_summary.json"
        ok = d["infer"][m] and _newer(es, man)
        if ok:                                                    # older evaluations lack the pairing groups
            try:
                ok = "bootstrap_groups" in json.loads(es.read_text())
            except Exception:  # noqa: BLE001
                ok = False
        d["evaluate"][m] = ok
        evs.append(es)
        d["geo"][m] = ok and m in geo_models and _newer(gman, es)
    d["compare"] = all(d["evaluate"].values()) and _newer(edir / "model_comparison.csv", *evs)
    return d


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
        c = {"pipe": name, "uq": uq}
        pc.append(c)
        uq_cfg = _read(uq)
        models = _models(sp, uq_cfg)
        if resume:
            f = finished(uq_cfg, models)
            done.update({("features", name): f["features"], ("train", name): f["train"],
                         ("compare", name): f["compare"]})
            for st in ("infer", "evaluate", "geo"):
                done.update({(st, name, m): v for m, v in f[st].items()})
            report.append(f"  {name}: features={'done' if f['features'] else 'run'} "
                          f"train={'done' if f['train'] else 'run'} " + " ".join(
                              f"{m}[infer={'done' if f['infer'][m] else 'run'},"
                              f"eval={'done' if f['evaluate'][m] else 'run'},"
                              f"geo={'done' if f['geo'][m] else 'run'}]" for m in models))
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
        cmd("classifier_inference_uq.py", "{uq}", "--model {model} --shard {shard}/{ni}", "-n {cpus}"),
        after=["train"], id_keys=("pipe", "model", "shard"))
    job("infer_merge", "infer", [dict(c, ni=ni[(c["pipe"], c["model"])]) for c in model_cases],
        cmd("classifier_inference_uq.py", "{uq}", "--model {model} --merge --n-shards {ni}", "-n {cpus}"),
        after=["infer"], id_keys=("pipe", "model"))
    job("evaluate", "evaluate", model_cases, cmd("evaluate_uq.py", "{uq}", "--models {model}", "-n {cpus}"),
        after=["infer_merge"], id_keys=("pipe", "model"))
    job("compare", "evaluate", pc, cmd("evaluate_uq.py", "{uq}", "--compare-only", "-n {cpus}"),
        after=["evaluate"])
    job("geo", "geo", model_cases,
        cmd("export_geo_products.py", "{uq}", "--models {model} --update-manifest", "-n {cpus}"),
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
        fc.append({"fusion": name, "fcfg": ab(sp["config"]), "nfs": int(sp.get("shards", shards["fusion"])),
                   "_needs": [tok("geo", p) for p in after if p in geo_planned]})   # finished geo: no wait
    # needs are per job spec, so one fit spec per fusion (each waits only on its own pipelines' geo jobs)
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
                provides=[tok("fusion", "all")])
    jv = cfg.get("joint_visuals")
    if jv:
        jv = jv if isinstance(jv, dict) else {"config": jv}
        needs = ([tok("visuals", "all")] if vis_cases and "visuals" in stages else []) + \
                ([tok("fusion", "all")] if fc and "fusion" in stages else [])
        job("joint_visuals", "joint_visuals", [{"jcfg": ab(jv["config"])}], cmd("make_joint_visuals.py", "-c {jcfg}"),
            id_keys=(), needs=needs)

    names = {j["name"] for j in J}
    for j in J:                          # drop dependencies on omitted stages (assumed done)
        j["after"] = [a for a in j["after"] if a in names]
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
    n = sum(len(j["cases"]) for j in plan["jobs"])
    run = [sys.executable, "-m", "sit_fuse.local_launcher", "run", str(out)] + extra
    print(f"wrote {out}: {len(plan['jobs'])} job specs, {n} jobs\n  " + " ".join(map(shlex.quote, run)))
    if a.run:
        env = dict(os.environ, PYTHONPATH=plan["launcher"]["env"]["PYTHONPATH"])
        sys.exit(subprocess.call(run, env=env))


if __name__ == "__main__":
    main()
