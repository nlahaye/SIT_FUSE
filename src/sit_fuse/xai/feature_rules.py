"""
Rule-based binary environmental features from named channels (generic; numpy only).

A rule is a nested dict:
  {all: [rule, ...]}            every sub-rule true (Kleene logic: a definite False makes the cell valid)
  {any: [rule, ...]}            at least one sub-rule true
  {not: rule}
  condition                     {var (or channel), op, value | percentile, minus, abs, missing, among, scope}
    var / channel   channel name
    minus           optional second channel: term = var - minus
    abs             true: term = |term|
    op              ">", "<", ">=", "<=" (or gt, lt, ge, le)
    value           fixed threshold, or
    percentile      threshold = percentile of the term, pooled over every block (scope: pooled, default) or per
                    block (scope: scene). among: <rule> restricts the cells the percentile is taken over
                    (e.g. the 10th SSS percentile among coastal cells).
    missing         invalid (default): a NaN term makes the condition unknown; false: a NaN term counts as a valid
                    False (e.g. a temperature level below the sea floor cannot hold an inversion).

Blocks are callables name -> array (one per scene, or the used / available track values). resolve() fixes the
pooled percentile thresholds once; evaluate() returns (mask, valid) per block. Paths such as
"river_plume/all[1]" key the thresholds, so a second tool can reuse them (fixed=) for identical absolute cuts.
Kept identical in SIT-FUSE (sit_fuse.xai.feature_rules) and NatureNet (naturenet.drivers.feature_rules).
"""
import numpy as np

OPS = {">": np.greater, "gt": np.greater, "<": np.less, "lt": np.less,
       ">=": np.greater_equal, "ge": np.greater_equal, "<=": np.less_equal, "le": np.less_equal}


def _is_cond(r):
    return not any(k in r for k in ("all", "any", "not"))


def _var(c):
    v = c.get("var", c.get("channel"))
    if v is None:
        raise ValueError(f"feature rule condition without var / channel: {c}")
    return v


def rule_vars(rule):
    """Every channel a rule reads (var, minus, among)."""
    out = []
    if _is_cond(rule):
        out.append(_var(rule))
        if rule.get("minus"):
            out.append(rule["minus"])
        if rule.get("among"):
            out += rule_vars(rule["among"])
    elif "not" in rule:
        out += rule_vars(rule["not"])
    else:
        for r in rule.get("all", rule.get("any", [])):
            out += rule_vars(r)
    return list(dict.fromkeys(out))


def term(c, get):
    x = np.asarray(get(_var(c)), dtype=np.float64)
    if c.get("minus"):
        x = x - np.asarray(get(c["minus"]), dtype=np.float64)
    if c.get("abs"):
        x = np.abs(x)
    return x


def _percentile_of(x, sel, q):
    v = x[sel & np.isfinite(x)]
    return (float(np.nanpercentile(v, q)), int(v.size)) if v.size else (np.nan, 0)


def _children(rule, path):
    if "not" in rule:
        return [(rule["not"], path + "/not")]
    key = "all" if "all" in rule else "any"
    return [(r, f"{path}/{key}[{i}]") for i, r in enumerate(rule[key])]


def resolve(rules, blocks, fixed=None, max_per_block=200000, seed=0):
    """Pooled percentile thresholds {path: value} (fixed entries are kept as given) and a log of rows."""
    thr, log = dict(fixed or {}), []
    rng = np.random.default_rng(seed)

    def walk(rule, path):
        if not _is_cond(rule):
            for r, p in _children(rule, path):
                walk(r, p)
            return
        if rule.get("among"):
            walk(rule["among"], path + "/among")
        if "percentile" not in rule or rule.get("scope", "pooled") == "scene":
            return
        if path in thr:
            log.append(_row(path, rule, thr[path], np.nan, "fixed"))
            return
        vals = []
        for get in blocks:
            x = term(rule, get).reshape(-1)
            sel = np.isfinite(x)
            if rule.get("among"):
                am, av = evaluate(rule["among"], get, thr, path + "/among")
                sel &= (am & av).reshape(-1)
            v = x[sel]
            if v.size > max_per_block:
                v = rng.choice(v, max_per_block, replace=False)
            vals.append(v)
        v = np.concatenate(vals) if vals else np.zeros(0)
        thr[path] = float(np.percentile(v, float(rule["percentile"]))) if v.size else np.nan
        log.append(_row(path, rule, thr[path], int(v.size), "pooled"))

    for name, rule in rules.items():
        walk(rule, name)
    return thr, log


def _row(path, c, value, n, source, block=None):
    return dict(feature=path.split("/")[0], path=path, var=_var(c), minus=c.get("minus"), abs=bool(c.get("abs")),
                op=c.get("op", ">"), percentile=c.get("percentile"), among=bool(c.get("among")), source=source,
                block=block, threshold=value, n=n)


def evaluate(rule, get, thr, path, log=None, block=None):
    """(mask, valid) boolean arrays of one block."""
    if _is_cond(rule):
        x = term(rule, get)
        fin = np.isfinite(x)
        if "value" in rule:
            t = float(rule["value"])
        elif "percentile" in rule:
            if rule.get("scope", "pooled") == "scene":
                sel = np.ones(x.shape, bool)
                if rule.get("among"):
                    am, av = evaluate(rule["among"], get, thr, path + "/among")
                    sel = am & av
                t, n = _percentile_of(x, sel, float(rule["percentile"]))
                if log is not None:
                    log.append(_row(path, rule, t, n, "scene", block))
            else:
                if path not in thr:
                    raise KeyError(f"threshold {path} not resolved (call resolve first)")
                t = thr[path]
        else:
            raise ValueError(f"condition {path} needs value or percentile")
        op = OPS[str(rule.get("op", ">"))]
        with np.errstate(invalid="ignore"):
            m = fin & op(np.where(fin, x, 0.0), t) & np.isfinite(t)
        v = np.ones(x.shape, bool) if str(rule.get("missing", "invalid")) == "false" else fin
        return m, v & bool(np.isfinite(t))
    parts = [evaluate(r, get, thr, p, log, block) for r, p in _children(rule, path)]
    if "not" in rule:
        m, v = parts[0]
        return ~m & v, v
    ms = np.stack([m for m, _ in parts])
    vs = np.stack([v for _, v in parts])
    if "all" in rule:
        definite_false = np.any(vs & ~ms, 0)
        valid = np.all(vs, 0) | definite_false
        return np.all(ms & vs, 0) & valid, valid
    definite_true = np.any(vs & ms, 0)
    valid = np.all(vs, 0) | definite_true
    return definite_true, valid


def evaluate_all(rules, blocks, fixed=None, max_per_block=200000, seed=0, available=None):
    """{name: [(mask, valid) per block]}, the threshold rows. Rules reading channels not in `available`
    (a set of names, optional) are skipped with a message."""
    use = {}
    for name, r in rules.items():
        miss = [c for c in rule_vars(r) if available is not None and c not in available]
        if miss:
            print(f"feature rule {name}: channels {miss} not available - skipped")
            continue
        use[name] = r
    thr, log = resolve(use, blocks, fixed, max_per_block, seed)
    out = {}
    for name, r in use.items():
        out[name] = [evaluate(r, get, thr, name, log, b) for b, get in enumerate(blocks)]
    return out, log


def load_rules(path_or_dict):
    """{name: rule} from a YAML file with a top-level `rules:` mapping, or a dict (either form)."""
    d = path_or_dict
    if isinstance(d, str):
        import yaml
        with open(d) as f:
            d = yaml.safe_load(f)
    d = d or {}
    return dict(d.get("rules", d))


def thresholds_from_csv(path, kinds=("pooled", "fixed")):
    """{path: threshold} from a feature_thresholds CSV (pooled rows only: per-scene rows are not transferable)."""
    import pandas as pd
    t = pd.read_csv(path)
    t = t[t["source"].isin(kinds)]
    return {str(p): float(v) for p, v in zip(t["path"], t["threshold"])}
