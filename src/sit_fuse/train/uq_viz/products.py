"""
Product registry for SIT-FUSE UQ visualization.

Each product (smoke now, fire later) declares which geo-product bands carry the
probability, mask, interval and uncertainty layers, how to color them, and
which comparison references are expected. Everything here can be overridden
from the visuals config (`product_overrides:`), so a new product needs only a
new ProductSpec entry or a config block, not new plotting code.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProductSpec:
    name: str
    label: str                                   # "Smoke", "Fire"
    prob_band: str = "p_1"
    mask_band: str = "pred_label"
    selective_band: str = "pred_label_selective"
    q_lo_band: str | None = "ensemble_p1_q05"
    q_hi_band: str | None = "ensemble_p1_q95"
    uncertainty_bands: dict[str, str] = field(default_factory=lambda: {
        "predictive_entropy": "predictive entropy",
        "ensemble_mutual_information": "ensemble MI (epistemic)",
    })
    excluded_band: str | None = "excluded_cluster_mask"
    prob_cmap: str = "magma"
    unc_cmap: str = "viridis"
    mask_color: str = "#d95f02"
    comparison_color: str = "#1f78b4"
    reference_color: str = "#33a02c"
    # nested probabilistic envelope (q05 >= thr: likely, p >= thr: expected, q95 >= thr: possible)
    envelope_colors: tuple[str, str, str] = ("#fdd0a2", "#fd8d3c", "#a63603")
    area_units: str = "km$^2$"
    # display-only fill of 1-pixel seams between adjacent granules
    fill_seams: bool = True
    log_prob: bool = False                       # fire: very small probabilities -> log color scale
    prob_vmin: float = 0.0
    prob_vmax: float = 1.0
    default_references: dict[str, dict[str, Any]] = field(default_factory=dict)


SMOKE = ProductSpec(
    name="smoke",
    label="Smoke",
    default_references={
        # Roles: ADP primary, VIIRS secondary (both two-sided: smoke / clear / no data), HMS quick look only.
        # Paths come from the visuals config (glob or matchups); references without paths are skipped.
        "adp_smoke": {"type": "raster", "role": "primary", "label": "ADP smoke", "short_label": "ADP",
                      "match": "time", "tolerance_minutes": 2, "value_map": {0: 0, 1: 1},
                      "nodata_values": [255, -1], "resampling": "nearest", "color": "#33a02c"},
        "viirs_aerosol_smoke": {"type": "raster", "role": "secondary", "label": "VIIRS aerosol-type smoke",
                                "short_label": "VIIRS", "match": "time", "tolerance_minutes": 30,
                                "value_map": {0: 0, 1: 1}, "nodata_values": [255], "resampling": "nearest",
                                "color": "#1b9e77"},
        # positive-only; outside HMS is NOT confirmed smoke-free. Polygons (time-matched) by default;
        # set type: raster, match: date, glob: .../*_smoke.tif for daily HMS rasters.
        "hms_smoke": {"type": "hms_polygons", "role": "quicklook", "label": "HMS smoke", "kind": "smoke",
                      "url": "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/"
                             "Shapefile/{Y}/{m}/hms_smoke{Ymd}.zip",
                      "min_density": "Light", "time_match": "interval", "positive_only": True},
    },
)

FIRE = ProductSpec(
    name="fire",
    label="Fire",
    prob_cmap="inferno",
    mask_color="#e31a1c",
    envelope_colors=("#fee391", "#fe9929", "#993404"),
    log_prob=True,
    prob_vmin=1e-4,
    default_references={
        # FDCC is the primary fire reference (scene-grid, bucket-max truth from the matchups);
        # HMS fire points secondary. Paths come from the matchup / attach_references YAMLs.
        "fdcc_fire": {"type": "raster", "label": "GOES FDCC", "field": "truth_fdcc_fire",
                      "value_map": {0: 0, 1: 1}, "nodata_values": [255, -1], "role": "primary"},
        "hms_fire": {"type": "hms_points", "label": "HMS fire",
                     "url": "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Fire_Points/"
                            "Shapefile/{Y}/{m}/hms_fire{Ymd}.zip", "positive_only": True},
    },
)

REGISTRY: dict[str, ProductSpec] = {"smoke": SMOKE, "fire": FIRE}


def get_product(name: str, overrides: dict | None = None) -> ProductSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown product {name!r}; known: {', '.join(REGISTRY)} "
                       "(add a ProductSpec to uq_viz/products.py or define it under product_overrides)")
    spec = copy.deepcopy(REGISTRY[name])
    for k, v in (overrides or {}).items():
        if not hasattr(spec, k):
            raise KeyError(f"product_overrides.{name}: unknown field {k!r}")
        setattr(spec, k, tuple(v) if k == "envelope_colors" else v)
    return spec


# Display names for models in figures (config `model_labels` overrides)
MODEL_LABELS = {
    "hist_gradient_boosting": "HGB (calibrated)",
    "random_forest": "Random forest",
    "logistic_regression": "Logistic regression",
    "soft_assignment": "Soft assignment",
    "assignment_ensemble": "Assignment ensemble",
    "hard_assignment": "Hard assignment (baseline)",
}

MODEL_COLORS = {
    "hist_gradient_boosting": "#d95f02",
    "random_forest": "#7570b3",
    "logistic_regression": "#66a61e",
    "soft_assignment": "#e7298a",
    "assignment_ensemble": "#a6761d",
    "hard_assignment": "#1f78b4",
}

SPLIT_COLORS = {"dev": "#9ecae1", "calibration": "#fdae6b", "holdout": "#d95f02", "unassigned": "#d9d9d9"}
