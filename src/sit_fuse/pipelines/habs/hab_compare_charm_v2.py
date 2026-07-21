import argparse
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pprint import pprint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from netCDF4 import Dataset
from pyresample import geometry, kd_tree
from pyresample.utils.rasterio import get_area_def_from_raster


from enum import Enum


INST_SUBDIRS = {
    "modis": "AQUA_MODIS",
    "jpss1": "JPSS1_VIIRS",
    "jpss2": "JPSS2_VIIRS",
    "pace": "PACE_OCI",
    "s3a": "S3A_OLCI_ERRNT",
    "s3b": "S3B_OLCI_ERRNT",
    "snpp": "SNPP_VIIRS",
}

PRODUCT_RE = {
    "pnd": r"(\d{8})_DAY.pseudo_nitzschia_delicatissima_bloom.tif",
    "pns": r"(\d{8})_DAY.pseudo_nitzschia_seriata_bloom.tif",
    'pda': r"(\d{8})_DAY.particulate_domoic_acid.tif"
}

DEFAULTS = {
    "sf_base_dir": "/mnt/data/HAB_Data_SIT_FUSE/MERGED_HAB_20250225_S_CA/",
    "charm_files": [
        #"/mnt/data/CHARM/charmForecast0day_LonPM180.nc",
        "/mnt/data/CHARM/wvcharmV3_0day_LonPM180.nc",
    ],
 
    "pn_combine_modes": ["sum_capped", "max_severity"],
    "dt_start": "20240101",
    "dt_end": "20251231",
    "charm_thresh": 0.75,
    "sf_thresh": 2,
    "radius_of_influence": 10000,
    "instrument_order": ["modis", "snpp", "jpss1", "s3a", "s3b", "pace", "jpss2"],
    "product_order": ["pnd", "pns", 'pda'],
    "severity_levels": [1, 2, 3, 4, 5, 6],
    "output_dir": ".",
    "write_pickle": True,
    "write_geotiff": True,
    "write_boxplots": True,
    "pickle_filename": "hab_compare_charm_histograms.pkl",
    "diff_map_filename": "TOTAL_DIFF_MAP.tif",
    "total_diffs_filename": "TOTAL_DIFFS.tif",
}



class PnCombineMode(str, Enum):
    OR_PRESENCE = "or_presence"        # presence if either stream has severity > 0
    AND_PRESENCE = "and_presence"      # presence only if both streams have severity > 0
    MAX_SEVERITY = "max_severity"      # combined severity is max(pnd, pns)
    SUM_CAPPED = "sum_capped"          # combined severity is min(pnd + pns, max_level)
    MEAN_SEVERITY = "mean_severity"
    MEAN_CEIL_SEVERITY = "mean_ceil_severity"
    MEAN_FLOOR_SEVERITY = "mean_floor_severity"

@dataclass(frozen=True)
class ComparisonConfig:
    sf_base_dir: str
    charm_files: Tuple[str]
    dt_start: datetime
    dt_end: datetime
    charm_thresh: float
    sf_thresh: float
    radius_of_influence: int
    instrument_order: Tuple[str, ...]
    product_order: Tuple[str, ...]
    severity_levels: Tuple[int, ...]
    pn_combine_modes: Tuple[str, ...]
    output_dir: str
    write_pickle: bool
    write_geotiff: bool
    write_boxplots: bool
    pickle_filename: str
    diff_map_filename: str
    total_diffs_filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CHARM and SIT-FUSE HAB outputs using a YAML config.")
    parser.add_argument("-y", "--yaml", required=True, help="Path to YAML config file.")
    return parser.parse_args()


def parse_date(value: str) -> datetime:
    return datetime.strptime(str(value), "%Y%m%d")


def load_yaml_config(yaml_path: Path) -> ComparisonConfig:
    with yaml_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}

    merged = {**DEFAULTS, **user_cfg}
    charm_files = tuple(merged["charm_files"])

    return ComparisonConfig(
        sf_base_dir=str(merged["sf_base_dir"]),
        charm_files=[str(charm_files[0])], #str(charm_files[0]), str(charm_files[1])),
        dt_start=parse_date(merged["dt_start"]),
        dt_end=parse_date(merged["dt_end"]),
        charm_thresh=float(merged["charm_thresh"]),
        sf_thresh=float(merged["sf_thresh"]),
        radius_of_influence=int(merged["radius_of_influence"]),
        instrument_order=tuple(merged["instrument_order"]),
        pn_combine_modes=tuple(merged["pn_combine_modes"]),
        product_order=tuple(merged["product_order"]),
        severity_levels=tuple(int(x) for x in merged["severity_levels"]),
        output_dir=str(merged["output_dir"]),
        write_pickle=bool(merged["write_pickle"]),
        write_geotiff=bool(merged["write_geotiff"]),
        write_boxplots=bool(merged.get("write_boxplots", True)),
        pickle_filename=str(merged["pickle_filename"]),
        diff_map_filename=str(merged["diff_map_filename"]),
        total_diffs_filename=str(merged["total_diffs_filename"]),
    )


def combine_pnd_pns_ordinal(
    pnd_severity: np.ndarray,
    pns_severity: np.ndarray,
    mode: PnCombineMode,
    invalid_value: int = -1,
    max_level: int = 6,
) -> np.ndarray:
    """
    Combine pnd and pns ordinal severity fields into a single ordinal severity field.

    pnd_severity, pns_severity: integer bin indices for SIT-FUSE pnd/pns.
    invalid_value: value used to mark invalid pixels.
    max_level: maximum ordinal severity level to cap SUM_CAPPED.
    """

    # Invalid where either stream is invalid
    invalid = (pnd_severity == invalid_value) | (pns_severity == invalid_value)

    # Presence masks in ordinal space: severity > 0 means some level of bloom
    pnd_present = (pnd_severity > 0) & ~invalid
    pns_present = (pns_severity > 0) & ~invalid

    if mode == PnCombineMode.OR_PRESENCE:
        # Any presence -> use max severity level across streams
        combined = np.where(
            invalid,
            invalid_value,
            np.where(pnd_present | pns_present,
                     np.maximum(pnd_severity, pns_severity),
                     0),
        )

    elif mode == PnCombineMode.AND_PRESENCE:
        # Only where both have presence -> max severity; else 0
        combined = np.where(
            invalid,
            invalid_value,
            np.where(pnd_present & pns_present,
                     np.maximum(pnd_severity, pns_severity),
                     0),
        )

    elif mode == PnCombineMode.MAX_SEVERITY:
        # Always take max severity if valid; keep 0 where both are 0
        combined = np.where(
            invalid,
            invalid_value,
            np.maximum(pnd_severity, pns_severity),
        )

    elif mode == PnCombineMode.SUM_CAPPED:
        combined = np.where(
            invalid,
            invalid_value,
            np.minimum(pnd_severity + pns_severity, max_level),
        )

    elif mode == PnCombineMode.MEAN_SEVERITY:
        combined = np.where(
            invalid,
            invalid_value,
            np.rint((pnd_severity.astype(np.float32) + pns_severity.astype(np.float32)) / 2.0),
        )       

    elif mode == PnCombineMode.MEAN_CEIL_SEVERITY:
        combined = np.where(
            invalid,
            invalid_value,
            np.ceil((pnd_severity.astype(np.float32) + pns_severity.astype(np.float32)) / 2.0),
        )

    elif mode == PnCombineMode.MEAN_FLOOR_SEVERITY:
        combined = np.where(
            invalid,
            invalid_value,
            np.floor((pnd_severity.astype(np.float32) + pns_severity.astype(np.float32)) / 2.0),
        ) 

    else:
        raise ValueError(f"Unknown PnCombineMode: {mode}")

    return combined.astype(np.int16)

def make_combined_sf(
    pnd_path: str,
    pns_path: str,
    mode: PnCombineMode,
    config: ComparisonConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    # Load pnd and pns as you already do
    pnd_binned, pnd_severity = load_sf_product(pnd_path, config)
    pns_binned, pns_severity = load_sf_product(pns_path, config)

    combined_severity = combine_pnd_pns_ordinal(
        pnd_severity,
        pns_severity,
        mode=mode,
        invalid_value=-1,
        max_level=max(config.severity_levels),
    )

    combined_binned = binarize_sf_image(
        combined_severity,
        threshold=config.sf_thresh,
    )

    return combined_binned, combined_severity


def compare_combined_pn_for_date(
    instrument: str,
    date_str: str,
    pnd_path: Optional[str],
    pns_path: Optional[str],
    charm_arr: np.ndarray,
    mode: PnCombineMode,
    config: ComparisonConfig,
) -> Optional[Dict[str, Any]]:
    if not pnd_path or not pns_path:
        return None

    combined_binned, combined_severity = make_combined_sf(
        pnd_path, pns_path, mode, config
    )

    concentration_metrics = compute_concentration_metrics(
        combined_severity, combined_binned, charm_arr, config
    )
    overall_metrics = compute_overall_metrics(
        combined_severity, combined_binned, charm_arr
    )

    product_key = f"pn_combined_{mode.value}"

    return {
        "date": date_str,
        "instrument": instrument,
        "product": product_key,
        "pnd_path": pnd_path,
        "pns_path": pns_path,
        "concentration_metrics": concentration_metrics,
        "overall_metrics": {
            k: v for k, v in overall_metrics.items() if isinstance(v, dict)
        },
        "aligned_sf": overall_metrics["aligned_sf"],
        "aligned_charm": overall_metrics["aligned_charm"],
    }



def initialize_product_catalog() -> Dict[str, Dict[str, Dict[str, str]]]:
    return {instrument: {product: {} for product in PRODUCT_RE} for instrument in INST_SUBDIRS}


def find_sit_fuse_products(config: ComparisonConfig) -> Dict[str, Dict[str, Dict[str, str]]]:
    products = initialize_product_catalog()
    for instrument, subdir in INST_SUBDIRS.items():
        walk_dir = os.path.join(config.sf_base_dir, subdir)
        for root, _, files in os.walk(walk_dir):
            for filename in files:
                for product, pattern in PRODUCT_RE.items():
                    match = re.search(pattern, filename)
                    if not match:
                        continue
                    date_str = match.group(1)
                    dt = datetime.strptime(date_str, "%Y%m%d")
                    if config.dt_start <= dt <= config.dt_end:
                        products[instrument][product][date_str] = os.path.join(root, filename)
    return products


def first_available_sf_filename(sf_products: Dict[str, Dict[str, Dict[str, str]]]) -> str:
    for instrument_data in sf_products.values():
        for product_data in instrument_data.values():
            for path in product_data.values():
                return path
    raise RuntimeError("Could not find any SIT-FUSE product files in the configured directory tree.")


def create_charm_area_definition(lat: np.ndarray, lon: np.ndarray) -> geometry.AreaDefinition:
    return geometry.AreaDefinition(
        "charm_area",
        "CHARM source grid",
        "EPSG:4326",
        "EPSG:4326",
        lat.shape[0],
        lon.shape[0],
        (lon.min(), lat.min(), lon.max(), lat.max()),
    )


def binarize_charm_cube(data_arr: np.ndarray, threshold: float) -> np.ndarray:
    out = np.array(data_arr, copy=True)
    invalid = out < 0.0
    out[out < threshold] = 0
    out[out >= threshold] = 1
    out[invalid] = -1.0
    return out.astype(np.int16)


def load_and_regrid_charm(charm_fname: str, sf_fname: str, config: ComparisonConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Dataset(charm_fname) as nc:
        nc.set_auto_maskandscale(False)
        lat = nc.variables["latitude"][:]
        lon = nc.variables["longitude"][:]
        nc.set_auto_maskandscale(True)
        data_arr = nc.variables["pseudo_nitzschia"][:]
        time_arr = nc.variables["time"][:]

    data_arr = np.moveaxis(data_arr, 0, 2)
    data_arr = np.swapaxes(data_arr, 0, 1)
    data_arr = np.flipud(data_arr)
    binned = binarize_charm_cube(data_arr, config.charm_thresh)

    charm_area_def = create_charm_area_definition(lat, lon)
    target_area_def = get_area_def_from_raster(sf_fname)
    regridded = np.squeeze(
        kd_tree.resample_nearest(
            charm_area_def,
            binned,
            target_area_def,
            radius_of_influence=config.radius_of_influence,
            fill_value=-1,
        )
    )
    return regridded, data_arr, time_arr


def binarize_sf_image(sf_image: np.ndarray, threshold: float) -> np.ndarray:
    out = np.array(sf_image, copy=True)
    invalid = out < 1.0
    out[out < threshold] = 0
    out[out >= threshold] = 1
    out[invalid] = -1.0
    return out.astype(np.int16)


def load_sf_product(sf_fname: str, config: ComparisonConfig) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(sf_fname) as src:
        sf_image = np.squeeze(src.read(1))
    return binarize_sf_image(sf_image, config.sf_thresh), sf_image


def init_weighted_series() -> Dict[str, List[Any]]:
    return {"f1": [], "count": [], "confusion_matrix": []}


def init_concentration_histogram(config: ComparisonConfig) -> Dict[int, Dict[str, List[Any]]]:
    return {level: init_weighted_series() for level in config.severity_levels}


def safe_weighted_average(values: List[float], weights: List[float]) -> Optional[float]:
    if not values or not weights or sum(weights) <= 0:
        return None
    return float(np.average(values, weights=weights))


def compute_binary_f1_arrays(sf_data_comp: np.ndarray, charm_arr_comp: np.ndarray, positive_values=[0,1]) -> Tuple[float, int, int, int, int]:


    f1s = []
    counts = []
    fps = []
    fns = []
    tps = []
    for cls in positive_values:
        tp = int(np.count_nonzero((sf_data_comp == cls) & (charm_arr_comp == cls)))
        fp = int(np.count_nonzero((sf_data_comp == cls) & (charm_arr_comp != cls) & (charm_arr_comp >= 0)))
        fn = int(np.count_nonzero((sf_data_comp != cls) & (sf_data_comp >= 0) & (charm_arr_comp == cls)))

        total = tp + fn 

        if total == 0: #Catching performance for all classes, and averaging, so skipping cases where class does not exist
            continue
            #return np.nan, total, tp, fp, fn

        counts.append(total)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

        if (tp + fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + fp)

        if (tp + fn) == 0:
            recall = 1
        else:
            recall = tp / (tp + fn)
        
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = 2 * ((precision * recall) / (precision + recall))

        f1s.append(float(f1))



    if len(f1s) == 0:
        return [np.nan], counts, tps, fps, fns

    return f1s, counts, tps, fps, fns


def masked_pair_from_indices(sf_binned: np.ndarray, charm_arr: np.ndarray, mask_indices: Tuple[np.ndarray, ...]) -> Tuple[np.ndarray, np.ndarray]:

    sf_comp = np.full(sf_binned.shape, -1, dtype=np.int16)
    charm_comp = np.full(sf_binned.shape, -1, dtype=np.int16)
    sf_comp[mask_indices] = sf_binned[mask_indices]
    charm_comp[mask_indices] = charm_arr[mask_indices]
    return sf_comp, charm_comp


def confusion_matrix_to_list(tp: int, fp: int, fn: int, tn: Optional[int] = None) -> List[List[int]]:
    if tn is None:
        tn = 0
    return [[int(tn), int(fp)], [int(fn), int(tp)]]


def update_weighted_hist(series: Dict[str, List[Any]], f1: float, count: int, confusion_matrix: Optional[List[List[int]]] = None) -> None:
 
    for f in range(len(f1)):
        if np.isnan(f1[f]) or count[f] <= 0:
            continue
        series["f1"].append(float(f1[f]))
        series["count"].append(int(count[f]))
        #series["confusion_matrix"].append(confusion_matrix)


def compute_concentration_metrics(
    sf_data: np.ndarray,
    sf_binned: np.ndarray,
    charm_arr: np.ndarray,
    config: ComparisonConfig,
) -> Dict[int, Dict[str, Any]]:
    metrics: Dict[int, Dict[str, Any]] = {}

    for level in config.severity_levels:
        level_indices = np.where(sf_data == level)

        sf_comp, charm_comp = masked_pair_from_indices(sf_binned, charm_arr, level_indices)

        positive_values = [0]
        if level > 1:
            positive_values = [1]

        f1, total, tp, fp, fn = compute_binary_f1_arrays(sf_comp, charm_comp, positive_values=positive_values)
        metrics[level] = {
            "f1": f1,
            "count": total,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            #"class_value": 1,
            #"confusion_matrix": confusion_matrix_to_list(tp, fp, fn),
        }

    return metrics


def compute_overall_metrics(sf_data: np.ndarray, sf_binned: np.ndarray, charm_arr: np.ndarray) -> Dict[str, Any]:
    valid_indices = np.where((sf_data > -1) & (charm_arr > -1))
    sf_comp, charm_comp = masked_pair_from_indices(sf_binned, charm_arr, valid_indices)
    f1, total, tp, fp, fn = compute_binary_f1_arrays(sf_comp, charm_comp)
    return {
        "presence": {
            "f1": f1,
            "count": total,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "aligned_sf": sf_comp,
        "aligned_charm": charm_comp,
    }


def choose_charm_slice(
    instrument: str,
    time_index: int,
    charm_standard: np.ndarray,
    charm_pace: np.ndarray,
    time_standard: np.ndarray,
    time_pace: np.ndarray,
) -> Optional[Tuple[str, np.ndarray]]:
    if instrument == "pace":
        if time_index >= time_pace.shape[0]:
            return None
        charm_dt = datetime.fromtimestamp(time_pace[time_index], tz=timezone.utc)
        charm_arr = np.squeeze(charm_pace[:, :, time_index])
    else:
        charm_dt = datetime.fromtimestamp(time_standard[time_index], tz=timezone.utc)
        charm_arr = np.squeeze(charm_standard[:, :, time_index])

    return charm_dt.strftime("%Y%m%d"), charm_arr


def initialize_histograms(config: ComparisonConfig) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    hist_by_concentration = {product: init_concentration_histogram(config) for product in config.product_order}
    hist_by_inst = {instrument: {product: init_weighted_series() for product in config.product_order} for instrument in config.instrument_order}
    hist_total = {product: init_weighted_series() for product in config.product_order}
    return hist_by_concentration, hist_by_inst, hist_total


def initialize_date_histograms(config: ComparisonConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    hist_by_date = {product: {} for product in config.product_order}
    hist_by_date_and_instrument = {
        product: {instrument: {} for instrument in config.instrument_order}
        for product in config.product_order
    }
    return hist_by_date, hist_by_date_and_instrument


def initialize_payload(config: ComparisonConfig, template_sf_fname: str) -> Dict[str, Any]:
    with rasterio.open(template_sf_fname) as src:
        template_profile = src.profile.copy()

    return {
        "metadata": {
            "sf_base_dir": config.sf_base_dir,
            "charm_files": list(config.charm_files),
            "dt_range": [config.dt_start.strftime("%Y%m%d"), config.dt_end.strftime("%Y%m%d")],
            "charm_thresh": config.charm_thresh,
            "sf_thresh": config.sf_thresh,
            "radius_of_influence": config.radius_of_influence,
            "instrument_order": list(config.instrument_order),
            "product_order": list(config.product_order),
            "severity_levels": list(config.severity_levels),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "template_raster": template_sf_fname,
            "template_profile": template_profile,
        },
        "hist_by_concentration": {},
        "hist_by_inst": {},
        "hist_total": {},
        "hist_by_date": {},
        "hist_by_date_and_instrument": {},
        "comparisons": {},
        "diff_map": None,
        "total_diffs": None,
    }


def update_diff_maps(
    diff_map: Optional[np.ndarray],
    total_diffs: Optional[np.ndarray],
    sf_comp: np.ndarray,
    charm_comp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if diff_map is None:
        diff_map = np.zeros(sf_comp.shape, dtype=np.float32)
        total_diffs = np.zeros(sf_comp.shape, dtype=np.float32)
    diff_indices = np.where(((sf_comp == 0) & (charm_comp == 1)) | ((sf_comp == 1) & (charm_comp == 0)))
    total_indices = np.where((sf_comp >= 0) & (charm_comp >= 0))
    diff_map[diff_indices] += 1
    total_diffs[total_indices] += 1
    return diff_map, total_diffs


def finalize_diff_map(diff_map: Optional[np.ndarray], total_diffs: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if diff_map is None or total_diffs is None:
        return None
    out = np.array(diff_map, copy=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(out, total_diffs, out=np.full_like(out, np.nan), where=total_diffs > 0)
    return out


def compare_for_date(
    instrument: str,
    product: str,
    date_str: str,
    sf_path: str,
    charm_arr: np.ndarray,
    config: ComparisonConfig,
) -> Dict[str, Any]:
    sf_binned, sf_data = load_sf_product(sf_path, config)
    concentration_metrics = compute_concentration_metrics(sf_data, sf_binned, charm_arr, config)


    overall_metrics = compute_overall_metrics(sf_data, sf_binned, charm_arr)
    return {
        "date": date_str,
        "instrument": instrument,
        "product": product,
        "sf_path": sf_path,
        "concentration_metrics": concentration_metrics,
        "overall_metrics": {k: v for k, v in overall_metrics.items() if isinstance(v, dict)},
        "aligned_sf": overall_metrics["aligned_sf"],
        "aligned_charm": overall_metrics["aligned_charm"],
    }


def make_boxplot(data_by_label: Dict[str, List[float]], title: str, ylabel: str, output_path: Path) -> None:
    filtered = {key: val for key, val in data_by_label.items() if len(val) > 0}
    if not filtered:
        return
    labels = list(filtered.keys())
    values = [filtered[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(labels)), 6))
    ax.boxplot(values)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Group")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def flatten_grouped_f1(series_by_key: Dict[str, Dict[str, List[Any]]]) -> Dict[str, List[float]]:
    return {key: value.get("f1", []) for key, value in series_by_key.items()}


def flatten_nested_grouped_f1(series_by_outer: Dict[str, Dict[str, Dict[str, List[Any]]]]) -> Dict[str, List[float]]:
    flattened: Dict[str, List[float]] = {}
    for outer_key, inner_dict in series_by_outer.items():
        for inner_key, value in inner_dict.items():
            flattened[f"{outer_key}:{inner_key}"] = value.get("f1", [])
    return flattened


def save_boxplots(payload: Dict[str, Any], output_dir: Path, config: ComparisonConfig) -> None:
    make_boxplot(
        flatten_grouped_f1(payload["hist_total"]),
        "F1 by product",
        "F1 score",
        output_dir / "validation_boxplot_products.png",
    )

    instrument_grouped: Dict[str, Dict[str, List[Any]]] = {}
    for instrument in config.instrument_order:
        combined = init_weighted_series()
        for product in config.product_order:
            combined["f1"].extend(payload["hist_by_inst"][instrument][product]["f1"])
            combined["count"].extend(payload["hist_by_inst"][instrument][product]["count"])
            #combined["confusion_matrix"].extend(payload["hist_by_inst"][instrument][product]["confusion_matrix"])
        instrument_grouped[instrument] = combined
    make_boxplot(
        flatten_grouped_f1(instrument_grouped),
        "F1 by instrument",
        "F1 score",
        output_dir / "validation_boxplot_instruments.png",
    )

    severity_grouped: Dict[str, Dict[str, List[Any]]] = {str(level): init_weighted_series() for level in config.severity_levels}
    for product in config.product_order:
        for level in config.severity_levels:
            severity_grouped[str(level)]["f1"].extend(payload["hist_by_concentration"][product][level]["f1"])
            severity_grouped[str(level)]["count"].extend(payload["hist_by_concentration"][product][level]["count"])
            #severity_grouped[str(level)]["confusion_matrix"].extend(payload["hist_by_concentration"][product][level]["confusion_matrix"])
    make_boxplot(
        flatten_grouped_f1(severity_grouped),
        "F1 by severity level",
        "F1 score",
        output_dir / "validation_boxplot_severity.png",
    )

    #pprint(payload["hist_by_date"])
    make_boxplot(
        flatten_nested_grouped_f1(payload["hist_by_date"]),
        "F1 by product and date",
        "F1 score",
        output_dir / "validation_boxplot_product_date.png",
    )

    make_boxplot(
        flatten_nested_grouped_f1(payload["hist_by_date_and_instrument"]),
        "F1 by product, instrument, and date",
        "F1 score",
        output_dir / "validation_boxplot_product_instrument_date.png",
    )

    concentration_nested: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
    for product in config.product_order:
        concentration_nested[product] = {str(level): payload["hist_by_concentration"][product][level] for level in config.severity_levels}
    make_boxplot(
        flatten_nested_grouped_f1(concentration_nested),
        "F1 by product and severity level",
        "F1 score",
        output_dir / "validation_boxplot_product_severity.png",
    )

    inst_nested: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
    for instrument in config.instrument_order:
        inst_nested[instrument] = {product: payload["hist_by_inst"][instrument][product] for product in config.product_order}
    make_boxplot(
        flatten_nested_grouped_f1(inst_nested),
        "F1 by instrument and product",
        "F1 score",
        output_dir / "validation_boxplot_instrument_product.png",
    )


def run_comparison(config: ComparisonConfig) -> Dict[str, Any]:
    sf_products = find_sit_fuse_products(config)
    template_sf_fname = first_available_sf_filename(sf_products)

    charm_standard, _, time_standard = load_and_regrid_charm(config.charm_files[0], template_sf_fname, config)
    charm_pace, _, time_pace = load_and_regrid_charm(config.charm_files[0], template_sf_fname, config)

    hist_by_concentration, hist_by_inst, hist_total = initialize_histograms(config)
    hist_by_date, hist_by_date_and_instrument = initialize_date_histograms(config)
    payload = initialize_payload(config, template_sf_fname)

    diff_map = None
    total_diffs = None

    for time_index in range(len(time_standard)):
        for instrument in config.instrument_order:
            chosen = choose_charm_slice(instrument, time_index, charm_standard, charm_pace, time_standard, time_pace)
            if chosen is None:
                continue
            date_str, charm_arr = chosen

            for product in config.product_order:
                sf_path = sf_products.get(instrument, {}).get(product, {}).get(date_str)
                if not sf_path:
                    continue

                comparison = compare_for_date(instrument, product, date_str, sf_path, charm_arr, config)
                hist_by_date[product].setdefault(date_str, init_weighted_series())
                hist_by_date_and_instrument[product][instrument].setdefault(date_str, init_weighted_series())

                payload["comparisons"].setdefault(product, {}).setdefault(instrument, {})[date_str] = {
                    "sf_path": sf_path,
                    "concentration_metrics": comparison["concentration_metrics"],
                    "overall_metrics": comparison["overall_metrics"],
                }

                for level, metric in comparison["concentration_metrics"].items():
                    update_weighted_hist(
                        hist_by_concentration[product][level],
                        metric["f1"],
                        metric["count"],
                        #metric.get("confusion_matrix"),
                    )

                metric = comparison["overall_metrics"]["presence"]
                update_weighted_hist(
                    hist_by_inst[instrument][product],
                    metric["f1"],
                    metric["count"],
                    #metric.get("confusion_matrix"),
                )
                update_weighted_hist(
                    hist_total[product],
                    metric["f1"],
                    metric["count"],
                    #metric.get("confusion_matrix"),
                )
                update_weighted_hist(
                    hist_by_date[product][date_str],
                    metric["f1"],
                    metric["count"],
                    #metric.get("confusion_matrix"),
                )
                update_weighted_hist(
                    hist_by_date_and_instrument[product][instrument][date_str],
                    metric["f1"],
                    metric["count"],
                    #metric.get("confusion_matrix"),
                )

                diff_map, total_diffs = update_diff_maps(
                    diff_map,
                    total_diffs,
                    comparison["aligned_sf"],
                    comparison["aligned_charm"],
                )

            pnd_path = sf_products.get(instrument, {}).get("pnd", {}).get(date_str)
            pns_path = sf_products.get(instrument, {}).get("pns", {}).get(date_str)

            for mode_str in config.pn_combine_modes:
                mode = PnCombineMode(mode_str)
                combined_comp = compare_combined_pn_for_date(
                    instrument, date_str, pnd_path, pns_path, charm_arr, mode, config
                )
                if combined_comp is None:
                    continue

                product_key = combined_comp["product"]

                # Store comparison
                payload["comparisons"].setdefault(product_key, {}).setdefault(
                    instrument, {}
                )[date_str] = {
                    "pnd_path": pnd_path,
                    "pns_path": pns_path,
                    "concentration_metrics": combined_comp["concentration_metrics"],
                    "overall_metrics": combined_comp["overall_metrics"],
                }

                # Update histograms using your existing pattern
                metric = combined_comp["overall_metrics"]["presence"]
                hist_by_inst.setdefault(instrument, {}).setdefault(
                    product_key, init_weighted_series()
                )
                hist_total.setdefault(product_key, init_weighted_series())

                update_weighted_hist(
                    hist_by_inst[instrument][product_key],
                    metric["f1"],
                    metric["count"],
                )
                update_weighted_hist(
                    hist_total[product_key],
                    metric["f1"],
                    metric["count"],
                )

                # Concentration histograms
                if product_key not in hist_by_concentration:
                    hist_by_concentration[product_key] = init_concentration_histogram(config)
                    for level, level_metric in combined_comp["concentration_metrics"].items():
                        update_weighted_hist(
                            hist_by_concentration[product_key][level],
                            level_metric["f1"],
                            level_metric["count"],
                        )

                product_key = combined_comp["product"]
  
                if product_key not in hist_by_concentration:
                    hist_by_concentration[product_key] = init_concentration_histogram(config)

                if product_key not in hist_total:
                    hist_total[product_key] = init_weighted_series()

                if product_key not in hist_by_date:
                    hist_by_date[product_key] = {}

                if product_key not in hist_by_date_and_instrument:
                    hist_by_date_and_instrument[product_key] = {
                        inst: {} for inst in config.instrument_order
                    }

                if product_key not in hist_by_inst[instrument]:
                    hist_by_inst[instrument][product_key] = init_weighted_series()

                for level, metric in combined_comp["concentration_metrics"].items():
                    update_weighted_hist(
                        hist_by_concentration[product_key][level],
                        metric["f1"],
                        metric["count"],
                    )
 
                metric = combined_comp["overall_metrics"]["presence"]
 
                update_weighted_hist(
                    hist_by_inst[instrument][product_key],
                    metric["f1"],
                    metric["count"],
                )

                update_weighted_hist(
                    hist_total[product_key],
                    metric["f1"],
                    metric["count"],
                )

                hist_by_date[product_key].setdefault(date_str, init_weighted_series())
                update_weighted_hist(
                    hist_by_date[product_key][date_str],
                    metric["f1"],
                    metric["count"],
                )

                hist_by_date_and_instrument[product_key][instrument].setdefault(
                    date_str, init_weighted_series()
                )
                update_weighted_hist(
                    hist_by_date_and_instrument[product_key][instrument][date_str],
                    metric["f1"],
                    metric["count"],
                )


    payload["hist_by_concentration"] = hist_by_concentration
    payload["hist_by_inst"] = hist_by_inst
    payload["hist_total"] = hist_total
    payload["hist_by_date"] = hist_by_date
    payload["hist_by_date_and_instrument"] = hist_by_date_and_instrument
    payload["diff_map"] = finalize_diff_map(diff_map, total_diffs)
    payload["total_diffs"] = total_diffs
    return payload


def save_pickle_payload(payload: Dict[str, Any], output_path: Path) -> None:
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_geotiff(array: np.ndarray, output_path: Path, template_profile: Dict[str, Any], nodata_value: float) -> None:
    profile = template_profile.copy()
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": str(array.dtype),
        "nodata": nodata_value,
    })
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array, 1)


def save_diff_geotiffs(payload: Dict[str, Any], output_dir: Path, config: ComparisonConfig) -> None:
    diff_map = payload.get("diff_map")
    total_diffs = payload.get("total_diffs")
    template_profile = payload.get("metadata", {}).get("template_profile")
    if template_profile is None:
        raise ValueError("Missing template raster profile in payload metadata; cannot write geolocated GeoTIFFs.")

    if diff_map is not None:
        diff_map_to_write = np.where(np.isnan(diff_map), -9999.0, diff_map).astype(np.float32)
        save_geotiff(diff_map_to_write, output_dir / config.diff_map_filename, template_profile, nodata_value=-9999.0)

    if total_diffs is not None:
        total_diffs_to_write = np.where(np.isnan(total_diffs), -1.0, total_diffs).astype(np.float32)
        save_geotiff(total_diffs_to_write, output_dir / config.total_diffs_filename, template_profile, nodata_value=-1.0)




def print_summary(payload: Dict[str, Any], config: ComparisonConfig) -> None:
    products = ordered_products_for_summary(payload, config)

    print("Instruments:", list(config.instrument_order))
    print("Products:", products)
    print()

    for product in products:
        avg_total = safe_weighted_average(
            payload["hist_total"][product]["f1"],
            payload["hist_total"][product]["count"],
        )
        n_total = int(sum(payload["hist_total"][product]["count"]))
        print(f"{product}: overall weighted F1={avg_total}, support={n_total}")

        for instrument in config.instrument_order:
            inst_series = payload["hist_by_inst"].get(instrument, {}).get(product)
            if not inst_series:
                continue
            avg_inst = safe_weighted_average(
                inst_series["f1"],
                inst_series["count"],
            )
            n_inst = int(sum(inst_series["count"]))
            print(f"  {instrument}: weighted F1={avg_inst}, support={n_inst}")

        if product in payload["hist_by_concentration"]:
            for level in config.severity_levels:
                lvl_series = payload["hist_by_concentration"][product][level]
                avg_level = safe_weighted_average(
                    lvl_series["f1"],
                    lvl_series["count"],
                )
                n_level = int(sum(lvl_series["count"]))
                if avg_level is not None:
                    print(f"  severity {level}: weighted F1={avg_level}, support={n_level}")
        print()
    print_combined_pn_summary(payload, config)

    summary = summarize_series(payload["hist_total"][product])
    print(
        f"{product}: weighted F1={summary['weighted_f1']}, "
        f"support={summary['support']}, scenes={summary['n_scenes']}"
    )
    print_combined_pn_severity_deltas(payload, config)


def print_combined_pn_severity_deltas(payload: Dict[str, Any], config: ComparisonConfig) -> None:
    combined_products = sorted(
        [p for p in payload.get("hist_by_concentration", {}) if p.startswith("pn_combined_")]
    )

    if not combined_products:
        print("No combined PN products found in hist_by_concentration.")
        print("Available products:", list(payload.get("hist_by_concentration", {}).keys()))
        return

    print("Combined PN by severity")
    print("-----------------------")
    for product in combined_products:
        print(product)
        for level in config.severity_levels:
            comb = payload["hist_by_concentration"][product][level]
            comb_avg = safe_weighted_average(comb["f1"], comb["count"])
            if comb_avg is None:
                continue

            pnd_avg = safe_weighted_average(
                payload["hist_by_concentration"].get("pnd", {}).get(level, {}).get("f1", []),
                payload["hist_by_concentration"].get("pnd", {}).get(level, {}).get("count", []),
            )
            pns_avg = safe_weighted_average(
                payload["hist_by_concentration"].get("pns", {}).get(level, {}).get("f1", []),
                payload["hist_by_concentration"].get("pns", {}).get(level, {}).get("count", []),
            )

            msg = f"  severity {level}: {comb_avg:.4f}"
            if pnd_avg is not None:
                msg += f", delta vs pnd={comb_avg - pnd_avg:+.4f}"
            if pns_avg is not None:
                msg += f", delta vs pns={comb_avg - pns_avg:+.4f}"
            print(msg)
        print()




def ordered_products_for_summary(payload: Dict[str, Any], config: ComparisonConfig) -> List[str]:
    base = list(config.product_order)
    extras = sorted([p for p in payload["hist_total"].keys() if p not in base])
    return base + extras


def print_combined_pn_summary(payload: Dict[str, Any], config: ComparisonConfig) -> None:
    combined_products = sorted([p for p in payload["hist_total"] if p.startswith("pn_combined_")])
    if not combined_products:
        return

    print("Combined PN vs native products")
    print("------------------------------")

    pnd_avg = safe_weighted_average(
        payload["hist_total"].get("pnd", {}).get("f1", []),
        payload["hist_total"].get("pnd", {}).get("count", []),
    )
    pns_avg = safe_weighted_average(
        payload["hist_total"].get("pns", {}).get("f1", []),
        payload["hist_total"].get("pns", {}).get("count", []),
    )

    print(f"baseline pnd: {pnd_avg}")
    print(f"baseline pns: {pns_avg}")

    for product in combined_products:
        avg_total = safe_weighted_average(
            payload["hist_total"][product]["f1"],
            payload["hist_total"][product]["count"],
        )
        print(f"{product}: {avg_total}")

        if pnd_avg is not None:
            print(f"  delta vs pnd: {avg_total - pnd_avg:+.4f}")
        if pns_avg is not None:
            print(f"  delta vs pns: {avg_total - pns_avg:+.4f}")

        for instrument in config.instrument_order:
            inst_series = payload["hist_by_inst"].get(instrument, {}).get(product)
            if not inst_series or not inst_series["count"]:
                continue

            avg_inst = safe_weighted_average(inst_series["f1"], inst_series["count"])
            pnd_inst = safe_weighted_average(
                payload["hist_by_inst"].get(instrument, {}).get("pnd", {}).get("f1", []),
                payload["hist_by_inst"].get(instrument, {}).get("pnd", {}).get("count", []),
            )
            pns_inst = safe_weighted_average(
                payload["hist_by_inst"].get(instrument, {}).get("pns", {}).get("f1", []),
                payload["hist_by_inst"].get(instrument, {}).get("pns", {}).get("count", []),
            )

            print(f"  {instrument}: {avg_inst}")
            if pnd_inst is not None:
                print(f"    delta vs pnd: {avg_inst - pnd_inst:+.4f}")
            if pns_inst is not None:
                print(f"    delta vs pns: {avg_inst - pns_inst:+.4f}")
        print()


def summarize_series(series: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {
        "weighted_f1": safe_weighted_average(series["f1"], series["count"]),
        "support": int(sum(series["count"])) if series["count"] else 0,
        "n_scenes": len(series["f1"]),
    }



def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.yaml).expanduser())
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = run_comparison(config)

    if config.write_pickle:
        save_pickle_payload(payload, output_dir / config.pickle_filename)
    if config.write_geotiff:
        save_diff_geotiffs(payload, output_dir, config)
    #if config.write_boxplots:
    #    save_boxplots(payload, output_dir, config)

    print_summary(payload, config)


if __name__ == "__main__":
    main()
