import numpy as np
import dask
import dask.array as da
from  datetime import datetime
import copy
import os
import glob
import re
import yaml

from sit_fuse.utils import read_oc_geo
from sit_fuse.compute_resources import merge_compute_conf

from sit_fuse.pipelines.habs.hab_training_constants import *
from sit_fuse.pipelines.habs.hab_scene_split import scene_split_conf, apply_training_split
from sit_fuse.pipelines.habs.hab_sensor_groups import oc_sources
from sit_fuse.pipelines.data_transform_and_prep.goes_combine_and_clip_pipeline import goes_combine_and_clip_pipeline

from sit_fuse.datasets.sf_dataset import run_data_prep
from sit_fuse.train.pretrain_encoder import run_pretraining
from sit_fuse.train.finetune_dc_mlp_head import run_tuning
from sit_fuse.train.train_heirarchichal_deep_cluster import run_heir_training
 
def build_config_fname_data_prep_and_train(config_dir, instrument, with_trop = False):

    config_fname = os.path.join(config_dir, "model", "train_model_" + instrument)

    if with_trop:
        config_fname = config_fname + "_trop"

    config_fname = config_fname + ".yaml"
    return config_fname

def gen_geo_zarr_oc_daily(yml_conf):
 
    start_lon = yml_conf["start_lon"]
    end_lon = yml_conf["end_lon"]
    start_lat = yml_conf["start_lat"]
    end_lat = yml_conf["end_lat"]
    
    fglob = glob.glob(yml_conf["geo_zarr"]["oc_geo_fpattern"])
    # geo_zarr.reader_kwargs: e.g. {pace_res: 1km} for gridded PACE L2 stems
    loc = read_oc_geo(fglob[0], **(yml_conf["geo_zarr"].get("reader_kwargs") or {}))
 
    lat = np.array(loc[0])
    lon = np.array(loc[1])
    inds1 = np.where((lat >= start_lat) & (lat <= end_lat))
    inds2 = np.where((lon >= start_lon) & (lon <= end_lon))

    nind1, nind2 = np.meshgrid(inds2[0], inds1[0])
    lat = lat[inds1]
    lon = lon[inds2]
    loc = np.array(np.meshgrid(lat, lon))

    loc = np.moveaxis(loc, 0,2)
    loc = np.swapaxes(loc,0,1)
 
    data2 = da.from_array(loc)
    da.to_zarr(data2, os.path.join(yml_conf["geo_zarr"]["out_dir"], yml_conf["geo_zarr"]["out_uid"] + ".zarr"))


def build_trop_fname(yml_conf, datetm):

    dirpth = yml_conf["sif_dir"]
    fname = os.path.join(dirpth, "TROPO_redSIF_" + datetm.strftime("%Y-%m-%d") + "_ungridded.nc") 
    return fname



# Band files each reader opens per scene (sit_fuse.utils read_*_oc): <prefix><band>.4km.nc,
# falling back to <prefix><band>.4km.NRT.nc when reader_kwargs.nrt is true.
_VIIRS_SNPP_BANDS = ["RRS.Rrs_410", "RRS.Rrs_443", "RRS.Rrs_486", "RRS.Rrs_551", "RRS.Rrs_671"]
_VIIRS_JPSS_BANDS = ["RRS.Rrs_411", "RRS.Rrs_445", "RRS.Rrs_489", "RRS.Rrs_556", "RRS.Rrs_667"]
_OLCI_BANDS = ["RRS.Rrs_400", "RRS.Rrs_412", "RRS.Rrs_443", "RRS.Rrs_490", "RRS.Rrs_510", "RRS.Rrs_560",
               "RRS.Rrs_620", "RRS.Rrs_665", "RRS.Rrs_674", "RRS.Rrs_681", "RRS.Rrs_709"]
_MODIS_BANDS = ["RRS.Rrs_412", "RRS.Rrs_443", "RRS.Rrs_469", "RRS.Rrs_488", "RRS.Rrs_531", "RRS.Rrs_547",
                "RRS.Rrs_555", "RRS.Rrs_645", "RRS.Rrs_667", "RRS.Rrs_678"]


def _required_bands(prefix):
    """Reader band list for a scene prefix, or None (PACE single file, GOES, unknown)."""
    b = os.path.basename(prefix)
    if b.startswith(("S3A_OLCI", "S3B_OLCI")):
        return _OLCI_BANDS
    if b.startswith("SNPP_VIIRS"):
        return _VIIRS_SNPP_BANDS
    if b.startswith(("JPSS1_VIIRS", "JPSS2_VIIRS")):
        return _VIIRS_JPSS_BANDS
    if b.startswith(("AQUA_MODIS", "TERRA_MODIS")):
        return _MODIS_BANDS
    return None


def _missing_bands(prefix, allow_nrt=True):
    bands = _required_bands(prefix)
    if bands is None:
        return None
    return [v for v in bands if not (os.path.exists(prefix + v + ".4km.nc")
                                     or (allow_nrt and os.path.exists(prefix + v + ".4km.NRT.nc")))]


def _reader_complete_entries(entries, n_files, allow_nrt=True, report=None):
    """Unique entries whose scene has every band file the reader opens.

    Ocean-colour scenes are checked against the reader's band list, so extra files
    matched by input_oc_re (aot_862, angstrom, NRT + refined copies of a band) do not
    reject a scene. Other entries (PACE, GOES, ...) keep the match-count rule."""
    counts, first = {}, {}
    for e in entries:
        k = tuple(e) if isinstance(e, list) else e
        counts[k] = counts.get(k, 0) + 1
        first.setdefault(k, e)
    out = []
    for k in sorted(counts, key=str):
        e = first[k]
        prefix = e[0] if isinstance(e, (list, tuple)) else e
        miss = _missing_bands(prefix, allow_nrt)
        ok = (counts[k] == n_files) if miss is None else (len(miss) == 0)
        if report is not None:
            plat = os.path.basename(prefix).split(".")[0]
            r = report.setdefault(plat, {"scenes": 0, "complete": 0, "incomplete": 0, "example_missing": None})
            r["scenes"] += 1
            r["complete" if ok else "incomplete"] += 1
            if not ok and r["example_missing"] is None:
                r["example_missing"] = (os.path.basename(prefix), miss if miss is not None else
                                        f"{counts[k]} matched files, expected {n_files}")
        if ok:
            out.append(e)
    return out


def _complete_entries(entries, n_files):
    """Sorted unique entries that were matched exactly n_files times."""
    counts = {}
    first = {}
    for e in entries:
        k = tuple(e) if isinstance(e, list) else e
        counts[k] = counts.get(k, 0) + 1
        first.setdefault(k, e)
    return [first[k] for k in sorted(counts, key=str) if counts[k] == n_files]


def find_train_test_fpaths(yml_conf, config_dict, instrument, key):
 
    # With scene_split enabled, train/test come from the stratified day split; the
    # train_min_date / train_max_date window is then not used (and may be omitted).
    use_split = scene_split_conf(yml_conf) is not None
    entry_dates = {}
    if use_split and not (yml_conf.get("train_min_date") and yml_conf.get("train_max_date")):
        train_min, train_max = datetime.min, datetime.max
    else:
        train_min = datetime.strptime(yml_conf["train_min_date"], "%Y%m%d")
        train_max = datetime.strptime(yml_conf["train_max_date"], "%Y%m%d")

    print(train_min, train_max)

    train_fnames = []
    test_fnames = []

    # One or more (input_oc_dir, input_oc_re) sources: sensor groups (e.g. VIIRS =
    # SNPP + JPSS1 + JPSS2, OLCI = S3A + S3B) list one directory per platform.
    # Entries keep their own platform prefix, so every scene is read with its own bands.
    sources = oc_sources(yml_conf["instruments"][instrument][key])
    matched_per_source = {}
    for in_dir, in_re in sources:
      if not os.path.isdir(in_dir):
          print(f"WARNING: {instrument}/{key}: input_oc_dir does not exist: {in_dir}")
      matched_per_source[in_dir] = 0
      for root, dirs, files in os.walk(in_dir):
         
        for fle in files:

            mtch = re.search(str(in_re), fle)
            if mtch:
                matched_per_source[in_dir] += 1
                if "GOES" in instrument and len(mtch.group(2)) != 8:   # raw ABI scan start; daily composites: %Y%m%d
                    dt = datetime.strptime(mtch.group(2), "%Y%j%H%M%S")
                else:
                    dt = datetime.strptime(mtch.group(2), "%Y%m%d")
                lst_entry = os.path.join(root, mtch.group(1))
                #trop/no_trop
                if "troposif" == instrument or "no" not in key:
                    trop_fname = build_trop_fname(yml_conf, dt)
                     
                    lst_entry = [lst_entry, trop_fname]
                entry_dates[os.path.basename(os.path.join(root, mtch.group(1)))] = dt.date()

                print(train_max > train_min, train_max, train_min, dt, dt >= train_min, dt <= train_max)    
                if dt >= train_min and dt <= train_max:
                    train_fnames.append(lst_entry)
                    print(dt, mtch.group(2))  
                else:
                    test_fnames.append(lst_entry)

    # A scene is complete when all NUM_CHANNELS[instrument] band files matched
    # (one match per band file). Counted per entry, so the last scene is kept too.
    # Scene completeness: every band file the reader opens exists (refined, or NRT when
    # reader_kwargs.nrt is true); see _reader_complete_entries.
    allow_nrt = bool(((config_dict.get("data") or {}).get("reader_kwargs") or {}).get("nrt", True))
    report = {}
    final_test_fnames = _reader_complete_entries(test_fnames, NUM_CHANNELS[instrument], allow_nrt, report)
    final_train_fnames = _reader_complete_entries(train_fnames, NUM_CHANNELS[instrument], allow_nrt, report)

    print(f"{instrument}/{key} input files matched per input_oc_dir:")
    for d, n_m in matched_per_source.items():
        print(f"   {n_m:8d}  {d}")
        if n_m == 0:
            print(f"WARNING: {instrument}/{key}: no file in {d} matched input_oc_re")
    print(f"{instrument}/{key} scenes per platform (complete / incomplete):")
    for plat, r in sorted(report.items()):
        print(f"   {plat:16s} {r['complete']:6d} / {r['incomplete']:6d}"
              + (f"   e.g. {r['example_missing']}" if r["example_missing"] else ""))
    for plat, r in sorted(report.items()):
        if r["complete"] == 0:
            print(f"WARNING: {instrument}/{key}: {plat} has no complete scene and is left out of the model")


    if use_split:
        uniq, seen = [], set()
        for e in final_train_fnames + final_test_fnames:
            k = str(e)
            if k not in seen:
                seen.add(k)
                uniq.append(e)
        return apply_training_split(yml_conf, config_dict, uniq, entry_dates)

    config_dict["data"]["files_train"] = list(set(final_train_fnames))
    config_dict["data"]["files_test"] = list(set(final_test_fnames))

    return config_dict


def update_training_config(yml_conf, config_dict, instrument, key):

    config_dict = find_train_test_fpaths(yml_conf, config_dict, instrument, key)

    region_bb = REGION_BBS[yml_conf["roi"]]
    for bb_key in region_bb:
        config_dict["data"]["reader_kwargs"][bb_key] = region_bb[bb_key]
    


    if "no" not in key:
            config_dict["data"]["reader_type"] = READER_TYPE_MAP["TROP"] 
    else:
        for rt_key in READER_TYPE_MAP:
            if rt_key == "TROP":
                continue

            if instrument in rt_key:
                config_dict["data"]["reader_type"] = READER_TYPE_MAP[rt_key] 


    print(config_dict["data"]["reader_type"], config_dict["data"]["files_train"])

    dbn_arch = DBN_ARCH_MAP["DEFAULT"] 
    for arch_key in DBN_ARCH_MAP:
        if instrument in arch_key:
            dbn_arch = DBN_ARCH_MAP[arch_key]

    config_dict["dbn"]["dbn_arch"] = dbn_arch

    config_dict["output"]["out_dir"] = os.path.join(yml_conf["output"], instrument + "_" + key)

    # instruments.<INST>.<key>.model_overrides: deep-merged into the model YAML
    # (e.g. data.number_channels, data.reader_kwargs.pace_res for non-default inputs)
    _merge_overrides(config_dict, yml_conf["instruments"][instrument][key].get("model_overrides"))

    # Carry the pipeline's compute: section into the model YAML so training
    # (run_heir_training) and later inference use the same limits.
    merge_compute_conf(yml_conf, config_dict)
    nlw = (config_dict.get("compute") or {}).get("num_loader_workers")
    if nlw is not None:
        config_dict["data"]["num_loader_workers"] = int(nlw)

    os.makedirs(config_dict["output"]["out_dir"], exist_ok=True)

    return config_dict


def _merge_overrides(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge_overrides(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def run_data_preprocessing(yml_conf):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    configs = {}
    
    for instrument in instrument_dict:
        if instrument not in configs:
            configs[instrument] = {}
        for key in instrument_dict[instrument]:
            if key not in configs[instrument]:
                configs[instrument][key] = {}

            trop = True
            if "no" in key: #Key is trop no_trop
                trop = False
              
            config_dict = copy.deepcopy(YAML_TEMPLATE_HAB_TRAIN)
 
            config_dict = update_training_config(yml_conf, config_dict, instrument, key)

            if "GOES" in instrument and instrument_dict[instrument][key]["run_preprocess"]:
                yml_conf["input_dir"] = instrument_dict[instrument][key]["input_oc_dir"]
                goes_combine_and_clip_pipeline(yml_conf)                

            if instrument_dict[instrument][key]["run_preprocess_and_scale"]:
                run_data_prep(config_dict)             

            configs[instrument][key] = config_dict

            config_fname = build_config_fname_data_prep_and_train(config_dir, instrument, with_trop = trop)

            with open(config_fname, 'w') as fle:
               yaml.dump(config_dict, fle)

    return configs



def run_model_training(yml_conf, configs):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    for instrument in instrument_dict:
        for key in instrument_dict[instrument]:
            config_dict = configs[instrument][key]

            run_pretraining(config_dict) 
            run_tuning(config_dict)
            run_heir_training(config_dict)




