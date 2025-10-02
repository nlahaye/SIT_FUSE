import os

from sit_fuse.utils import read_yaml
from sit_fuse.preprocessing.colocate_and_resample import resample_or_fuse_data

from sit_fuse.pipelines.data_transform_and_prep.goes_combine_and_clip_constants import *

def update_config_goes_netcdf_to_gtiff(yml_conf, config):

    fdir = yml_conf["input_dir"]

    start_lon = yml_conf["start_lon"]
    end_lon = yml_conf["end_lon"]
    start_lat = yml_conf["start_lat"]
    end_lat = yml_conf["end_lat"]
 
    config["fusion"]["lon_bounds"] = [start_lon, end_lon]
    config["fusion"]["lat_bounds"] = [start_lat, end_lat]

   
    fnames = []
    #Find all files
    for root, dirs, files in os.walk(fdir):
        for fle in files:
            mtch = re.search(GOES_BASIC_RE, fle)
            if mtch:
                fnames.append(fle)

    #Sort based on (1) datetime and (2) channel
    #GOES filename-specific indices for start time and channel
    fnames = sorted(fnames, key = lambda x: (x[27:41], x[18:21]))

    #Check for sets that don't contain all 16 channels
    last_chan = -999
    chan = -999
    scene_files = []
    fname_sets = []
    out_files = []
    channels = np.zeros(GOES_NCHAN, dtype=np.int8)
    for i in range(len(fnames)):
        mtch = re.search(GOES_CHAN_RE, fnames[i])
        if mtch:
            chan = int(mtch.group(1))
            if channels[chan] > 0 or last_chan > chan:
                if np.sum(channels) == GOES_NCHAN:
                    fname_sets.append(copy.deepcopy(scene_files))
                    out_files.append(scene_files[0].replace(".nc", ".tif"))
                scene_files = []
                channels = np.zeros(GOES_NCHAN, dtype=np.int8)
            channels[chan] = 1
            scene_files.append(os.path.join(fdir, fnames[i]))
            last_chan = chan
    config["low_res"]["data"]["filenames"] = fname_sets
    config["low_res"]["data"]["geo_filenames"] = fname_sets
    config["output_files"] = out_files

    return config


def run_goes_combine_and_clip(yml_conf):

    config_dict = copy.deepcopy(YAML_TEMPLATE_GOES_NCDF_TO_GTFF)
    config_dict = update_config_goes_netcdf_to_gtiff(yml_conf, config_dict)
    resample_or_fuse_data(config_dict)

    return config_dict








