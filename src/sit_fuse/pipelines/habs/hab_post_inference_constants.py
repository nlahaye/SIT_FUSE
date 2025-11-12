
YAML_TEMPLATE_CF_GTIFF = {
"low_res" : {
 "data" : {
  "valid_min" : -100.0,
  "valid_max" : 99999999999,
  "reader_type" : "zarr_to_numpy",
  "reader_kwargs" : { "none" : "None"},
  "geo_reader_type" : "zarr_to_numpy",
  "geo_reader_kwargs" : {
   "no_arg" : "no_arg" },
  "filenames" : [],
  "geo_filenames": [],
  "chan_dim" : 2,
  "geo_coord_dim" : 2,
  "geo_lat_index" : 0,
  "geo_lon_index" : 1,
 },
},

"high_res" : {
 "data" : {
  "valid_min" : 0.0,
  "valid_max" : 99999999999,
  "reader_type" : "zarr_to_numpy",
  "reader_kwargs" : {
   "no_arg" : "no_arg"},
  "geo_reader_type" : "zarr_to_numpy",
  "geo_reader_kwargs" : {
   "no_arg" : "no_arg"},
  "filenames" : [],
  "geo_filenames" : [],
  "chan_dim" : 0,
  "geo_coord_dim" : 2,
  "geo_lat_index" : 0,
  "geo_lon_index" : 1,
 },
},

"fusion": {
 "projection_id" : 4326,
 "description" : "Generating HAB GeoTiffs",
 "area_id" : "HAB Coastal",
 "projection_proj4" : {
  "proj" : "longlat",
  "datum" : "WGS84"},
 "final_resolution" : 0.055,
 "projection_units" : "degrees",
 "resample_radius" : 5000,
 "resample_n_neighbors" : 64,
 "resample_n_procs" : 10,
 "resample_epsilon" :  1.6,
 "use_bilinear" : False
},

"output_files" : []

}


YAML_TEMPLATE_MULTI_HIST = {
"xl_fname" : "",
"start_date" : "",
"end_date" : "",
"clusters" : "",

"clusters_dir" : "",
"radius_degrees" : [],
"global_max" : 10000000,
"ranges" : [0,1000,10000,100000,1000000,10000000, 100000000],
"input_file_type" : "",
"use_key" : ""
}


INSTRUMENT_PREFIX = {
"pace" : "PACE_OCI",
"s3a" : "S3A_OLCI",
"s3b" : "S3B_OLCI",
"jpss1" : "JPSS1_VIIRS",
"jpss2" : "JPSS2_VIIRS",
"snpp" : "SNPP_VIIRS",
"modis" : "AQUA_MODIS",
"goes18": "GOES18_ABI",
"troposif" : "SNPP_VIIRS",
}

NO_HEIR_ADD = "\.no_heir.*"

RE_STR = "\.\d{8}\.L3m\.DAY).*"
RE_STR_2 = "data_\d+clusters\.zarr"
RE_STR_2_PROBA = "data_proba.zarr"


RE_STR_DATE = ".*(\d{8}).*"


HAB_USE_KEYS =  {

"gulf_of_mexico": ['Karenia_Brevis'],
"california": ['Alexandrium_spp',
'Pseudo_nitzschia_delicatissima_group',
'Pseudo_nitzschia_seriata_group',
'Total_Phytoplankton']
}

USE_KEY_FNAME_MAP = {

"Karenia_Brevis" : "karenia_brevis_bloom", 
"Pseudo_nitzschia_seriata_group" : "pseudo_nitzschia_seriata_bloom",
"Pseudo_nitzschia_delicatissima_group" : "pseudo_nitzschia_delicatissima_bloom",
"Alexandrium_spp" : "alexandrium_bloom",
"Total_Phytoplankton" : "total_phytoplankton"
}


YAML_TEMPLATE_GTIFF = {
"gen_from_geotiffs" : True,

"data" : {
 "clust_reader_type" : "zarr_to_numpy",
 "reader_kwargs" : {
   "no_arg" : "no_arg",
 },
 "subset_inds" : [],
 "create_separate" : False,
 
 "gtiff_data" : [],
 "cluster_fnames" : [],
},
 
"context" : {
 "apply_context" : True,
 "clusters" : [],
 "background_class" : 1,
 "name" : "",
 "compare_truth" : False,
 "generate_union" : False
},
}



YAML_TEMPLATE_ZONAL_HIST = {
"data" : {
 "min_thresh": 0.0,
 "regrid": True,
 "zone_min": 0,
 "zone_max": 6,
 "multiclass": True,
 "clust_gtiffs" : [],
 "label_gtiffs" : [],
},

"output" : {
 "out_dir": "", #"/data/nlahaye/output/Learnergy/SNPP_VIIRS_Gulf_of_Mexico/"
 "class_name": ""
}
}


YAML_TEMPLATE_HEIR_CLASS_COMPARE = {
"dbf_list" : [[
""
]],
"dbf" : True,
"dbf_percentage_thresh" : 0.51

}

YAML_TEMPLATE_DAILY_MERGE = {

"input_paths" : [],
#["/data/nlahaye/output/Learnergy/JPSS1_VIIRS_Gulf_of_Mexico/", "/data/nlahaye/output/Learnergy//TROP_ONLY_Gulf_of_Mexico/", "/data/nlahaye/output/Learnergy/JPSS1_VIIRS_ONLY_Gulf_of_Mexico/"]
"fname_str" : "", #"DAY.karenia_brevis_bloom.tif"
"out_dir" : "", #"/data/nlahaye/output/MERGED_HAB_20250225/JPSS1_VIIRS/"
"num_classes" : 6,
"re_index" : 1,
 
"gen_daily" : True,
"gen_monthly" : False
}


USE_KEY_RE_INDEX = {
"Karenia_Brevis_SIF" : 0,
"Karenia_Brevis" : 1,
"Karenia_Brevis_no_heir": 2,
"Pseudo_nitzschia_seriata_group" : 3,
"Pseudo_nitzschia_delicatissima_group" : 4,
"Alexandrium_spp" : 5,
"Total_Phytoplankton": 6,
"Pseudo_nitzschia_seriata_group_no_heir" : 7,
"Pseudo_nitzschia_delicatissima_group_no_heir" : 8,
"Alexandrium_spp_no_heir" : 9,
"Total_Phytoplankton_no_heir": 10
}
