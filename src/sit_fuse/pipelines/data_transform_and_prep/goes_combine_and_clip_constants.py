

YAML_TEMPLATE_GOES_NCDF_TO_GTFF = {
"low_res" : {
 "data" : {
  "valid_min" : -100.0,
  "valid_max" : 99999999999,
  "reader_type" : "goes_netcdf",
  "reader_kwargs" : { "none" : "None"},
  "geo_reader_type" : "goes_netcdf_geo",
  "geo_reader_kwargs" : {
   "no_arg" : "no_arg" },
  "filenames" : [],
  "geo_filenames": [],
  "chan_dim" : 0,
  "geo_coord_dim" : 0,
  "geo_lat_index" : 0,
  "geo_lon_index" : 1,
 },
},

"high_res:" : {
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
 "description" : "Generating GOES GeoTiffs",
 "area_id" : "GOES",
 "projection_proj4" : {
  "proj" : "longlat",
  "datum" : "WGS84"},
 "final_resolution" : 0.018,
 "projection_units" : "degrees",
 "resample_radius" : 5000,
 "resample_n_neighbors" : 64,
 "resample_n_procs" : 10,
 "resample_epsilon" :  1.6,
 "use_bilinear" : True,
 "lon_bounds" : [0,0],
 "lat_bounds" : [0,0]
},

"output_files" : []

}



GOES_BASIC_RE = "OR_ABI.*\.nc"
GOES_NCHAN = 16
GOES_CHAN_RE = "C([0-1]\d_)"



