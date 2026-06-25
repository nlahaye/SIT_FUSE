

YAML_TEMPLATE_TEMPO_NCDF_TO_GTFF = {
"low_res" : {
 "data" : {
  "fill_value": -9.96921e+36,
  "valid_min" : -9.0e+35,
  "valid_max" : 9.0e+36,
  "reader_type" : "tempo_netcdf",
  "reader_kwargs" : { "none" : "None"},
  "geo_reader_type" : "tempo_netcdf_geo",
  "geo_reader_kwargs" : {
   "no_arg" : "no_arg" },
  "filenames" : [],
  "geo_filenames": [],
  "chan_dim" : 2,
  "geo_coord_dim" : 0,
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
 "description" : "Generating TEMPO GeoTiffs",
 "area_id" : "TEMPO",
 "projection_proj4" : {
  "proj" : "longlat",
  "datum" : "WGS84"},
 "final_resolution" : 0.09,
 "projection_units" : "degrees",
 "resample_radius" : 50000,
 "resample_n_neighbors" : 128,
 "resample_n_procs" : 20,
 "resample_epsilon" :  1.6,
 "use_bilinear" : True,
 "lon_bounds" : [0,0],
 "lat_bounds" : [0,0]
},


"output_files" : []

}



TEMPO_BASIC_RE = "TEMPO_RAD_L1_V.*\.nc"



