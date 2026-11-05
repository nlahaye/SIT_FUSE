
# Cap native thread pools (OpenMP/BLAS/MKL/joblib) before numpy/torch load
# when run as a script; run_* functions re-apply via apply_compute_limits.
if __name__ == "__main__":
    from sit_fuse.compute_resources import limit_env_from_argv
    limit_env_from_argv()

from sit_fuse.utils import read_yaml
from sit_fuse.compute_resources import apply_compute_limits, add_num_threads_arg, apply_num_threads_arg

from sit_fuse.preprocessing.colocate_and_resample import resample_or_fuse_data

from sit_fuse.postprocessing.multi_hist_insitu import run_multi_hist
from sit_fuse.postprocessing.generate_cluster_geotiffs import run_geotiff_gen
from sit_fuse.postprocessing.zonal_histogram import run_zonal_hist
from sit_fuse.postprocessing.class_compare import run_class_compare

from sit_fuse.pipelines.habs.hab_post_inference_constants import *
from sit_fuse.pipelines.habs.hab_post_inference_utils import run_context_free_geotiff_generation, run_context_assignment, run_geotiff_generation,\
run_multi_tier_zonal_histogram, run_multi_tier_class_compare, merge_class_sets, class_dict_from_confs, run_data_stream_merge, run_validation
   
import os
import yaml
import argparse
import pickle

def run_hab_post_inference_pipeline(yml_conf):

    # GeoTIFF generation, resampling, zonal histograms, validation (+ head export / UQ)
    apply_compute_limits(yml_conf)

    #Run 
    roi = yml_conf["roi"]
    species_run = HAB_USE_KEYS[roi]

    no_heir = True
    if "no_heir" in yml_conf:
        no_heir = yml_conf["no_heir"]

    #no_heir = False

    print("Running Context-Free Geotiff Generation")
    yml_conf = run_context_free_geotiff_generation(yml_conf) 

    # Optional encoder-side probabilistic outputs (hab_inference_pipeline):
    # head distributions (+ MC input noise, embedding OOD) on the same grid as
    # the context-free GeoTIFFs. Existing *.head_outputs.tif are kept, so this
    # is a no-op when hab_inference_pipeline already ran with run_head_export.
    if yml_conf.get("run_head_export", False):
        from sit_fuse.pipelines.habs.hab_inference_pipeline import run_head_output_export, validate_head_outputs
        print("Running Head Output Export")
        run_head_output_export(yml_conf)
        hx = yml_conf.get("head_export", {})
        if hx.get("validate", True):
            print("Validating head outputs against context-free GeoTIFFs")
            yml_conf["head_export_validation"] = validate_head_outputs(
                yml_conf, min_agreement=float(hx.get("min_joint_agreement", 0.99)))

    out_dir = yml_conf["final_product_dir"]
 
    if "validation_only" not in yml_conf or yml_conf["validation_only"] is False: 
        if not yml_conf["reuse_context"]:
            for run in species_run:
                print("Generating products for", run)

                print("Running Context Assignment")
                classes = run_context_assignment(yml_conf, run)

                print("Running Context-Assigned Geotiff Generation")
                run_geotiff_generation(yml_conf, classes, run, is_final = False)    
 
                if no_heir:

                    print("Running No_Heir data stream merge")
                    yml_conf = run_data_stream_merge(yml_conf, out_dir, run, no_heir = True)

                    print("Running Zonal Histogramming for Heir vs No_Heir Products")
                    run_multi_tier_zonal_histogram(yml_conf, run)

                    print("Running class comparison for Heir vs No_Heir Products")
                    iter2_classes = run_multi_tier_class_compare(yml_conf, run)

                    print("Merging class sets")
                    final_class_set = merge_class_sets(yml_conf, run, classes, iter2_classes)
    
                    print("Generating final products")
                    run_geotiff_generation(yml_conf, final_class_set, run, is_final = True)

        else:
            for run in species_run:
                print("Generating products for", run)

                conf_dict = yml_conf["reuse_configs"][run]
                classes = class_dict_from_confs(conf_dict)
 
                yml_conf["no_heir"] = False
                print("Running Context-Assigned Geotiff Generation")
                run_geotiff_generation(yml_conf, classes, run, is_final = True)    

        out_dir = yml_conf["final_product_dir"]
        print("Running final data stream merge")

        for run in species_run:
            yml_conf = run_data_stream_merge(yml_conf, out_dir, run, no_heir = False)

  
    #Splitting out loop for ease of visualization and to allow for validation-only workflows
    for run in species_run:
        validation_output = run_validation(yml_conf, run)
        if "validation" not in yml_conf:
            yml_conf["validation"] = {}
        yml_conf["validation"][run] = validation_output

    # Optional probabilistic / UQ products, evaluation and multi-sensor fusion
    if yml_conf.get("run_uq", False) and "uq" in yml_conf:
        from sit_fuse.pipelines.habs.uq.hab_uq_pipeline import run_hab_uq_pipeline
        stages = tuple(yml_conf["uq"].get("stages", ["train", "evaluate", "infer", "fuse"]))
        yml_conf["uq_summary"] = run_hab_uq_pipeline(yml_conf, stages)

    out_fname = os.path.join(yml_conf["final_product_dir"], "final_output")
    os.makedirs(out_fname, exist_ok=True)

    out_fname = os.path.join(out_fname, "output_vals.pkl")
    with open(out_fname, 'wb') as fle:
        pickle.dump(yml_conf, fle, protocol=pickle.HIGHEST_PROTOCOL)



#run_data_stream_merge(yml_conf, out_dir, species_run, no_heir = False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    add_num_threads_arg(parser)
    args = parser.parse_args()
    apply_num_threads_arg(args)

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_hab_post_inference_pipeline(yml_conf)







