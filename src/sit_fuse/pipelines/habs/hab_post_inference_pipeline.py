
from sit_fuse.utils import read_yaml

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


def run_hab_post_inference_pipeline(yml_conf):

    #Run 
    roi = yml_conf["roi"]
    species_run = HAB_USE_KEYS[roi]

    no_heir = True
    if "no_heir" in yml_conf:
        no_heir = yml_conf["no_heir"]

    print("Running Context-Free Geotiff Generation")
    yml_conf = run_context_free_geotiff_generation(yml_conf) 
 
    if not yml_conf["reuse_context"]:
        for run in species_run:
            print("Generating products for", run)

            print("Running Context Assignment")
            classes = run_context_assignment(yml_conf, run)

            print("Running Context-Assigned Geotiff Generation")
            run_geotiff_generation(yml_conf, classes, run, is_final = False)    
 
            if no_heir:
                out_dir = yml_conf["final_product_dir"]

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

                print("Running final data stream merge")
                yml_conf = run_data_stream_merge(yml_conf, out_dir, run, no_heir = False)
    else:
        for run in species_run:
            print("Generating products for", run)

            conf_dict = yml_conf["reuse_configs"][run]
            classes = class_dict_from_confs(conf_dict)
 
            yml_conf["no_heir"] = False
            print("Running Context-Assigned Geotiff Generation")
            run_geotiff_generation(yml_conf, classes, run, is_final = True)    

 
    
    for run in species_run:
        validation_output = run_validation(yml_conf, run)
        yml_conf["validation"][run] = validation_output

    out_fname = os.path.join(yml_conf["final_product_dir"], "final_output")
    os.makedirs(out_fname, exists_ok=True)

    out_fname = os.path.join(out_fname, "output_vals.yaml")
    with open(out_fname, 'w') as fle:
        yaml.dump(yml_conf, fle)



#run_data_stream_merge(yml_conf, out_dir, species_run, no_heir = False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()

    #Translate config to dictionary 
    yml_conf = read_yaml(args.yaml)

    run_hab_post_inference_pipeline(yml_conf)







