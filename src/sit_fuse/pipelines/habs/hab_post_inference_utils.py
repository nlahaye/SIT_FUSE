
from sit_fuse.preprocessing.colocate_and_resample import resample_or_fuse_data

from sit_fuse.postprocessing.multi_hist_insitu import run_multi_hist
from sit_fuse.postprocessing.generate_cluster_geotiffs import run_geotiff_gen
from sit_fuse.postprocessing.zonal_histogram import run_zonal_hist
from sit_fuse.postprocessing.class_compare import run_class_compare
from sit_fuse.postprocessing.merge_datasets import run_merge
 
from sit_fuse.pipelines.habs.hab_post_inference_constants import *

import os
import yaml


def build_res(instrument):

    inst_prefx = "(" + INSTRUMENT_PREFIX[instrument]

    re_heir = inst_prefx + RE_STR + RE_STR_2
    re_no_heir = inst_prefx + RE_STR + NO_HEIR_ADD + RE_STR_2
    re_prob = inst_prefx + RE_STR + RE_STR_2_PROBA 
    re_prob_no_heir = inst_prefx + RE_STR + NO_HEIR_ADD + RE_STR_2_PROBA

    return re_heir, re_no_heir, re_prob, re_prob_no_heir


def build_config_fname_cf_gtiff_gen(config_dir, instrument, proba = False, no_heir = True, with_trop = False):

    config_fname = os.path.join(config_dir, "preprocessing", "colocate_and_resample", "fuse_" + instrument + "_oc")
    if with_trop:
        config_fname = config_fname + "_trop"
    if proba:
        config_fname = config_fname + "_proba"
    if no_heir:
        config_fname = config_fname + "_no_heir"

    config_fname = config_fname + ".yaml"
    return config_fname


def build_config_fname_multi_hist(config_dir, instrument, no_heir = True, with_trop = False, validation = False):

    config_fname = os.path.join(config_dir, "postprocessing", "multi_hsit_" + instrument + "_oc")
    if with_trop:
        config_fname = config_fname + "_trop"
    if no_heir:
        config_fname = config_fname + "_no_heir"
    if validation:
        config_fname = config_fname + "_validation"

    config_fname = config_fname + ".yaml"
    return config_fname


def build_config_fname_gtiff_gen(config_dir, instrument, no_heir = True, with_trop = False, is_final = False):

    config_fname = os.path.join(config_dir, "postprocessing", "geotiff_gen_" + instrument + "_oc")
    if with_trop:
        config_fname = config_fname + "_trop"
    if no_heir:
        config_fname = config_fname + "_no_heir"
    if is_final:
        config_fname = config_fname + "_final"

    config_fname = config_fname + ".yaml"
    return config_fname


def build_config_fname_zonal_hist(config_dir, instrument, with_trop = False):

    config_fname = os.path.join(config_dir, "postprocessing", "zonal_histogram_" + instrument + "_oc")
    if with_trop:
        config_fname = config_fname + "_trop"

    config_fname = config_fname + ".yaml"
    return config_fname

def build_config_fname_class_comp(config_dir, instrument, with_trop = False):

    config_fname = os.path.join(config_dir, "postprocessing", "class_compare_" + instrument + "_oc")
    if with_trop:
        config_fname = config_fname + "_trop"

    config_fname = config_fname + ".yaml"
    return config_fname

 
def build_config_fname_data_stream_merge(config_dir, instrument, daily = True):

    config_fname = os.path.join(config_dir, "postprocessing", "merge_datasets_" + instrument + "_oc")
    if daily:
        config_fname = config_fname + "_daily"
    else:
        config_fname = config_fname + "_monthly"

    config_fname = config_fname + ".yaml"
    return config_fname




def update_config_cf_gtiff_gen(fdir, config, instrument, geo_zarr_path, proba = False, no_heir = True):

    re_heir, re_no_heir, re_prob, re_prob_no_heir = build_res(instrument)

    for root, dirs, files in os.walk(fdir):
        for fle in files:
            mtch = re.search(re_heir, fle)
            mtch_no_heir = re.search(re_no_heir, fle)
            if mtch_no_heir and no_heir:
                config["low_res"]["data"]["filenames"].append(os.path.join(root, fle))
                config["low_res"]["data"]["geo_filenames"].append(geo_zarr_path) 
                out_path = os.path.join(root, mtch_no_heir.group(1)) + "no_heir.tif"
                config["output_files"].append(out_path)
                #This is the same file every time, can make this more configurable if needed
            elif mtch_no_heir is None and mtch:
                config["low_res"]["data"]["filenames"].append(os.path.join(root, fle))
                config["low_res"]["data"]["geo_filenames"].append(geo_zarr_path) 
                out_path = os.path.join(root, mtch_no_heir.group(1)) + ".tif"
                config["output_files"].append(out_path)

            if proba:
                mtch_prob = re.search(re_prob, fle)
                mtch_prob_no_heir = re.search(re_prob_no_heir, fle)
                if mtch_prob_no_heir and no_heir:
                    config["low_res"]["data"]["filenames"].append(os.path.join(root, fle))
                    config["low_res"]["data"]["geo_filenames"].append(geo_zarr_path) 
                    out_path = os.path.join(root, mtch_no_heir.group(1)) + "no_heir.proba.tif"
                    config["output_files"].append(out_path)
                elif mtch_prob_no_heir is None and mtch_prob:
                    config["low_res"]["data"]["filenames"].append(os.path.join(root, fle))
                    config["low_res"]["data"]["geo_filenames"].append(geo_zarr_path)
                    out_path = os.path.join(root, mtch_no_heir.group(1)) + ".proba.tif"
                    config["output_files"].append(out_path)

    #config["low_res"]["data"]["filenames"] = sorted(config["low_res"]["data"]["filenames"]) 
    #config["output_files"] = sorted(config["output_files"])

    return config


 
def update_config_multi_hist(out_dir, config_dict, yml_conf, instrument, species_key, no_heir = True, validation = False):

    config_dict["xl_fname"] = yml_conf["insitu_file"]
    config_dict["radius_degrees"] = yml_conf["insitu_matchup_circle_max_radius"]
    config_dict["start_date"] = yml_conf["start_date"]
    config_dict["end_date"] = yml_conf["end_date"]
    config_dict["clusters_dir"] = out_dir
    config_dict["use_key"] = species_key

    config_dict["clusters"] = yml_conf["clusters"]
    config_dict["input_file_type"] = INSTRUMENT_PREFIX[instrument]

    if no_heir:
        config_dict["input_file_type"] =  config_dict["input_file_type"] + "no_heir"
    elif validation:
        config_dict["input_file_type"] =  'daily'

    
    return config_dict


def update_config_gtiff_gen(config_dict, gtiff_list, classes, species_run, no_heir=True):


    full_file_list = np.array(copy.deepcopy(gtiff_list))
    del_inds = []
    for i in range(len(full_file_list)):
        if not no_heir and "no_heir" in full_file_list[i]:
            del_inds.append(i)
        elif no_heir and "no_heir" not in full_file_list[i]:
            del_inds.append(i)
        elif "prob" in full_file_list[i]:
            del_inds.append(i)

    full_file_list = np.delete(full_file_list, del_inds)

    config_dict["data"]["gtiff_data"] = full_file_list
    config_dict["data"]["cluster_fnames"] = full_file_list
    
    config_dict["context"]["name"] = USE_KEY_FNAME_MAP[species_run]
    config_dict["context"]["clusters"] = classes
 
    return config_dict


def update_config_zonal_hist(config_dict, gtiff_list, species_run, no_heir = True):

    full_file_list = np.array(copy.deepcopy(gtiff_list))
    del_inds = []
    for i in range(len(full_file_list)):
        if "no_heir" in full_file_list[i] or "prob" in full_file_list[i]:
            del_inds.append(i)
    full_file_list = np.delete(full_file_list, del_inds)

    clust_gtiffs = []
    label_gtiffs = []
    for i in range(len(full_file_list)):
        mtch = re.search(full_file_list[i])
        if mtch is None:
            continue
 
        dtestr = mtch.group(1)
        if no_heir:
            dtestr = dtestr + "_no_heir."
        else:
            dtestr = dtestr + "_DAY."

        dtestr = dtestr + USE_KEY_FNAME_MAP[species_run] + ".tif"
        drpth = os.path.dirname(yml_conf[instrument]["merged_products"]) 
        label_pth = os.path.join(drpth, dtestr)
 

        label_gtiffs.append(label_pth)

    label_gtiff_final = []
    for i in range(len(YAML_TEMPLATE_MULTI_HIST["ranges"])):
        label_gtiff_final.append(label_gtiffs) 

    config_dict["data"]["clust_gtiffs"] = full_file_list
    config_dict["data"]["label_gtiffs"] = label_gtiff_final
    
    config_dict["output"]["out_dir"] = os.path.dirname(full_file_list[0])
    config_dict["output"]["class_name"] = "tiered_hab_" + USE_KEY_FNAME_MAP[species_run]
 
    return config_dict

def update_config_class_compare(config_dict, out_dir, species_run):

    config_dict["dbf_list"] = [[os.path.join(out_dir, "tiered_hab_" +  USE_KEY_FNAME_MAP[species_run] + "_hist_dict.pkl")]]
    
    return config_dict


def update_config_data_stream_merge(yml_conf, config_dict, out_dir, species_run, instrument, daily = True, no_heir = False):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    input_paths = []

    append = INSTRUMENT_PREFIX[instrument]
    if no_heir:
        append = append + "_no_heir"

    dr = os.path.join(out_dir, append)
    config_dict["out_dir"] = dr

    os.makedirs(dr, exist_ok = True) 

    if "no_trop" in yml_conf[instrument]:
        input_paths.append(os.path.dirname(yml_conf[instrument]["trop"][0]))
    else:
        input_paths.append("")

    if "troposif" in yml_conf:
        input_paths.append(os.path.dirname(yml_conf["troposif"]["trop"][0]))
    else:
        input_paths.append("")
 
        input_paths.append(os.path.dirname(yml_conf[instrument]["no_trop"][0]))

    config_dict["input_paths"] = input_paths
 
    if daily:
        config_dict["fname_str"] = "DAY." + USE_KEY_FNAME_MAP[species_run] + ".tif"
        config_dict["gen_monthly"] = False
        config_dict["gen_daily"] = True
    else:
        USE_KEY_FNAME_MAP[species_run]
        config_dict["gen_monthly"] = True
        config_dict["gen_daily"] = False
        config_dict["max_dqi"] = len(input_paths)-1
        config_dict["dirname"] = config_dict["out_dir"]
        config_dict["max_class"] = config_dict["num_classes"] -1

    config_dict["out_dir"] = out_dir
    config_dict["num_classes"] = len( yml_conf['ranges'])-1 

    if no_heir:
        species_run = species_run + "_no_heir"

    if "sif_finalday" in yml_conf["troposif"]["trop"][0]:
        species_run = species_run + "_SIF"

    config_dict["re_index"] = USE_KEY_RE_INDEX[species_run]

    return config_dict
 
def run_data_stream_merge(yml_conf, out_dir, species_run, no_heir = False):


    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    for instrument in instrument_dict:
        if instrument == "troposif":
            continue
 
        #Daily Merge
        #Generate config
        config_dict = copy.deepcopy(YAML_TEMPLATE_DAILY_MERGE)
        config_dict = update_config_data_stream_merge(yml_conf, config_dict, out_dir, species_run, instrument, daily = True, no_heir = no_heir)
 
        #Dump to file
        config_fname = build_config_fname_data_stream_merge(config_dir, instrument, daily = True)

        products, dqi = run_merge(config_fname)
        products = sorted(products)
        dqi = sorted(dqi)
 
        #Monthly Merge
        #Generate config
        config_dict = copy.deepcopy(YAML_TEMPLATE_DAILY_MERGE)
        config_dict = update_config_data_stream_merge(yml_conf, config_dict, out_dir, species_run, instrument, daily = False, no_heir = no_heir)

        #Dump to file
        config_fname = build_config_fname_data_stream_merge(config_dir, instrument, daily = False)
 
        monthly_products, monthly_dqi = run_merge(config_fname)
        monthly_products = sorted(monthly_products)
        monthly_dqi = sorted(monthly_dqi)

        key = "heir"
        if no_heir:
            key = "no_" + key

        if "merged_products" not in yml_conf[instrument]:
            yml_conf[instrument]["merged_products"] = {}

        if "merged_dqi" not in yml_conf[instrument]:
            yml_conf[instrument]["merged_dqi"] = {}

        if "monthly_merged_dqi" not in yml_conf[instrument]:
            yml_conf[instrument]["monthly_merged_dqi"] = {}

        if "monthly_merged_products" not in yml_conf[instrument]:
            yml_conf[instrument]["monthly_merged_products"] = {}

        yml_conf[instrument]["merged_products"][key] = products
        yml_conf[instrument]["merged_dqi"][key] = dqi

        yml_conf[instrument]["monthly_merged_dqi"][key] = monthly_dqi
        yml_conf[instrument]["monthly_merged_products"][key] = monthly_products
        
    return yml_conf


def run_multi_tier_class_compare(yml_conf, species_run):
 
    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]


    iter2_classes = {}
    for instrument in instrument_dict:
        if instrument not in iter2_classes:
            iter2_classes[instrument] = {}
        for key in instrument_dict[instrument]:
            if key not in iter2_classes[instrument]:
                iter2_classes[instrument][key] = {}

            trop = True
            if "no" in key: #Key is trop no_trop
                trop = False


            out_dir = os.path.dirname(yml_conf[instrument][key]["cf_gtiffs"][0])

            #Generate config
            config_dict = copy.deepcopy(YAML_TEMPLATE_HEIR_CLASS_COMPARE)
            config_dict = update_config_class_compare(config_dict, out_dir, species_run)

            #Dump to file
            config_fname = build_config_fname_class_comp(config_dir, instrument, with_trop)
            with open(config_fname, 'w') as fle:
                yaml.dump(config_dict, fle)

            assignments, uncertain = run_class_compare(config_dict)
            iter2_classes[instrument][key]["classes"] = assignments 
            iter2_classes[instrument][key]["uncertain"] = uncertain 
            

    return iter2_classes


def run_multi_tier_zonal_histogram(yml_conf, species_run):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    for instrument in instrument_dict:
        for key in instrument_dict[instrument]:

            trop = True
            if "no" in key: #Key is trop no_trop
                trop = False

            config_dict = copy.deepcopy(YAML_TEMPLATE_ZONAL_HIST)

            #Generate config
            config_dict = update_config_zonal_hist(config_dict, yml_conf[instrument][key]["cf_gtiffs"], species_run, no_heir=True)

            #Dump to file
            config_fname = build_config_fname_zonal_hist(config_dir, instrument, with_trop = trop)
            with open(config_fname, 'w') as fle:
                yaml.dump(config_dict, fle)

            run_zonal_hist(yml_conf)


 
def run_geotiff_generation(yml_conf, classes, species_run, is_final = False):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    no_heir = True
    prob = False

    if "no_heir" in yml_conf:
        no_heir = yml_conf["no_heir"]


    for instrument in instrument_dict:
        for key in instrument_dict[instrument]:
            trop = True
            if "no" in key: #Key is trop no_trop
                trop = False

            config_dict = copy.deepcopy(YAML_TEMPLATE_GTIFF)
            
            #Generate config
            config_dict = update_config_gtiff_gen(config_dict, yml_conf[instrument][key]["cf_gtiffs"], classes[instrument][key]["heir"]["classes"], species_run, no_heir = False)
            
            #Dump to file
            config_fname = build_config_fname_gtiff_gen(config_dir, instrument, no_heir = False, with_trop = trop, is_final = is_final) 
            with open(config_fname, 'w') as fle:
                yaml.dump(config_dict, fle)          

            run_geotiff_gen(config_dict)

            if no_heir:
                config_dict = copy.deepcopy(YAML_TEMPLATE_GTIFF)

                #Generate config
                config_dict = update_config_gtiff_gen(config_dict, yml_conf[instrument][key]["cf_gtiffs"], classes[instrument][key]["no_heir"]["classes"], species_run, no_heir = True)
  
                #Dump to file
                config_fname = build_config_fname_gtiff_gen(config_dir, instrument, no_heir = True, with_trop = trop, is_final = is_final)
                with open(config_fname, 'w') as fle:
                    yaml.dump(config_dict, fle)

                run_geotiff_gen(config_dict)
                


def run_context_free_geotiff_generation(yml_conf):

    instrument_dict = yml_conf["instruments"]
    geo_zarr = yml_conf["geo_zarr"]
    config_dir = yml_conf["config_dir"]

    no_heir = True
    prob = False

    if "no_heir" in yml_conf:
        no_heir = yml_conf["no_heir"]
    if "prob" in yml_conf:
        prob = yml_conf["prob"]
    #command line inputs: instrument, no_heir, prob, HAB geo zarr, config dir

    for instrument in instrument_dict:
        for key in instrument_dict[instrument]:
            trop = True
            if "no" in key: #Key is trop no_trop
                trop = False
            config_dict = copy.deepcopy(YAML_TEMPLATE_CF_GTIFF)

            #Generate config
            config_dict = update_config_cf_gtiff_gen(instrument_dict[instrument][key]["out_dir"], config_dict, instrument, geo_zarr, proba = prob, no_heir = False)
            
            #Dump to file
            config_fname = build_config_fname_cf_gtiff_gen(config_dir, instrument, proba = prob, no_heir = False, with_trop = trop)       
            with open(config_fname, 'w') as fle:
                yaml.dump(config_dict, fle)
          
            resample_or_fuse_data(config_dict) 
        
            yml_conf[instrument][key]["cf_gtiffs"] = config_dict["output_files"]

            if no_heir:
                config_dict = copy.deepcopy(YAML_TEMPLATE_CF_GTIFF)

                #Generate config
                config_dict = update_config_cf_gtiff_gen(instrument_dict[instrument][key]["out_dir"], config_dict, instrument, geo_zarr, proba = prob, no_heir = True)
             
                #Dump to file
                config_fname = build_config_fname_cf_gtiff_gen(config_dir, instrument, proba = prob, no_heir = True, with_trop = trop)   
                with open(config_fname, 'w') as fle:
                    yaml.dump(config_dict, fle)

                resample_or_fuse_data(config_dict)

                yml_conf[instrument][key]["cf_gtiffs"].extend(config_dict["output_files"])
 


    return yml_conf

 
def run_context_assignment(yml_conf, species_run):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    no_heir = True
    if "no_heir" in yml_conf:
        no_heir = yml_conf["no_heir"]


    classes = {}
    for instrument in instrument_dict:
        if instrument not in classes:
            classes[instrument] = {}
        for key in instrument_dict[instrument]:
            if key not in classes[instrument]:
                classes[instrument][key] = {}

            trop = True
            if "no" in key: #Key is trop no_trop
                trop = False

            #Generate config
            config_dict = copy.deepcopy(YAML_TEMPLATE_MULTI_HIST)       
            config_dict = update_config_multi_hist(instrument_dict[instrument][key]["out_dir"], config_dict, yml_conf, instrument, species_run, no_heir = False)
 
            config_fname = build_config_fname_multi_hist(config_dir, instrument, no_heir = False, with_trop = trop)
            classes_heir, _ = run_multi_hist(config_dict)
            classes[instrument][key]["heir"]["classes"] = classes_heir

            with open(config_fname, 'w') as fle:
                yaml.dump(config_dict, fle)

            classes_no_heir = []
            if no_heir:
                config_dict = copy.deepcopy(YAML_TEMPLATE_MULTI_HIST)
                config_dict = update_config_multi_hist(instrument_dict[instrument][key]["out_dir"], config_dict, yml_conf, instrument, species_run, no_heir = True)
                classes_no_heir, _ = run_multi_hist(config_dict) 
                classes[instrument][key]["no_heir"]["classes"] = classes_no_heir

                config_fname = build_config_fname_multi_hist(config_dir, instrument, no_heir = True, with_trop = trop)
                with open(config_fname, 'w') as fle:
                    yaml.dump(config_dict, fle)

    return classes


def run_validation(yml_conf, species_run):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]

    classes = {}
    for instrument in instrument_dict:
        if instrument not in classes:
            classes[instrument] = {}
        for key in instrument_dict[instrument]:
            if key not in classes[instrument]:
                classes[instrument][key] = {}

            trop = True
            if "no" in key: #Key is trop no_trop
                trop = False

            #Generate config
            config_dict = copy.deepcopy(YAML_TEMPLATE_MULTI_HIST)
            out_dir = yml_conf["final_product_dir"]

            append = INSTRUMENT_PREFIX[instrument]
            out_dir = os.path.join(out_dir, append)

            config_dict = update_config_multi_hist(out_dir, config_dict, yml_conf, instrument, species_run,\
                 no_heir = False, validation = True)

            config_fname = build_config_fname_multi_hist(config_dir, instrument, no_heir = False, with_trop = trop, validation=True)
            hists, _ = run_multi_hist(config_dict)
            classes[instrument][key]["validation"] = hists

            with open(config_fname, 'w') as fle:
                yaml.dump(config_dict, fle)


    return classes




def merge_class_sets(yml_conf, species_run, classes, iter2_classes):

    instrument_dict = yml_conf["instruments"]
    config_dir = yml_conf["config_dir"]
 
    final_class_set = {}
    for instrument in instrument_dict:
        if instrument not in final_class_set:
            final_class_set[instrument] = {}
        for key in instrument_dict[instrument]:
            if key not in final_class_set[instrument]:
                final_class_set[instrument][key] = {}
 
            arr_tmp = [[] for x in range(0,(len( yml_conf['ranges'])-1))]
            arr_init = classes[instrument][key]["heir"]["classes"]
            arr_merge = iter2_classes[instrument][key]["classes"] 
 
            count = 0
            for c in range(1, len(arr_merge)):
                count = count + len(arr_merge)
            if count < 1:
                final_class_set[instrument][key] = classes[instrument][key]["heir"]["classes"]  
 
            for i in range(len(arr_init)):
                for j in range(len(arr_merge[i])):
                    ind = -1
                    for k in range(len(arr_init)):
                        if arr_merge[i][j] in arr_init[k]:
                            ind = k
                    if ind < i:
                        arr_tmp[i].append(labels[n][i][j])

            for i in range(len(arr_init)):
                for j in range(len(arr_init[i])):
                    ind = -1
                    for k in range(len(arr_init)):
                        if arr_init[i][j] in arr_tmp[k]:
                            ind = k
                    if ind < 0:
                        arr_tmp[i].append(arr_init[i][j])



            for i in range(len(arr_tmp)):
                arr_tmp[i] = sorted(arr_tmp[i]) 

            final_class_set[instrument][key]["classes"] = arr_tmp

    full_class_set = {"init_classes" : classes, "iter2_classes" : iter2_classes, "final_classes" : final_class_set}

    with open(os.path.join(config_dir, "class_set_dict.yaml"), 'w') as fle:
        yaml.dump(full_class_set, fle)

    return final_class_set

             
def class_dict_from_confs(yml_conf):

    instrument_dict = yml_conf["instruments"]

    classes = {}
    for instrument in instrument_dict:
        classes[instrument] = {}
        for key in instrument_dict[instrument]:
            classes[instrument][key] = {}
            conf = read_yaml(instrument_dict[instrument][key])
            classes[instrument][key]["heir"]["classes"] = conf["context"]["clusters"]

    return classes






