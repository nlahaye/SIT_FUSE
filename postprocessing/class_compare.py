"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""


import zarr
import shutil
import os
import datetime

def calc_class_stats(new_data_label_counts, init_data_label_counts):

        new_data_stat = {"total": 0}
        for key in new_data_label_counts.keys():
                new_data_stat[key] = {}
                new_data_stat["total"] = new_data_stat["total"] + new_data_label_counts[key]["total"]
                for key2 in new_data_label_counts[key].keys():
                        new_data_stat[key][key2] = new_data_label_counts[key][key2] / new_data_label_counts[key]["total"]
        init_data_stat = {"total": 0}
        for key in init_data_label_counts.keys():
                init_data_stat[key] = {}
                init_data_stat["total"] = init_data_stat["total"] + init_data_label_counts[key]["total"]
                for key2 in init_data_label_counts[key].keys():
                        init_data_stat[key][key2] = init_data_label_counts[key][key2] / init_data_label_counts[key]["total"]

        label_agree_new_data = {}
        for key in new_data_label_counts.keys():
                lst = []
                for key2 in init_data_label_counts.keys():
                        if key in init_data_label_counts[key2]:
                                lst.append([key2, init_data_label_counts[key2][key]])
                lst = sorted(lst, key = itemgetter(1), reverse=True)
                label_agree_new_data[key] =  lst

        label_agree_init_data = {}
        for key in init_data_label_counts.keys():
                lst = []
                for key2 in new_data_label_counts.keys():
                        if key in new_data_label_counts[key2]:
                                lst.append([key2, new_data_label_counts[key2][key]])
                lst = sorted(lst, key = itemgetter(1), reverse=True)
                label_agree_init_data[key] =  lst

        pprint(new_data_label_counts.keys())
        pprint(init_data_label_counts.keys())
        pprint(new_data_stat)
        pprint(init_data_stat)
        return new_data_stat, init_data_stat, label_agree_new_data, label_agree_init_data




def class_mask_gen_misr_svm(masks, good_vals, map_vals, key_order): #, refMaskAero):
    mask = None
    #refMaskAeroInt = refMaskAero.astype(np.int32)
    for key in key_order:
        if key in masks.keys():
            compMaskInt = masks[key].astype(np.int32)
            if mask is None:
                mask = np.zeros(masks[key].shape) - 1
                maskTrack = np.zeros(masks[key].shape) - 1
            for i in range(len(good_vals[key])):
                print(maskTrack.shape, compMaskInt.shape, key, "HERE SHAPE ISSUE")
                if key == "Total":
                    extraInds = np.where(mask.astype(np.int32) ==good_vals[key][i])
                    extra = 0
                    if len(extraInds[0]) > 0:
                        extra = max(mask[extraInds]) % 1
                        if good_vals[key][i] == 1:
                            extra = extra / 10.0    
                    inds = np.where(((maskTrack == -1) | (mask == -1)) & (compMaskInt == good_vals[key][i]))
                    mask[inds] = map_vals[key][good_vals[key][i]]
                    mask[inds] = mask[inds] + (masks[key][inds] % 1)
                    maskTrack[inds] = compMaskInt[inds]
                    mask[inds] = mask[inds] + extra    
                elif key == "Smoke" or key == "Dust":
                    extraInds = np.where(mask.astype(np.int32) == 0)
                    extra = 0
                    if len(extraInds[0]) > 0:
                        extra = (max(mask[extraInds]) % 1) / 10.0
                    inds = np.where((((maskTrack > 1) & (compMaskInt == 1)) | 
                                                       ((maskTrack > 2) & (compMaskInt == 2)) | 
                                                       ((maskTrack > 3) & (compMaskInt == 3)) | 
                                                       ((maskTrack == -1) & (compMaskInt == good_vals[key][i]))))
                    mask[inds] = map_vals[key][good_vals[key][i]]
                    mask[inds] = mask[inds] + (masks[key][inds] % 1)
                    maskTrack[inds] = compMaskInt[inds]
                    if map_vals[key][good_vals[key][i]] == 0:
                        mask[inds] = mask[inds] + extra
                else:
                    inds = np.where(((maskTrack > 1) & (compMaskInt == 1)) | 
                                                       ((maskTrack > 2) & (compMaskInt == 2)) | 
                                                       ((maskTrack > 3) & (compMaskInt == 3)) | 
                                                       ((maskTrack == -1) & (compMaskInt == good_vals[key][i])))

                    indsBad = np.where((maskTrack == good_vals[key][i]) & (compMaskInt == good_vals[key][i]))

                    mask[inds] = map_vals[key][good_vals[key][i]]
                    mask[inds] = mask[inds] + (masks[key][inds] % 1)
                    maskTrack[inds] = compMaskInt[inds]
                    mask[indsBad] = -1

    return mask


def class_mask_gen_modis(masks, good_vals, map_vals, key_order):
    mask = None
    for key in key_order:
        if key in masks.keys():
            compMaskInt = masks[key].astype(np.int32)
            if mask is None:
                mask = np.zeros(masks[key].shape) - 1
            for i in range(len(good_vals[key])):
                inds = None
                if key == "Aerosol":
                    sunGlInt = masks["SunGlint"].astype(np.int32)
                    mask[np.where((compMaskInt == good_vals[key][i]))] = -1
                    inds = np.where((mask == -1) & (compMaskInt == good_vals[key][i]) & (sunGlInt == good_vals["SunGlint"][i]))
 
                else:
                    inds = np.where((mask == -1) & (compMaskInt == good_vals[key][i]))

                mask[inds] = map_vals[key][good_vals[key][i]]
                mask[inds] = mask[inds] + (masks[key][inds] % 1)

    return mask



def class_mask_gen_basic(masks, good_vals, map_vals, key_order):
    mask = None
    for key in key_order:
        if key in masks.keys():
            if mask is None:
                mask = np.zeros(masks[key].shape) - 1
            for i in range(len(good_vals[key])):
                mask[np.where((mask == -1) & (masks[key] == good_vals[key][i]))] = map_vals[key][good_vals[key][i]]

    return mask



def get_class_mask_func(key):

    if key == "misr_svm":
        return class_mask_gen_misr_svm
    elif key == "modis":
        return class_mask_gen_modis
    else:
        return class_mask_gen_basic


def compare_label_sets(new_data, init_data, mask_name, map_vals, no_retrieval=-1, 
    new_data_label_counts = None, init_data_label_counts  = None, glint = None):

        init_data = init_data[:new_data.shape[0],:new_data.shape[1]]
        new_data = new_data[:init_data.shape[0], :init_data.shape[1]]

        if new_data_label_counts is None:
                new_data_label_counts = {}
        if init_data_label_counts is None:
                init_data_label_counts = {}

        for i in range(new_data.shape[0]):
                for j in range(new_data.shape[1]):
                        if init_data[i,j] == no_retrieval or \
                            new_data[i,j] == NO_RETRIEVAL_DBN or \
                                (mask_name == "Aerosol" and glint is not None and glint[i,j] == 0):
                                    init_data[i,j] = -1
                                    new_data[i,j] = -1
                                    continue


                        if init_data[i,j] in init_data_label_counts.keys():
                                if new_data[i,j] in init_data_label_counts[init_data[i,j]].keys():
                                        init_data_label_counts[init_data[i,j]][new_data[i,j]] = \
                                            init_data_label_counts[init_data[i,j]][new_data[i,j]] + 1
                                else:
                                        init_data_label_counts[init_data[i,j]][new_data[i,j]] = 1
                                init_data_label_counts[init_data[i,j]]["total"] = \
                                    init_data_label_counts[init_data[i,j]]["total"] + 1
                        else:
                                init_data_label_counts[init_data[i,j]] = {new_data[i,j] : 1, "total": 1}



                        if new_data[i,j] in new_data_label_counts.keys():
                                if init_data[i,j] in new_data_label_counts[new_data[i,j]].keys():
                                        new_data_label_counts[new_data[i,j]][init_data[i,j]] = \
                                            new_data_label_counts[new_data[i,j]][init_data[i,j]] + 1
                                else:
                                        new_data_label_counts[new_data[i,j]][init_data[i,j]] = 1
                                new_data_label_counts[new_data[i,j]]["total"] = \
                                    new_data_label_counts[new_data[i,j]]["total"] + 1
                        else:
                                new_data_label_counts[new_data[i,j]] = {init_data[i,j]: 1, "total": 1}

        return init_data, new_data, new_data_label_counts, init_data_label_counts


def plot_classifier_map(init_data, new_data, log_fname, total_data, total_mask,
    mask_name, map_vals, agree_new, agree_init, labels, gradient,
    grad_increase, gradient_local = None, no_retrieval=-1, glint = None):

        if total_mask is None:
                total_mask = np.zeros(new_data.shape) -1
        if total_data is None:
                total_data = np.zeros(new_data.shape) -1



        plotted_data = np.zeros(init_data.shape) - 1
        #print(plotted_data.shape)
        flat_data = []
        flat_predict = []

        if gradient_local is None:
                gradient_local = {}


        for i in range(new_data.shape[0]):
                for j in range(new_data.shape[1]):
                        if init_data[i,j] == no_retrieval or \
                            new_data[i,j] == NO_RETRIEVAL_DBN or \
                            (total_data is not None and total_data[i,j] > -1) or \
                            (mask_name == "Aerosol" and glint is not None and glint[i,j] == 0):
                                #plotted_data[i,j] = total_data[i,j]
                                continue


                        if agree_new[new_data[i,j]][0][0] in map_vals[mask_name].keys():
                                if agree_new[new_data[i,j]][0][0] in gradient_local.keys():
                                        if new_data[i,j] not in gradient_local[agree_new[new_data[i,j]][0][0]].keys():
                                                gradient_local[agree_new[new_data[i,j]][0][0]][new_data[i,j]] = \
                                                    agree_new[new_data[i,j]][0][0] + gradient[mask_name][agree_new[new_data[i,j]][0][0]]
                                                gradient[mask_name][agree_new[new_data[i,j]][0][0]] = \
                                                    gradient[mask_name][agree_new[new_data[i,j]][0][0]] + grad_increase
                                else:
                                        gradient_local[agree_new[new_data[i,j]][0][0]] = \
                                            {new_data[i,j] : agree_new[new_data[i,j]][0][0] + \
                                                gradient[mask_name][agree_new[new_data[i,j]][0][0]]}
                                        gradient[mask_name][agree_new[new_data[i,j]][0][0]] = \
                                            gradient[mask_name][agree_new[new_data[i,j]][0][0]] + grad_increase


                                flat_data.append(int(init_data[i,j]))
                                flat_predict.append(int(agree_new[new_data[i,j]][0][0]))
                                plotted_data[i,j] = gradient_local[agree_new[new_data[i,j]][0][0]][new_data[i,j]]

                        else:
                                plotted_data[i,j] = -1

                        #flat_data.append(int(init_data[i,j]))
                        #flat_predict.append(int(agree_new[new_data[i,j]][0][0]))
                        #if agree_new[new_data[i,j]][0][0] in map_vals[mask_name].keys():
                        #       plotted_data[i,j] = agree_new[new_data[i,j]][0][0]
                        #else:
                        #       plotted_data[i,j] = -1 



        #print(np.unique(flat_data), np.unique(flat_predict), labels[mask_name])


        with open(log_fname, "a") as f:
                pprint(gradient, f)


        if np.unique(flat_data).shape[0] > 0:
                cm = confusion_matrix(flat_data, flat_predict, labels = labels[mask_name])
                cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                #print(mask_name)
                with open(log_fname, "w") as f:
                        f.write(mask_name + ":")
                        pprint(labels[mask_name], f)
                        pprint(cm, f)
                        pprint(cm2, f)
                        f.write("\n\n")
        else:
                with open(log_fname, "a") as f:
                        f.write(mask_name + ": NO LABELS FOUND\n\n")

        plt.imshow(datPlt)
        plt.colorbar()
        plt.savefig(log_fname + ".DBN_DATA_AGREE.png")
        plt.clf()
        plt.imshow(datPlt.astype(np.float32))
        plt.colorbar()
        plt.savefig(log_fname + ".MASKED_DATA.png")
        plt.clf()
        return plotted_data, init_data, gradient_local

def run_compare(init_data, new_data, class_order, log_fname, out_ext, clust_ext, no_retrieval_init, \
        no_retrieval_new, good_vals, map_vals, labels, gradient, grad_increase, class_mask_gen_func):

        for clust in range(len(new_data[0])):
                for dbn1 in range(0, len(init_data), len(init_data)):
                        compare_new = {}
                        compare_init = {}
                        dstats = {}
                        mStats = {}
                        agree_new = {}
                        agree_init = {}
                        init_data = {}
                        new_data = {}
                        gradient_local = {}
                        grad = copy.deepcopy(gradient)
                        print("Comparing...", out_ext[dbn1])
                        for dbn in range(dbn1, dbn1+len(init_data)):
                                glint = None
                                for i in range(len(class_order)):

                                        if class_order[i] not in compare_new.keys():
                                                compare_new[class_order[i]] = None
                                                compare_init[class_order[i]] = None
                                                init_data[class_order[i]] = []
                                                new_data[class_order[i]] = []

                                        if not os.path.exists(new_data[dbn][clust]) or not os.path.exists(init_data[dbn][class_order[i]]):
                                                print(new_data[dbn][clust], init_data[dbn][class_order[i]], 
                                                    os.path.exists(new_data[dbn][clust]), 
                                                    os.path.exists(init_data[dbn][class_order[i]]), " ERROR MISSING FILE")
                                                break
                                        init_dat, out_dat, compare_new_single, compare_init_single = \
                                            compareClassifiers(new_data[dbn][clust], init_data[dbn][class_order[i]], \
                                                class_order[i], MAP_VALS, no_retrieval[i], \
                                                compare_new[class_order[i]],  compare_init[class_order[i]], glint)
                                        init_data[class_order[i]].append(init_dat)
                                        new_data[class_order[i]].append(out_dat)
                                        compare_new[class_order[i]] = compare_new_single
                                        compare_init[class_order[i]] = compare_init_single
                                        if class_order[i] == "SunGlint":
                                                glint =  init_dat

                        print("Generating Stats...")
                        for i in range(len(class_order)):
                                dStat, mStat, agreeD, agreeM = calc_class_stats(compare_new[class_order[i]], \
                                    compare_init[class_order[i]])
                                dstats[class_order[i]] = dStat
                                mStats[class_order[i]] = mStat
                                agree_new[class_order[i]] = agreeD
                                agree_init[class_order[i]] = agreeM

                        print("Mapping and Plotting...")
                        lbls = []
                        truth = []
                        for ind in range(len(init_data[class_order[0]])):
                                dbn = dbn1 + ind
                                geo_dat = geo_data[dbn]
                                total_mask = None
                                total_actual_mask = None
                                masks = {}
                                data = {}
                                glint = None
                                log_fname_fn = None
                                for i in range(len(class_order)):
                                        log_fname_fn = log_fname + "_" + class_order[i] + "_" + \
                                            out_ext[dbn] + "_" + clust_ext[clust] + ".txt"
                                        dat, actualMask, gradient_local = plot_classifier_map(init_data[class_order[i]][ind], 
                                            new_data[class_order[i]][ind], log_fname_fn, total_mask, total_actual_mask, class_order[i], 
                                            MAP_VALS, agree_new[class_order[i]], agree_init[class_order[i]], labels, grad, 
                                            grad_increase, gradient_local, no_retrieval[i], glint)
                                        data[class_order[i]] = dat
                                        masks[class_order[i]] = actualMask
                                        if class_order[i] == "SunGlint":
                                                data[class_order[i]] = masks[class_order[i]]
                                                glint =  masks[class_order[i]]

                                        total_actual_mask = class_mask_gen_funck(masks, good_vals, map_vals, class_order)
                                        total_mask = class_mask_gen_func(data, good_vals, map_vals, class_order)

                                #Should just use fuse_data for this
 
                                #genGeoTiff(total_actual_mask.astype(np.float32), geo_dat, log_fname_fn + "_TOTAL_ACTUAL_MASK" + ".tif")
                                #genGeoTiff(total_actual_mask.astype(np.int32), geo_dat, log_fname_fn + \
                                #     "_NO_GRADIENT_TOTAL_ACTUAL_MASK" + ".tif")
                                #genGeoTiff(total_mask.astype(np.float32), geo_dat, log_fname_fn + "_TOTAL_MASK" + ".tif")
                                #genGeoTiff(total_mask.astype(np.int32), geo_dat, log_fname_fn +"NO_GRADIENT_TOTAL_MASK" + ".tif")



                                #print(np.unique(total_actual_mask), np.unique(total_mask), labels["TotalMask"])
                                cm = confusion_matrix(np.ravel(total_actual_mask.astype(np.int32)), \
                                    np.ravel(total_mask.astype(np.int32)), labels=labels["TotalMask"])
                                acc = accuracy_score(np.ravel(total_actual_mask.astype(np.int32)), np.ravel(total_mask.astype(np.int32)))
                                balanced_acc = balanced_accuracy_score(np.ravel(total_actual_mask.astype(np.int32)), \
                                    np.ravel(total_mask.astype(np.int32)))
                                lbls.extend(np.ravel(total_mask.astype(np.int32)))
                                truth.extend(np.ravel(total_actual_mask.astype(np.int32)))
                                cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                                print(log_fname_fn)
                                with open(log_fname_fn, "a") as f:
                                        f.write("TotalMask:")
                                        pprint(labels["TotalMask"], f)
                                        pprint(cm, f)
                                        pprint(cm2, f)

                                        totalActSum = 0
                                        total_maskSum = 0
                                        for i in labels["TotalMask"]:
                                                totalActSum = totalActSum + (total_actual_mask.astype(np.int32) == i).sum()
                                                total_maskSum = total_maskSum + (total_mask.astype(np.int32) == i).sum()
                                                f.write("LABEL:" + " " + str(i) + " COUNT: " + \
                                                    str((total_actual_mask.astype(np.int32) == i).sum()) + " (ACTUAL DATA)\n")
                                                f.write("LABEL: " + str(i) + " COUNT: " + \
                                                    str((total_mask.astype(np.int32) == i).sum()) + " (DBN DATA)\n")
                                                f.write("N ACTUAL = " + str(totalActSum) + "\n")
                                                f.write("N MASK = " + str(total_maskSum) + "\n")
                                                f.write("Accuracy = " + str(acc) + " Balanced Accuracy = " + str(balanced_acc) + "\n")
                                zarr.save(log_fname_fn + "_TOTAL_MASK" + ".zarr", total_mask)
                                zarr.save(log_fname_fn + "_TOTAL_ACTUAL_MASK" + ".zarr", total_actual_mask)
                        print(out_ext[dbn1] + " TOTAL ACCURACY: " + str(accuracy_score(truth, lbls)) +  \
                            " BALANCED ACC: ", str(balanced_accuracy_score(truth, lbls)), "\n")
                        cm = confusion_matrix(truth, lbls, labels=labels["TotalMask"])
                        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        pprint(cm)
                        pprint(cm2)
                        print("N =", np.count_nonzero(np.array(lbls) > -1))


def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    init_data = conf["init_data"]
    new_data = conf["new_data"]
    class_order = conf["class_order"]
    log_fname = conf["log_fname"]
    out_ext = conf["out_ext"]
    clust_ext = conf["clust_ext"]
    no_retrieval_init = conf["no_retrieval_init"]
    no_retrieval_new = conf["no_retrieval_new"]
    good_vals = conf["good_vals"]
    map_vals = conf["map_vals"]
    labels = conf["labels"]
    gradient = conf["gradient"]
    grad_increase = conf["gradient_increase"]

    class_mask_gen_func = get_class_mask_gen_func(yml_conf["class_mask_gen_func"])

    run_compare(init_data, new_data, class_order, log_fname, out_ext, clust_ext, no_retrieval_init, \
        no_retrieval_new, good_vals, map_vals, labels, gradient, grad_increase, class_mask_gen_func)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)





