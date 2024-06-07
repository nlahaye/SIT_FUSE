"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

from collections import OrderedDict
from operator import itemgetter
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import argparse
import zarr
import shutil
import os
import datetime
import copy
import dbfread
from osgeo import gdal

from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func

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
        pprint(new_data_label_counts)
        pprint(init_data_label_counts)
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


def read_label_counts_dbfs(dbf_list):

    new_data_label_counts = {}
    init_data_label_counts = {}

    for i in range(len(dbf_list)):
        init_data_label_counts[i] = {'total' : 0} 
        for j in range(len(dbf_list[i])):
            if dbf_list[i][j] == "":
                continue
            for record in dbfread.DBF(dbf_list[i][j]):
                for key in record.keys():
                    try:
                        label = float(key)    
                        count = int(record[key])
                                                                          

                        if label not in new_data_label_counts.keys():
                            new_data_label_counts[label] = {i : count, 'total' : count}
                        elif i in new_data_label_counts[label].keys():
                             new_data_label_counts[label][i] = new_data_label_counts[label][i] + count
                             new_data_label_counts[label]['total'] = new_data_label_counts[label]['total'] + count
                        else:
                             new_data_label_counts[label][i] = count
                             new_data_label_counts[label]['total'] = new_data_label_counts[label]['total'] + count

                        if label not in init_data_label_counts[i].keys():
                            init_data_label_counts[i][label] = count
                        else:
                            init_data_label_counts[i][label] = init_data_label_counts[i][label] + count
                        init_data_label_counts[i]['total'] = init_data_label_counts[i]['total'] + count

                    except ValueError:
                        continue


    for key in init_data_label_counts.keys():
            init_data_label_counts[key] = OrderedDict(sorted(init_data_label_counts[key].items(), reverse = True, key=lambda item: item[1]))
    for key in new_data_label_counts.keys():
            new_data_label_counts[key] = OrderedDict(sorted(new_data_label_counts[key].items(), reverse = True, key=lambda item: item[1]))
  


    print("KEYS")
    pprint(init_data_label_counts)
    print("KEYS")
    pprint(new_data_label_counts)
    return new_data_label_counts, init_data_label_counts


def run_compare_dbf(dbf_list, percent_threshold):

    new_data_label_counts, init_data_label_counts = read_label_counts_dbfs(dbf_list)
        
    new_data_label_percentage = {}
    init_data_label_percentage = {}

    for key in new_data_label_counts.keys():
        new_data_label_percentage[key] = {}
        for key2 in new_data_label_counts[key].keys():
            new_data_label_percentage[key][key2] = float(new_data_label_counts[key][key2]) / float(new_data_label_counts[key]['total'])
                
    for key in init_data_label_counts.keys():
        init_data_label_percentage[key] = {}
        for key2 in init_data_label_counts[key].keys():
            init_data_label_percentage[key][key2] = float(init_data_label_counts[key][key2]) / float(init_data_label_counts[key]['total'])

    for key in init_data_label_percentage.keys():
            init_data_label_percentage[key] = OrderedDict(sorted(init_data_label_percentage[key].items(), reverse = True, key=lambda item: item[1]))
    for key in new_data_label_percentage.keys():
            new_data_label_percentage[key] = OrderedDict(sorted(new_data_label_percentage[key].items(), reverse = True, key=lambda item: item[1]))
 
 


    print("KEYS")
    pprint(init_data_label_percentage)
    print("KEYS")
    pprint(new_data_label_percentage)



    assignment = []
    uncertain = []
    for key in sorted(init_data_label_percentage.keys()):
        assignment.append([])
    assignment.append([])
    for key in new_data_label_percentage.keys():
        if key <= 0.0:
            continue
        #skip total
        assign = list(new_data_label_percentage[key].items())[1]
        if assign[0] == 'total':
            assign = list(new_data_label_percentage[key].items())[0]
        percentage = assign[1]
        index = assign[0]
        print(percentage)
        if percentage >= percent_threshold:
            assignment[index].append(key)
        else:
            assignment[-1].append(key)
            tmp = list(new_data_label_percentage[key].items())
            tmp.insert(0, key)
            uncertain.append(tmp)

    print("ASSIGNMENT")
    print(assignment)

    pprint(uncertain)

def compare_label_sets(new_data, init_data, mask_name, map_vals, no_retrieval_init=-1, 
    no_retrieval_new=-1, new_data_label_counts = None, init_data_label_counts  = None, glint = None):
 
        init_data = init_data[:new_data.shape[0],:new_data.shape[1]]
        new_data = new_data[:init_data.shape[0], :init_data.shape[1]]

        if new_data_label_counts is None:
                new_data_label_counts = {}
        if init_data_label_counts is None:
                init_data_label_counts = {}


        for i in range(new_data.shape[0]):
                for j in range(new_data.shape[1]):
                        if init_data[i,j] == no_retrieval_init or \
                            new_data[i,j] == no_retrieval_new or \
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


        print("KEYS")
        pprint(init_data_label_counts)
        print("KEYS")
        pprint(new_data_label_counts)
        return init_data, new_data, new_data_label_counts, init_data_label_counts


def plot_classifier_map(init_dat, new_dat, log_fname, total_data, total_mask,
    mask_name, map_vals, agree_new, agree_init, labels, gradient,
    grad_increase, gradient_local = None, no_retrieval_init=-1, no_retrieval_new=-1, glint = None):

        init_data = init_dat.ReadAsArray()
        new_data = new_dat.ReadAsArray()


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

        pprint(agree_new)


        print(init_data.shape, new_data.shape, "HERE SIZES")
        for i in range(new_data.shape[0]):
                for j in range(new_data.shape[1]):
                        if init_data[i,j] == no_retrieval_init or \
                            new_data[i,j] == no_retrieval_new or \
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



        print("CLASSES", np.unique(flat_data), np.unique(flat_predict), labels[mask_name])


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

        nx = new_data.shape[1]
        ny = new_data.shape[0]
        geoTransform = new_dat.GetGeoTransform()
        wkt = new_dat.GetProjection()
        print(wkt)
        fname = log_fname + ".DBN_DATA_AGREE.tif"
        print(fname, plotted_data.max())
        out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int32)
        out_ds.SetGeoTransform(geoTransform)
        out_ds.SetProjection(wkt)
        out_ds.GetRasterBand(1).WriteArray(plotted_data.astype(np.int32))
        out_ds.FlushCache()
        out_ds = None



        plt.imshow(plotted_data)
        plt.colorbar()
        plt.savefig(log_fname + ".DBN_DATA_AGREE.png")
        plt.clf()

        fname = log_fname + ".MASKED_DATA.tif"
        out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(geoTransform)
        out_ds.SetProjection(wkt)
        out_ds.GetRasterBand(1).WriteArray(plotted_data.astype(np.float32))
        out_ds.FlushCache()
        out_ds = None

        plt.imshow(plotted_data.astype(np.float32))
        plt.colorbar()
        plt.savefig(log_fname + ".MASKED_DATA.png")
        plt.clf()
        return plotted_data, init_data, gradient_local

def run_compare(init_input, new_input, class_order, log_fname, out_ext, clust_ext, no_retrieval_init, \
        no_retrieval_new, good_vals, map_vals, labels, gradient, grad_increase, class_mask_gen_func):


        print(new_input, init_input)

        for clust in range(len(new_input[0])):
                for dbn1 in range(0, len(init_input), len(new_input)):
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
                        for dbn in range(dbn1, dbn1+len(init_input)):
                                glint = None
                                for i in range(len(class_order)):

                                        if class_order[i] not in compare_new.keys():
                                                compare_new[class_order[i]] = None
                                                compare_init[class_order[i]] = None
                                                init_data[class_order[i]] = []
                                                new_data[class_order[i]] = []

                                        pprint(init_input)
                                        pprint(new_input)
                                        print(dbn, clust, class_order[i], len(new_input), len(init_input))
                                        if not os.path.exists(new_input[dbn][clust]) or not os.path.exists(init_input[dbn][class_order[i]]):
                                                print(new_input[dbn][clust], init_input[dbn][class_order[i]], 
                                                    os.path.exists(new_input[dbn][clust]), 
                                                    os.path.exists(init_input[dbn][class_order[i]]), " ERROR MISSING FILE")
                                                break
                                
                                        init_dat = gdal.Open(init_input[dbn][class_order[i]]).ReadAsArray()
                                        new_dat = gdal.Open(new_input[dbn][clust]).ReadAsArray()
                                        inds = np.where(init_dat > 0) #TODO generalize to multi-class
                                        init_dat[inds] = 1
                                        init_dat, out_dat, compare_new_single, compare_init_single = \
                                            compare_label_sets(new_dat, init_dat, \
                                                class_order[i], map_vals, no_retrieval_init[i], no_retrieval_new[i], \
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
                                total_mask = None
                                total_actual_mask = None
                                masks = {}
                                data = {}
                                glint = None
                                log_fname_fn = None
                                for i in range(len(class_order)):
                                        print(log_fname, class_order[i], i, dbn, out_ext[dbn], clust, clust_ext[clust])
                                        log_fname_fn = log_fname + "_" + class_order[i] + "_" + \
                                            out_ext[dbn] + "_" + clust_ext[clust] + ".txt"

                                        init_dat = gdal.Open(init_input[dbn][class_order[i]])
                                        new_dat = gdal.Open(new_input[dbn][clust])                                   
 
                                        dat, actualMask, gradient_local = plot_classifier_map(init_dat, 
                                            new_dat, log_fname_fn, total_mask, total_actual_mask, class_order[i], 
                                            map_vals, agree_new[class_order[i]], agree_init[class_order[i]], labels, grad, 
                                            grad_increase, gradient_local, no_retrieval_init[i], no_retrieval_new[i], glint)
                                        data[class_order[i]] = dat
                                        masks[class_order[i]] = actualMask
                                        if class_order[i] == "SunGlint":
                                                data[class_order[i]] = masks[class_order[i]]
                                                glint =  masks[class_order[i]]

                                        total_actual_mask = class_mask_gen_func(masks, good_vals, map_vals, class_order)
                                        total_mask = class_mask_gen_func(data, good_vals, map_vals, class_order)

                                #Should just use fuse_data for this
 
                                #genGeoTiff(total_actual_mask.astype(np.float32), geo_dat, log_fname_fn + "_TOTAL_ACTUAL_MASK" + ".tif")
                                #genGeoTiff(total_actual_mask.astype(np.int32), geo_dat, log_fname_fn + \
                                #     "_NO_GRADIENT_TOTAL_ACTUAL_MASK" + ".tif")
                                #genGeoTiff(total_mask.astype(np.float32), geo_dat, log_fname_fn + "_TOTAL_MASK" + ".tif")
                                #genGeoTiff(total_mask.astype(np.int32), geo_dat, log_fname_fn +"NO_GRADIENT_TOTAL_MASK" + ".tif")



                                #print(np.unique(total_actual_mask), np.unique(total_mask), labels["total_mask"])
                                cm = confusion_matrix(np.ravel(total_actual_mask.astype(np.int32)), \
                                    np.ravel(total_mask.astype(np.int32)), labels=labels["total_mask"])
                                acc = accuracy_score(np.ravel(total_actual_mask.astype(np.int32)), np.ravel(total_mask.astype(np.int32)))
                                balanced_acc = balanced_accuracy_score(np.ravel(total_actual_mask.astype(np.int32)), \
                                    np.ravel(total_mask.astype(np.int32)))
                                lbls.extend(np.ravel(total_mask.astype(np.int32)))
                                truth.extend(np.ravel(total_actual_mask.astype(np.int32)))
                                cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                                print(log_fname_fn)
                                with open(log_fname_fn, "a") as f:
                                        f.write("total_mask:")
                                        pprint(labels["total_mask"], f)
                                        pprint(cm, f)
                                        pprint(cm2, f)

                                        totalActSum = 0
                                        total_maskSum = 0
                                        for i in labels["total_mask"]:
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
                        cm = confusion_matrix(truth, lbls, labels=labels["total_mask"])
                        cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        pprint(cm)
                        pprint(cm2)
                        print("N =", np.count_nonzero(np.array(lbls) > -1))


def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 

    if yml_conf["dbf"]:
        run_compare_dbf(yml_conf["dbf_list"], yml_conf["dbf_percentage_thresh"])
 
    else:
        init_data = yml_conf["init_data"]
        new_data = yml_conf["new_data"]
        class_order = yml_conf["class_order"]
        log_fname = yml_conf["log_fname"]
        out_ext = yml_conf["out_ext"]
        clust_ext = yml_conf["clust_ext"]
        no_retrieval_init = yml_conf["no_retrieval_init"]
        no_retrieval_new = yml_conf["no_retrieval_new"]
        good_vals = yml_conf["good_vals"]
        map_vals = yml_conf["map_vals"]
        labels = yml_conf["labels"]
        gradient = yml_conf["gradient"]
        grad_increase = yml_conf["gradient_increase"]

        class_mask_gen_func = get_class_mask_func(yml_conf["class_mask_gen_func"])

        run_compare(init_data, new_data, class_order, log_fname, out_ext, clust_ext, no_retrieval_init, \
            no_retrieval_new, good_vals, map_vals, labels, gradient, grad_increase, class_mask_gen_func)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)





