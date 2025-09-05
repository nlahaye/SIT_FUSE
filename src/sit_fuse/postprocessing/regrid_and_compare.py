import salem
from salem import get_demo_file, open_xr_dataset
from sklearn.metrics import classification_report, confusion_matrix
from pyresample import area_config, bilinear, geometry, data_reduce, create_area_def, kd_tree
from pyresample.utils.rasterio import get_area_def_from_raster

import skill_metrics as sm

from scipy.spatial import distance

from skimage.metrics import structural_similarity, variation_of_information, adapted_rand_error, contingency_table
 
from osgeo import gdal, osr

import argparse
 
import os
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from osgeo import gdal, osr
import copy

from sit_fuse.utils import numpy_to_torch, read_yaml, get_read_func

def is_real_num(value):
    return (~np.isnan(value) and ~np.isinf(value))

def weighted_mean(data, counts):

    values = 0
    full_count = 0
    for i in range(len(data)):
        if is_real_num(data[i]) and is_real_num(counts[i]):
            values = values + (data[i]*counts[i])
            full_count = full_count + counts[i]

    if full_count > 0:
        return (values / full_count)
    else:
        return np.nan

def f1_score(tp, tn, fp, fn):

    if (float(tp) + float(fn)) == 0.0:
        recall = 0.0
    else:
        recall = float(tp) / (float(tp) + float(fn))

    if (float(tp) + float(fp)) == 0.0:
        precision = 0.0
    else:
        precision = float(tp) / (float(tp) + float(fp))


    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = (2*precision*recall) / (precision + recall)

    return precision, recall, f1

def dice_sim(labels, truth):
 
    dice = distance.dice(labels, truth)
    return dice

def diff_map(tmp2, tmp, sfmd, fle_ext):
        out_dat = np.squeeze(tmp - tmp2)
        print(tmp.min(), tmp2.min(), tmp.max(), tmp2.max(), tmp.shape, tmp2.shape, "COMPARE")
        print(np.nanmean(out_dat), np.nanmin(out_dat), np.nanmax(out_dat))

        dat = gdal.Open(sfmd)

        dat_tmp = dat.ReadAsArray()
        nx = out_dat.shape[1]
        ny = out_dat.shape[0]
        geoTransform = dat.GetGeoTransform()
        gt2 = [geoTransform[0], geoTransform[1], geoTransform[2], geoTransform[3], geoTransform[4], geoTransform[5]]
        gt2[1] = gt2[1] * dat_tmp.shape[1] / nx
        gt2[5] = gt2[5] * dat_tmp.shape[0] / ny

        metadata = dat.GetMetadata()
        wkt = dat.GetProjection()
        gcpcount = dat.GetGCPCount()
        gcp = None
        gcpproj = None

        if gcpcount > 0:
            gcp = dat.GetGCPs()
            gcpproj = dat.GetGCPProjection()
            dat.FlushCache()
            dat = None

        fbase = os.path.splitext(sfmd)[0] + "_DIFF_" + fle_ext + ".tif"
        print(out_dat.shape, np.nanmean(out_dat), np.nanmin(out_dat), np.nanmax(out_dat), nx, ny, "HERE")
        print(geoTransform, gt2)
        fname = fbase + ".tif"
        print(fname)
        out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
        out_ds.SetGeoTransform(gt2)
        out_ds.SetMetadata(metadata)
        out_ds.SetProjection(wkt)
        if gcpcount > 0:
            out_ds.SetGCPs(gcp, gcpproj)
        out_ds.GetRasterBand(1).WriteArray(out_dat)
        out_ds.FlushCache()
        out_ds = None



def calculate_iou(gt_mask, pred_mask):
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask)>0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou
 
def regrid_and_compare(config):

    sit_fuse_map_fnames = config["sit_fuse_maps"]
    truth_map_fnames = config["truth_maps"]


    other_map_fnames = None
    if "other_maps" in config.keys():
        other_map_fnames = config["other_maps"]

    ##truth_full = None
    ##sf_full = None
    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0

    tp2 = 0.0
    fp2 = 0.0
    tn2 = 0.0
    fn2 = 0.0
    ##ssim_full = 0.0

    tpb1 = 0.0
    fpb1 = 0.0
    tnb1 = 0.0
    fnb1 = 0.0

    tpb2 = 0.0
    fpb2 = 0.0
    tnb2 = 0.0
    fnb2 = 0.0

    errors = []
    precisions = []
    recalls = []
    splitss = []
    mergess = []
    ssims = []
    ssims2 = []
    dices = []
    dices2 = []
    pix_cnts = []
    bsss = []

    errors_b1 = []
    precisions_b1 = []
    recalls_b1 = []
    ssims_b1 = []

    errors_b2 = []
    precisions_b2 = []
    recalls_b2 = []
    ssims_b2 = []


    img_count = 0
    for i in range(len(sit_fuse_map_fnames)):
 
        map_fle = truth_map_fnames[i]
        sfmd = sit_fuse_map_fnames[i]

        other = None
        if other_map_fnames is not None:
            other = other_map_fnames[i]

        print(sfmd, map_fle)

        sfm = gdal.Open(sfmd).ReadAsArray()
        if sfm.max() < 0.0:
            continue


        print("Opening", map_fle)
        gm = gdal.Open(map_fle).ReadAsArray()
        #gm = gm.clip(min=0, max=1)

        print(gm.min(), gm.max())
        print("Regridding")

        area_def = get_area_def_from_raster(map_fle)
        final_area_def = get_area_def_from_raster(sfmd)
        print(area_def, final_area_def, gm.min(), gm.max(), gm.mean(), "HERE")
        gm_on_sfm = np.squeeze(kd_tree.resample_nearest(area_def, gm, final_area_def, radius_of_influence=500, fill_value = 0))
        print(gm_on_sfm.min(), gm_on_sfm.max(), gm_on_sfm.mean(), "HERE")
        #gm_on_sfm[np.where(gm_on_sfm  > 1)] = 0
        gm_on_sfm[np.where(gm_on_sfm  > 0)] = 1
        gm_on_sfm[np.where(gm_on_sfm  < 0)] = 0

        print(gm_on_sfm.min(), gm_on_sfm.max(), gm_on_sfm.mean(), "HERE")


        oth_on_sfm = None
        if other is not None:
            oth = gdal.Open(other).ReadAsArray()
            area_def = get_area_def_from_raster(other)
            print(area_def, final_area_def, oth.min(), oth.max(), oth.mean(), "HERE")
            oth_on_sfm = np.squeeze(kd_tree.resample_nearest(area_def, oth, final_area_def, radius_of_influence=500, fill_value = 0))
            print(oth_on_sfm.min(), oth_on_sfm.max(), oth_on_sfm.mean(), "HERE")
            #oth_on_sfm[np.where(oth_on_sfm  > 1)] = 0
            oth_on_sfm[np.where(oth_on_sfm  > 0)] = 1
            oth_on_sfm[np.where(oth_on_sfm  < 0)] = 0


        print(gm.min(), gm.max(), gm_on_sfm.min(), gm_on_sfm.max(), sfm.min(), sfm.max())


        tmp = np.array(sfm).astype(np.float32)

        tmp[np.where(tmp  > 0)] = 1
        tmp[np.where(tmp  < 0)] = 0
  
        tmp2 = np.array(gm_on_sfm).astype(np.float32)
        #tmp2_2 = cv2.dilate(tmp2.copy(), None, iterations=2)
        #tmp2_2 = cv2.erode(tmp2_2.copy(), None, iterations=1)
        #tmp2[np.where((tmp2_2 > 0) & (tmp2 == 0))] = 0.75
        #tmp2_3 = cv2.dilate(tmp2_2.copy(), None, iterations=1)
        #tmp2[np.where((tmp2_3 < 0.5) & (tmp2_3 > 0.0)& (tmp2 == 0))] = 0.25
        tmp2_neg = 1 - tmp2


        tmp2[np.where(np.isnan(tmp2))] = 0

        #tmp2 is truth, tmp is SIT-FUSE, tmp3 is 'other'
        diff_map(tmp2, tmp, sfmd, "TRUTH")

        baseline1 = np.zeros(tmp2.shape)
        baseline2 = np.ones(tmp2.shape)

        
        del sfm
        del gm_on_sfm
        del gm

        tmp3 = None
        if oth_on_sfm is not None:
            tmp3 = np.array(oth_on_sfm).astype(np.float32)
            tmp3[np.where(np.isnan(tmp3))] = 0
            diff_map(tmp2, tmp3, sfmd, "PRE_TRUTH")
            del oth
            del oth_on_sfm


        #ssim_b1 = structural_similarity(tmp2.astype(np.int32), baseline1.astype(np.int32))
        #ssim_b2 = structural_similarity(tmp2.astype(np.int32), baseline2.astype(np.int32))

        win_size = min(tmp2.shape[0], tmp2.shape[1])
        dice = np.nan
        if win_size % 2 == 0:
            win_size = win_size -1
        dice = dice_sim(tmp2.flatten(), tmp.flatten())
        ssim = structural_similarity(tmp2.astype(np.float32), tmp.astype(np.float32), data_range=1, gaussian_weights=False, win_size=win_size)
        pix_cnt = np.prod(tmp2.shape)

        ssim2 = np.nan
        dice2 = np.nan
        if tmp3 is not None:
            ssim2 = structural_similarity(tmp2.astype(np.float32), tmp3.astype(np.float32), data_range=1, gaussian_weights=False, win_size=win_size)
            dice2 = dice_sim(tmp2.flatten(), tmp3.flatten())
        print("SSIM", ssim)
        print("Pixel Count", pix_cnt)

        #ignore = [-1] #0

        tpb1 = tpb1 + np.sum(tmp2[np.where((tmp2 > 0) & (baseline1 > 0))])
        tnb1 = tnb1 + np.sum(tmp2[np.where((tmp2_neg < 1) & (tmp2_neg >= 0) & (baseline1 < 1))]) + len(np.where((tmp2 == 0) & (baseline1 == 0))[0]) 
        fpb1 = fpb1 + np.sum(tmp2[np.where((tmp2_neg < 1) & (tmp2_neg >= 0) & (baseline1 > 0))]) + len(np.where((tmp2 == 0) & (baseline1 > 0))[0]) 
        fnb1 = fnb1 + np.sum(tmp2[np.where((tmp2 > 0) & (baseline1 < 1))])
 
        tpb2 = tpb2 + np.sum(tmp2[np.where((tmp2 > 0) & (baseline2 > 0))]) 
        tnb2 = tnb2 + np.sum(tmp2[np.where((tmp2_neg < 1) & (tmp2_neg >= 0) & (baseline2 < 1))]) + len(np.where((tmp2 == 0) & (baseline2 == 0))[0])
        fpb2 = fpb2 + np.sum(tmp2[np.where((tmp2_neg < 1) & (tmp2_neg >= 0) & (baseline2 > 0))]) + len(np.where((tmp2 == 0) & (baseline2 > 0))[0])
        fnb2 = fnb2 + np.sum(tmp2[np.where((tmp2 > 0) & (baseline2 < 1))])


        #t1_b1 = contingency_table(tmp2.astype(np.int32), baseline1.astype(np.int32), ignore_labels=ignore, normalize=False)
        #t1_b2 = contingency_table(tmp2.astype(np.int32), baseline2.astype(np.int32), ignore_labels=ignore, normalize=False)

        #error_b1, precision_b1, recall_b1 = adapted_rand_error(tmp2.astype(np.int32), baseline1.astype(np.int32), ignore_labels=ignore, table=t1_b1)
        #error_b2, precision_b2, recall_b2 = adapted_rand_error(tmp2.astype(np.int32), baseline2.astype(np.int32), ignore_labels=ignore, table=t1_b2)


        #t1 = contingency_table(tmp2.astype(np.int32), tmp.astype(np.int32), ignore_labels=ignore, normalize=False)
  
        #error, precision, recall = adapted_rand_error(tmp2.astype(np.int32), tmp.astype(np.int32), ignore_labels=ignore, table=t1)
        #splits, merges = variation_of_information(tmp2.astype(np.int32), tmp.astype(np.int32), ignore_labels=ignore, table=t1)
          
 
        bss = np.nan
        if tmp3 is not None:
            #bss = sm.skill_score_brier(tmp,tmp3,tmp2)
            tp2 = tp2 + np.sum(tmp2[np.where((tmp2 > 0) & (tmp3 > 0))])
            tn2 = tn2 + np.sum(tmp2[np.where((tmp2_neg < 1) & (tmp2_neg >= 0) & (tmp3 < 1))]) + len(np.where((tmp2 == 0) & (tmp3 == 0))[0])
            fp2 = fp2 + np.sum(tmp2[np.where((tmp2_neg < 1) & (tmp2_neg >= 0) & (tmp3 > 0))]) + len(np.where((tmp2 == 0) & (tmp3 > 0))[0])
            fn2 = fn2 + np.sum(tmp2[np.where((tmp2 > 0) & (tmp3 < 1))])
 
        #errors.append(error)
        #precisions.append(precision)
        #recalls.append(recall)
        #splitss.append(splits)
        #mergess.append(merges)
        ssims.append(ssim)
        pix_cnts.append(pix_cnt)
        ssims2.append(ssim2)
        dices.append(dice)
        dices2.append(dice2)
        #bsss.append(bss)
     
        #errors_b1.append(error_b1)
        #precisions_b1.append(precision_b1)
        #recalls_b1.append(recall_b1)
        #ssims_b1.append(ssim_b1)
 
        #errors_b2.append(error_b2)
        #precisions_b2.append(precision_b2)
        #recalls_b2.append(recall_b2)
        #ssims_b2.append(ssim_b2)

        ##report = classification_report(tmp2.ravel(), tmp.ravel(), labels=[0,1])
        ##iou = calculate_iou(tmp2.ravel(), tmp.ravel()) 
        ##cm = confusion_matrix(tmp2.ravel(), tmp.ravel(), labels=[0,1])
        #report = classification_report(tmp[finite_points], tmp2[finite_points], labels=[0,1])
        ##if sf_full is None:
        ##    sf_full = tmp.ravel()
        ##    truth_full = tmp2.ravel()
        ##else:
        ##    sf_full = np.concatenate((sf_full, tmp.ravel()))
        ##    truth_full = np.concatenate((truth_full, tmp2.ravel()))
        tp = tp + np.sum(tmp2[np.where((tmp2 > 0) & (tmp > 0))])
        tn = tn + np.sum(tmp2[np.where((tmp2_neg < 1) & (tmp2_neg >= 0) & (tmp < 1))]) + len(np.where((tmp2 == 0) & (tmp == 0))[0])
        fp = fp + np.sum(tmp2[np.where((tmp2_neg < 1) & (tmp2_neg >= 0) & (tmp > 0))]) + len(np.where((tmp2 == 0) & (tmp > 0))[0])
        fn = fn + np.sum(tmp2[np.where((tmp2 > 0) & (tmp < 1))])
        ##ssim_full = ssim_full + ssim
        ##print(sf_full.shape, truth_full.shape, "LENGTHS")
        ##print(report)
        ##print(cm)
        ##print("IOU:", iou)
 
        ##print("TP", tp, "FP", fp, "FN", fn, "TN", tn, "SSIM", ssim)
        #print("SSIM", ssim)
        #print("Adapted Rand Error, Precision, Recall", error, precision, recall)
        #print("variation of information", splits, merges)
        ##del cm
        ##del report
        ##img_count = img_count + 1

    precision, recall, f1 = f1_score(tp, tn, fp, fn)

    cm_full = np.zeros((2,2))
    cm_full[0,0] = tn
    cm_full[1,0] = fp
    cm_full[0,1] = fn
    cm_full[1,1] = tp


    print("SIT-FUSE Comparison")
    print("Precision", precision)
    print("Recall", recall)
    print("F1", f1)
    print(cm_full)



    if other_map_fnames is not None:
        print("OTHER Comparison")
        precision, recall, f1 = f1_score(tp2, tn2, fp2, fn2)
        cm_full = np.zeros((2,2))
        cm_full[0,0] = tn2
        cm_full[1,0] = fp2
        cm_full[0,1] = fn2
        cm_full[1,1] = tp2
        print("Precision", precision)
        print("Recall", recall)
        print("F1", f1)
        print(cm_full)
 

    ##ssim_full = ssim_full / img_count
    #print("Precision", precisions)
    #print("Recall:", recalls)
    #print("Error:", errors)
    #print("BSS", bsss)
    print("SSIM", ssims)
    print("SSIM2", ssims2)
    #print("Precision:", np.nanmean(precisions), "Recall:", np.nanmean(recalls), "Error:", np.nanmean(errors))
    #print("BSS", np.nanmean(bsss))
    print("SSIM Full", np.nanmean(ssims))
    print("SSIM2 Full", np.nanmean(ssims2))

    print("SSIM Full Weighted", weighted_mean(ssims, pix_cnts))
    print("SSIM2 Full Weighted", weighted_mean(ssims2, pix_cnts))

    print("Dice", np.nanmean(dice))
    print("Dice_2", np.nanmean(dice2))

    print("Dice Weighted", weighted_mean(dices, pix_cnts))
    print("Dice_2 Weighted", weighted_mean(dices2, pix_cnts))
 
    print("N Pixels", sum(pix_cnts))


    print("BASELINES")

    precision, recall, f1 = f1_score(tpb1, tnb1, fpb1, fnb1)
    cm_full = np.zeros((2,2))
    cm_full[0,0] = tnb1
    cm_full[1,0] = fpb1
    cm_full[0,1] = fnb1
    cm_full[1,1] = tpb1
    print("Precision", precision)
    print("Recall", recall)
    print("F1", f1)
    print(cm_full)

    precision, recall, f1 = f1_score(tpb2, tnb2, fpb2, fnb2)
    cm_full = np.zeros((2,2))
    cm_full[0,0] = tnb2
    cm_full[1,0] = fpb2
    cm_full[0,1] = fnb2
    cm_full[1,1] = tpb2
    print("Precision", precision)
    print("Recall", recall)
    print("F1", f1)
    print(cm_full)


    #print("Precision", precisions_b1)
    #print("Recall:", recalls_b1)
    #print("Error:", errors_b1)
    #print("SSIM", ssims_b1)
    #print("Precision:", np.nanmean(precisions_b1), "Recall:", np.nanmean(recalls_b1), "Error:", np.nanmean(errors_b1))
    #print("SSIM Full", np.nanmean(ssims_b1))
 
    #print("Precision", precisions_b2)
    #print("Recall:", recalls_b2)
    #print("Error:", errors_b2)
    #print("SSIM", ssims_b2)
    #print("Precision:", np.nanmean(precisions_b2), "Recall:", np.nanmean(recalls_b2), "Error:", np.nanmean(errors_b2))
    #print("SSIM Full", np.nanmean(ssims_b2))

    ##print(cm_full)
  
    ##full_report = classification_report(truth_full, sf_full, labels=[0,1])
    ##cm_full = confusion_matrix(truth_full, sf_full, labels=[0,1])
    ##iou_full = calculate_iou(truth_full, sf_full)
    ##print(full_report)
    ##print(cm_full)
    ##print("IOU:", iou_full)

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    regrid_and_compare(yml_conf)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)


