import numpy as np
import matplotlib
matplotlib.use('agg')
from pprint import pprint
from sklearn.metrics import confusion_matrix
from osgeo import gdal
import argparse

from utils import numpy_to_torch, read_yaml, get_read_func

def generate_cluster_gtiffs(data_reader, data_reader_kwargs, subset_inds,
    cluster_data, gtiff_data, apply_context, context_clusters, context_name, compare, create_seperate):

	read_func = get_read_func(data_reader)

	total = []
	totalTruth = []
	for p in range(len(cluster_data)):
		print(cluster_data[p])
		print(gtiff_data[p])
               
		dbnDat1 = read_func(cluster_data[p], **data_reader_kwargs).astype(np.int32)
		print(dbnDat1.min(), dbnDat1.max())

		dat = gdal.Open(gtiff_data[p])
		imgData = dat.ReadAsArray()
		print(p)
		print(dbnDat1.shape, imgData.shape)
		nx = imgData.shape[1]
		ny = imgData.shape[0]
		geoTransform = dat.GetGeoTransform()
		wkt = dat.GetProjection()
			
		classes = np.unique(dbnDat1)
		print(int(classes.max() - classes.min() + 2))
		outDat = np.zeros(imgData.shape, dtype=np.int32) - 1
		if len(subset_inds[p]) > 0:
			outDat[subset_inds[0]:subset_inds[1],subset_inds[2]:subset_inds[3]] = dbnDat1
		else:
			outDat = dbnDat1

		file_ext = ".full_geo"
		fname = cluster_data[p] + file_ext + ".tif"
		out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
		print(fname)
		out_ds.SetGeoTransform(geoTransform)
		out_ds.SetProjection(wkt)
		out_ds.GetRasterBand(1).WriteArray(outDat)
		out_ds.FlushCache()
		out_ds = None 


		if apply_context:
			outDat = np.zeros(dbnDat1.shape, dtype=np.int32) - 1
			for i in range(len(context_clusters)):
				clss = context_clusters[i]
				ind = np.where(dbnDat1 == clss)
				outDat[ind] = 1

			outDatFull = np.zeros(imgData.shape, dtype=np.int32) - 1
			if len(subset_inds[p]) > 0:
				outDatFull[subset_inds[0]:subset_inds[1],subset_inds[2]:subset_inds[3]] = outDat
			else:
				outDatFull = outDat
			file_ext = "." + context_name

			fname = cluster_data[p] + file_ext + ".tif"
			out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
			out_ds.SetGeoTransform(geoTransform)
			out_ds.SetProjection(wkt)
			out_ds.GetRasterBand(1).WriteArray(outDatFull)
			out_ds.FlushCache()
			out_ds = None
 
			if compare:
				totalTruth.extend(np.ravel(imgData))
				total.extend(np.ravel(outDatFull))
				cm = confusion_matrix(np.ravel(imgData), np.ravel(outDat), labels = [0,1])
				cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				pprint(cm)
				pprint(cm2) 

		if create_seperate:
			for i in range(len(classes)):
				outDat = np.zeros(dbnDat1.shape, dtype=np.int32) - 1
				clss = classes[i]
				ind = np.where(dbnDat1 == clss)
				outDat[ind] = 1
				file_ext = ".cluster_class" + str(i)
 
				outDatFull = np.zeros(imgData.shape, dtype=np.int32) - 1
				if len(subset_inds[p]) > 0:
					outDatFull[subset_inds[0]:subset_inds[1],subset_inds[2]:subset_inds[3]] = outDat
				else:
					outDatFull = outDat
 
				fname = cluster_data[p] + file_ext + ".tif"
				out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
				out_ds.SetGeoTransform(geoTransform)
				out_ds.SetProjection(wkt)
				out_ds.GetRasterBand(1).WriteArray(outDatFull)
				out_ds.FlushCache()
				out_ds = None
 



	if apply_context and compare:
		cm = confusion_matrix(totalTruth, total)
		cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		unique, counts = np.unique(total, return_counts=True)
		px = dict(zip(unique, counts))         
		unique, counts = np.unique(totalTruth, return_counts=True)
		px2 = dict(zip(unique, counts))

		print("TOTAL STATS:")
		pprint(cm)
		pprint(cm2)
		print("TOTAL PIXELS:", len(totalTruth))
		print("POSITIVE PIXELS:", px)
		print("POSITIVE PIXELS Truth:", px2)
 

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    reader = yml_conf["data"]["clust_reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    cluster_data = yml_conf["data"]["cluster_fnames"]
    gtiff_data = yml_conf["data"]["gtiff_data"]
    create_seperate = yml_conf["data"]["create_seperate"]
    subset_inds = yml_conf["data"]["subset_inds"]

    apply_context = yml_conf["context"]["apply_context"]
    context_clusters = yml_conf["context"]["clusters"]
    context_name = yml_conf["context"]["name"]
    compare = yml_conf["context"]["compare_truth"]

    generate_cluster_gtiffs(data_reader = reader, data_reader_kwargs = data_reader_kwargs, subset_inds = subset_inds,
        cluster_data = cluster_data, gtiff_data = gtiff_data, apply_context = apply_context,
        context_clusters = context_clusters, context_name = context_name, compare = compare, create_seperate = create_seperate)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)
