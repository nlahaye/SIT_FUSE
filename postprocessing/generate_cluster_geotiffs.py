"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import numpy as np
import matplotlib
matplotlib.use('agg')
from pprint import pprint
from sklearn.metrics import confusion_matrix
from osgeo import gdal, osr
import argparse
import os
from utils import numpy_to_torch, read_yaml, get_read_func

def generate_cluster_gtiffs(data_reader, data_reader_kwargs, subset_inds,
    cluster_data, gtiff_data, apply_context, context_clusters, context_name, compare, create_separate, generate_union = False):

	read_func = get_read_func(data_reader)

	total = []
	totalTruth = []
	outUnion = None
	for p in range(len(cluster_data)):
		print(cluster_data[p])
		print(gtiff_data[p])
               
		dbnDat1 = read_func(cluster_data[p], **data_reader_kwargs).astype(np.int32)
		print(dbnDat1.min(), dbnDat1.max())
		#dbnDat1 = np.flipud(dbnDat1)
		dat = gdal.Open(gtiff_data[p])
		imgData = dat.ReadAsArray()

		#TODO beter generalize here - needed for GIM GeoTiff generation
		#dat = gdal.Open("NETCDF:{0}:{1}".format("/data/nlahaye/remoteSensing/GIM/jpli/2022/jpli0050.22i.nc", "tecmap"))
		#imgData = dat.ReadAsArray(0, 0, dat.RasterXSize, dat.RasterYSize)
		#ds_lon = gdal.Open('NETCDF:"'+"/data/nlahaye/remoteSensing/GIM/jpli/2022/jpli0050.22i.nc"+'":lon')
		#ds_lat = gdal.Open('NETCDF:"'+"/data/nlahaye/remoteSensing/GIM/jpli/2022/jpli0050.22i.nc"+'":lat')

		#lon = ds_lon.GetRasterBand(1).ReadAsArray()
		#lat = ds_lat.GetRasterBand(1).ReadAsArray()
		#ds_lon = None
		#ds_lat = None
		#wkt = osr.SpatialReference()
		#wkt.ImportFromEPSG(4326)
		#wkt = wkt.ExportToWkt()
		#nx = lon.shape[1]
		#ny = lat.shape[1]
		#xmin, ymin, xmax, ymax = [lon.min(), lat.min(), lon.max(), lat.max()]
		#xres = (xmax - xmin) / float(nx)
		#yres = (ymax - ymin) / float(ny)
		#print(ymax, ymin, ny, lat.shape, lon.shape)
		#geoTransform = (xmin, xres, 0, ymax, 0, -yres)
		#print(geoTransform)

 
		print(len(imgData.shape), imgData.shape)
		if len(imgData.shape) > 2:
			imgData = np.squeeze(imgData[0,:,:])
		print(p)
		print(dbnDat1.shape, imgData.shape)
		nx = imgData.shape[1]
		ny = imgData.shape[0]
		geoTransform = dat.GetGeoTransform()
		wkt = dat.GetProjection()
		print(wkt)
		dat.FlushCache()
		dat = None			
 
		classes = np.unique(dbnDat1)
		print(int(classes.max() - classes.min() + 2))
		outDat = np.zeros(imgData.shape, dtype=np.int32) - 1
		if len(subset_inds[p]) > 0:
			outDat[subset_inds[0]:subset_inds[1],subset_inds[2]:subset_inds[3]] = dbnDat1
		else:
			outDat = dbnDat1

		file_ext = ".full_geo"
		fname = cluster_data[p] + file_ext + ".tif"
		print(fname, "HERE", nx, ny, outDat.shape)
		out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
		print(fname)
		out_ds.SetGeoTransform(geoTransform)
		out_ds.SetProjection(wkt)
		out_ds.GetRasterBand(1).WriteArray(outDat)
		out_ds.FlushCache()
		out_ds = None 


		if apply_context:
			outDat = np.zeros(dbnDat1.shape, dtype=np.int32) - 1
			if generate_union and outUnion is None:
				#union cases assume input scenes are all the same size
				outUnion = np.zeros(dbnDat1.shape, dtype=np.int32) - 1			

			for i in range(len(context_clusters)):
				clss = context_clusters[i]
				ind = np.where(dbnDat1 == clss)
				outDat[ind] = 1
				if outUnion:
					outUnion[ind] = 1

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

			if generate_union and p == len(cluster_data)-1:
				fname = os.path.join(os.path.dirname(cluster_data[p]), context_name + ".Union.tif")
				out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
				out_ds.SetGeoTransform(geoTransform)
				out_ds.SetProjection(wkt)

				outUnionFull = None
				if len(subset_inds[p]) > 0:
					outUnionFull[subset_inds[0]:subset_inds[1],subset_inds[2]:subset_inds[3]] = outUnion
				else:
					outUnionFull = outUnion

				out_ds.GetRasterBand(1).WriteArray(outUnionFull) 
				out_ds.FlushCache()
				out_ds = None

			if compare:
				totalTruth.extend(np.ravel(imgData))
				total.extend(np.ravel(outDatFull))
				cm = confusion_matrix(np.ravel(imgData), np.ravel(outDat), labels = [0,1])
				cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				pprint(cm)
				pprint(cm2) 

		if create_separate:
			for i in range(len(classes)):
				outDat = np.zeros(dbnDat1.shape, dtype=np.int32) - 1
				clss = classes[i]
				ind = np.where(dbnDat1 == clss)
				outDat[ind] = 1
				file_ext = ".cluster_class" + str(clss)
 
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
 


def generate_separate_from_full(gtiff_data, apply_context, context_clusters, context_name, create_separate=True):
        for p in range(len(gtiff_data)):
                print(gtiff_data[p])

                dat = gdal.Open(gtiff_data[p])
                imgData = dat.ReadAsArray().astype(np.int32)
                print(len(imgData.shape), imgData.shape)
                if len(imgData.shape) > 2:
                        imgData = np.squeeze(imgData[0,:,:])
                print(p)
                print(imgData.shape)
                nx = imgData.shape[1]
                ny = imgData.shape[0]
                geoTransform = dat.GetGeoTransform()
                wkt = dat.GetProjection()
                print(wkt)
                dat.FlushCache()
                dat = None

                classes = np.unique(imgData)
 
                if apply_context:
                        outDatFull = np.zeros(imgData.shape, dtype=np.int32) - 1
                        for i in range(len(context_clusters)):
                                clss = context_clusters[i]
                                ind = np.where(imgData == clss)
                                outDatFull[ind] = 1

                        file_ext = "." + context_name

                        fname = os.path.splitext(gtiff_data[p])[0] + file_ext + ".tif"
                        print(fname)
                        out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
                        out_ds.SetGeoTransform(geoTransform)
                        out_ds.SetProjection(wkt)
                        out_ds.GetRasterBand(1).WriteArray(outDatFull)
                        out_ds.FlushCache()
                        out_ds = None


                if create_separate:
                    for i in range(len(classes)):
                        outDatFull = np.zeros(imgData.shape, dtype=np.int32) - 1
                        clss = classes[i]
                        ind = np.where(imgData == clss)
                        outDatFull[ind] = 1
                        file_ext = ".cluster_class" + str(clss)

                        fname = gtiff_data[p] + file_ext + ".tif"
                        out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Float32)
                        out_ds.SetGeoTransform(geoTransform)
                        out_ds.SetProjection(wkt)
                        out_ds.GetRasterBand(1).WriteArray(outDatFull)
                        out_ds.FlushCache()
                        out_ds = None





def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    reader = yml_conf["data"]["clust_reader_type"]
    data_reader_kwargs = yml_conf["data"]["reader_kwargs"]
    cluster_data = yml_conf["data"]["cluster_fnames"]
    gtiff_data = yml_conf["data"]["gtiff_data"]
    create_separate = yml_conf["data"]["create_separate"]
    subset_inds = yml_conf["data"]["subset_inds"]
    print(len(subset_inds))
    if len(subset_inds) == 0:
        subset_inds = [ [] for _ in range(len(gtiff_data)) ]
    print(len(subset_inds))

    apply_context = yml_conf["context"]["apply_context"]
    generate_union = yml_conf["context"]["generate_union"]
    context_clusters = yml_conf["context"]["clusters"]
    context_name = yml_conf["context"]["name"]
    compare = yml_conf["context"]["compare_truth"]

    gen_from_gtiffs = yml_conf["gen_from_geotiffs"]

    if gen_from_gtiffs:
        generate_separate_from_full(gtiff_data = gtiff_data, apply_context = apply_context,
            context_clusters = context_clusters, context_name = context_name, create_separate=create_separate, generate_union=generate_union)
    else: 
        generate_cluster_gtiffs(data_reader = reader, data_reader_kwargs = data_reader_kwargs, subset_inds = subset_inds,
            cluster_data = cluster_data, gtiff_data = gtiff_data, apply_context = apply_context,
            context_clusters = context_clusters, context_name = context_name, compare = compare, create_separate = create_separate, generate_union=generate_union)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)
