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
import zarr

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
 
def generate_cluster_masks_no_geo(data_reader, data_reader_kwargs, subset_inds, cluster_data, 
    apply_context, context_clusters, context_name, compare, create_separate, generate_union = False, cluster_dependencies={}):

        read_func = get_read_func(data_reader)
    
        total = []
        totalTruth = []
        outUnion = None
        unionCount = None
        for p in range(len(cluster_data)):
            print("FNAME1", cluster_data[p])

            dbnDat1 = read_func(cluster_data[p], **data_reader_kwargs).astype(np.int32)
            classes = np.unique(dbnDat1)
            outDat = np.zeros((ny,nx), dtype=np.int32)
 
            if apply_context:
               outDat = np.zeros(dbnDat1.shape, dtype=np.int32)
               inds = np.where(dbnDat1 < 0)
               outDat[inds] = -1
               if generate_union > 0 and outUnion is None:
                   #union cases assume input scenes are all the same size        
                   outUnion = np.zeros((ny,nx), dtype=np.int32)
                   unionCount = np.zeros((ny,nx), dtype=np.int32)

               if not isinstance(context_clusters[0], list):
                       tmp = []
                       tmp.append(context_clusters)
                       context_clusters = tmp
               for j in range(len(context_clusters)):
                       for i in range(len(context_clusters[j])):
                           clss = context_clusters[j][i]
                           ind = np.where(dbnDat1 == clss)
                           if context_clusters[j][i] in cluster_dependencies:
                                                ind = apply_dependencies(cluster_dependencies[context_clusters[j][i]], ind, dbnDat1)

                           outDat[ind] = (j+1)
                           if generate_union > 0 and outUnion is not None:
                               outUnion[ind] = outUnion[ind] + (j+1)
                               unionCount[ind] = unionCount[ind] + 1
    
               inds = np.where((outDat < 0) & (dbnDat1 >= 0))
               outDat[inds] = 0
               outDatFull = np.zeros((ny,nx), dtype=np.int32) - 1
               if len(subset_inds[p]) > 0:
                    outDatFull[subset_inds[p][0]:subset_inds[p][1],subset_inds[p][2]:subset_inds[p][3]] = outDat
               else:
                    outDatFull = outDat
               file_ext = "." + context_name
    
               fname = cluster_data[p] + file_ext + ".zarr"
               zarr.save(fname,outDatFull)
               img = plt.imshow(outDatFull, vmin=-1, vmax=1)
               plt.savefig(fname + ".png", dpi=400, bbox_inches='tight')

               if generate_union > 0 and p == len(cluster_data)-1:
                                fname = os.path.join(os.path.dirname(cluster_data[p]), context_name + ".Union.zarr")

                                inds = np.where(unionCount == 0)
                                unionCount[inds] = 1
                                outUnion = np.divide(outUnion,unionCount).astype(np.int32)
    
                                outUnionFull = None
                                if len(subset_inds[p]) > 0:
                                    outUnionFull[subset_inds[p][0]:subset_inds[p][1],subset_inds[p][2]:subset_inds[p][3]] = outUnion
                                else:
                                    outUnionFull = outUnion
                                inds = np.where(outUnionFull <= 0)
                                outUnionFull[inds] = 0
                                #inds = np.where(outUnionFull > 0)
                                #outUnionFull[inds] = 1
                                zarr.save(fname,outUnionFull)
                                img = plt.imshow(outDatFull, vmin=-1, vmax=1)
                                plt.savefig(fname + ".png", dpi=400, bbox_inches='tight')
 

            if create_separate:
                            for i in range(len(classes)):
                                    outDat = np.zeros((ny,nx), dtype=np.int32) - 1
                                    clss = classes[i]
                                    ind = np.where(dbnDat1 == clss)
                                    outDat[ind] = 1
                                    inds = np.where((outDat < 0) & (dbnDat1 >= 0))
                                    outDat[inds] = 0
                                    file_ext = ".cluster_class" + str(clss)

                                    outDatFull = np.zeros((ny,nx), dtype=np.int32) - 1
                                    if len(subset_inds[p]) > 0:
                                            outDatFull[subset_inds[p][0]:subset_inds[p][1],subset_inds[p][2]:subset_inds[p][3]] = outDat
                                    else:
                                            outDatFull = outDat

                                    fname = cluster_data[p] + file_ext + ".zarr"
                                    zarr.save(fname,outDatFull)
                                    img = plt.imshow(outDatFull, vmin=-1, vmax=1)
                                    plt.savefig(fname + ".png", dpi=400, bbox_inches='tight') 
 

def generate_cluster_gtiffs(data_reader, data_reader_kwargs, subset_inds,
    cluster_data, gtiff_data, apply_context, context_clusters, context_name, compare, create_separate, generate_union = False, cluster_dependencies={}):

	read_func = get_read_func(data_reader)

	total = []
	totalTruth = []
	outUnion = None
	unionCount = None
	for p in range(len(cluster_data)):

		if not os.path.exists(cluster_data[p]):
			continue		

		dbnDat1 = read_func(cluster_data[p], **data_reader_kwargs).astype(np.int32)
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

 
		if len(imgData.shape) > 2:
			imgData = np.squeeze(imgData[0,:,:])
		nx = max(dbnDat1.shape[1],imgData.shape[1])
		ny = max(dbnDat1.shape[0],imgData.shape[0])
		metadata=dat.GetMetadata()
		geoTransform = dat.GetGeoTransform()
		wkt = dat.GetProjection()
		gcpcount = dat.GetGCPCount()
		gcp = None
		gcpproj = None
		if gcpcount > 0:
			gcp = dat.GetGCPs()
			gcpproj = dat.GetGCPProjection()
		dat.FlushCache()
		dat = None			
 
		classes = np.unique(dbnDat1)
		outDat = np.zeros((ny,nx), dtype=np.int32)
		#dbnDat1 = dbnDat1 / 1000.0
		dbnDat1 = dbnDat1.astype(np.int32)
		if len(subset_inds[p]) > 0:
			outDat[subset_inds[p][0]:subset_inds[p][1],subset_inds[p][2]:subset_inds[p][3]] = dbnDat1
		else:
			outDat[0:dbnDat1.shape[0],0: dbnDat1.shape[1]] = dbnDat1


		inds = np.where(imgData < 0)
		outDat[inds] = -1
		file_ext = ".full_geo"
		fname = cluster_data[p] + file_ext + ".tif"
		out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int32)
		out_ds.SetMetadata(metadata)
		out_ds.SetGeoTransform(geoTransform)
		out_ds.SetProjection(wkt)
		if gcpcount > 0:
			out_ds.SetGCPs(gcp, gcpproj)
		out_ds.GetRasterBand(1).WriteArray(outDat)
		out_ds.FlushCache()
		out_ds = None 


		if apply_context:
			outDat = np.zeros(dbnDat1.shape, dtype=np.int32)
			inds = np.where(imgData < 0)
			outDat[inds] = -1
			if generate_union > 0 and outUnion is None:
				#union cases assume input scenes are all the same size
				outUnion = np.zeros(dbnDat1.shape, dtype=np.int32) 	
				unionCount = np.zeros(dbnDat1.shape, dtype=np.int32) 

			if not isinstance(context_clusters[0], list):
				tmp = []
				tmp.append(context_clusters)
				context_clusters = tmp
			for j in range(len(context_clusters)):
				for i in range(len(context_clusters[j])):
					clss = context_clusters[j][i]
					ind = np.where(dbnDat1 == clss)
					if context_clusters[j][i] in cluster_dependencies:
						ind = apply_dependencies(cluster_dependencies[context_clusters[j][i]], ind, dbnDat1)
					outDat[ind] = (j+1)
					if generate_union > 0 and outUnion is not None:
						outUnion[ind] = outUnion[ind] + (j+1)
						unionCount[ind] = unionCount[ind] + 1

			inds = np.where((outDat[0:dbnDat1.shape[0],0: dbnDat1.shape[1]] < 0) & (dbnDat1 >= 0))
			outDat[inds] = 0
			outDatFull = np.zeros(imgData.shape, dtype=np.int32) - 1
			if len(subset_inds[p]) > 0:
				outDatFull[subset_inds[p][0]:subset_inds[p][1],subset_inds[p][2]:subset_inds[p][3]] = outDat
			else:
				outDatFull = outDat
			file_ext = "." + context_name

			fname = cluster_data[p] + file_ext + ".tif"
			out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int32)
			out_ds.SetGeoTransform(geoTransform)
			out_ds.SetMetadata(metadata)
			out_ds.SetProjection(wkt)
			if gcpcount > 0:
				out_ds.SetGCPs(gcp, gcpproj)
			out_ds.GetRasterBand(1).WriteArray(outDatFull)
			out_ds.FlushCache()
			out_ds = None

			if generate_union > 0 and p == len(cluster_data)-1:
				fname = os.path.join(os.path.dirname(cluster_data[p]), context_name + ".Union.tif")
				out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int32)
				out_ds.SetGeoTransform(geoTransform)
				out_ds.SetMetadata(metadata)
				out_ds.SetProjection(wkt)
				if gcpcount > 0:
					out_ds.SetGCPs(gcp, gcpproj)

				inds = np.where(unionCount == 0)
				unionCount[inds] = 1
				outUnion = np.divide(outUnion,unionCount).astype(np.int32)

				outUnionFull = None
				if len(subset_inds[p]) > 0:
					outUnionFull[subset_inds[p][0]:subset_inds[p][1],subset_inds[p][2]:subset_inds[p][3]] = outUnion
				else:
					outUnionFull = outUnion
				inds = np.where(outUnionFull <= 0)
				outUnionFull[inds] = 0 
				#inds = np.where(outUnionFull > 0)
				#outUnionFull[inds] = 1
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
				inds = np.where((outDat < 0) & (dbnDat1 >= 0))
				outDat[inds] = 0
				file_ext = ".cluster_class" + str(clss)
 
				outDatFull = np.zeros(imgData.shape, dtype=np.int32) - 1
				if len(subset_inds[p]) > 0:
					outDatFull[subset_inds[p][0]:subset_inds[p][1],subset_inds[p][2]:subset_inds[p][3]] = outDat
				else:
					outDatFull = outDat
 
				fname = cluster_data[p] + file_ext + ".tif"
				out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int32)
				out_ds.SetGeoTransform(geoTransform)
				out_ds.SetMetadata(metadata)
				out_ds.SetProjection(wkt)
				if gcpcount > 0:
					out_ds.SetGCPs(gcp, gcpproj)
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
 


def generate_separate_from_full(gtiff_data, apply_context, context_clusters, context_name, create_separate=True, generate_union=False, cluster_dependencies={}):
        outUnion = None
        unionCount = None
        for p in range(len(gtiff_data)):
                print(gtiff_data[p])

                dat = gdal.Open(gtiff_data[p])
                imgData = dat.ReadAsArray().astype(np.int32)
                if len(imgData.shape) > 2:
                        imgData = np.squeeze(imgData[0,:,:])
                nx = max(dbnDat1.shape[1],imgData.shape[1])
                ny = max(dbnDat1.shape[0],imgData.shape[0])
                geoTransform = dat.GetGeoTransform()
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

                classes = np.unique(imgData)
 
                if apply_context:
                        outDatFull = np.zeros((ny,nx), dtype=np.int32) - 1
                        if generate_union > 0 and outUnion is None:
                                #union cases assume input scenes are all the same size
                                outUnion = np.zeros((ny,nx), dtype=np.int32) - 1
                                unionCount = np.zeros((ny,nx), dtype=np.int32)

                        if not isinstance(context_clusters[0], list):
                            tmp = []
                            tmp.append(context_clusters)
                            context_clusters = tmp
                        for j in range(len(context_clusters)):
                            for i in range(len(context_clusters[j])):
                                clss = context_clusters[j][i]
                                ind = np.where(imgData == clss)
                                if context_clusters[j][i] in cluster_dependencies:
                                                ind = apply_dependencies(cluster_dependencies[context_clusters[j][i]], ind, dbnDat1)	
                                outDatFull[ind] = (j+1)
                                if generate_union > 0 and outUnion is not None:
                                    outUnion[ind] = outUnion[ind] + 1
                                    unionCount[ind] = unionCount[ind] + 1

                        inds = np.where((outDatFull < 0) & (imgData >= 0))
                        outDatFull[inds] = 0
                        file_ext = "." + context_name

                        fname = os.path.splitext(gtiff_data[p])[0] + file_ext + ".tif"
                        out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int32)
                        out_ds.SetGeoTransform(geoTransform)
                        out_ds.SetMetadata(metadata)
                        out_ds.SetProjection(wkt)
                        if gcpcount > 0:
                            out_ds.SetGCPs(gcp, gcpproj)
                        out_ds.GetRasterBand(1).WriteArray(outDatFull)
                        out_ds.FlushCache()
                        out_ds = None
                         
                        if generate_union > 0 and p == len(gtiff_data)-1:
                                fname = os.path.join(os.path.dirname(gtiff_data[p]), context_name + ".Union.tif")
                                out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int32)
                                out_ds.SetGeoTransform(geoTransform)
                                out_ds.SetMetadata(metadata)
                                out_ds.SetProjection(wkt)
                                if gcpcount > 0:
                                    out_ds.SetGCPs(gcp, gcpproj)
 
                                inds = np.where(unionCount == 0)
                                unionCount[inds] = 1
                                outUnion = np.divide(outUnion,unionCount,dtype=np.int32)


                                outUnionFull = None
                                if len(subset_inds[p]) > 0:
                                        outUnionFull[subset_inds[p][0]:subset_inds[p][1],subset_inds[p][2]:subset_inds[p][3]] = outUnion
                                else:
                                        outUnionFull = outUnion

                                inds = np.where(outUnionFull <= 0)
                                outUnionFull[inds] = 0
                                #inds = np.where(outUnionFull > 0)
                                #outUnionFull[inds] = 1
                                out_ds.GetRasterBand(1).WriteArray(outUnionFull)
                                out_ds.FlushCache()
                                out_ds = None


                if create_separate:
                    for i in range(len(classes)):
                        outDatFull = np.zeros((ny,nx), dtype=np.int32) - 1
                        clss = classes[i]
                        ind = np.where(imgData == clss)
                        outDatFull[ind] = 1
                        file_ext = ".cluster_class" + str(clss)

                        fname = gtiff_data[p] + file_ext + ".tif"
                        out_ds = gdal.GetDriverByName("GTiff").Create(fname, nx, ny, 1, gdal.GDT_Int32)
                        out_ds.SetGeoTransform(geoTransform)
                        out_ds.SetMetadata(metadata)
                        out_ds.SetProjection(wkt)
                        if gcpcount > 0:
                            out_ds.SetGCPs(gcp, gcpproj)
                        out_ds.GetRasterBand(1).WriteArray(outDatFull)
                        out_ds.FlushCache()
                        out_ds = None




def apply_dependencies(clust_deps, inds, dbnDat, window = 10): #TODO configurable

    final_inds_y = []
    final_inds_x = []
    for i in range(len(inds[1])):
        wind_min_y = max(0,inds[0][i]-window)
        wind_max_y = min(dbnDat.shape[0],inds[0][i]+window)
        wind_min_x = max(0,inds[1][i]-window)
        wind_max_x = min(dbnDat.shape[1],inds[1][i]+window)

        running_count = 0
        for d in range(len(clust_deps)):
            if clust_deps[d] in dbnDat[wind_min_y:wind_max_y,wind_min_x:wind_max_x]:
                    running_count += len(np.where(dbnDat[wind_min_y:wind_max_y,wind_min_x:wind_max_x] == clust_deps[d])[0])
                    if running_count > 10:
                        final_inds_y.append(inds[0][i])
                        final_inds_x.append(inds[1][i])
                        break

    return final_inds_y,final_inds_x

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

    if len(subset_inds) == 0 or len(subset_inds) < len(gtiff_data):
        subset_inds = [ [] for _ in range(len(gtiff_data)) ]

    apply_context = yml_conf["context"]["apply_context"]
    generate_union = yml_conf["context"]["generate_union"]
    context_clusters = yml_conf["context"]["clusters"]
    context_name = yml_conf["context"]["name"]
    compare = yml_conf["context"]["compare_truth"]

    gen_from_gtiffs = yml_conf["gen_from_geotiffs"]

    clust_dep = {}
    if "cluster_dependencies" in yml_conf["context"]:
        clust_dep = yml_conf["context"]["cluster_dependencies"]

 
    if gtiff_data is not None:
        if gen_from_gtiffs:
            generate_separate_from_full(gtiff_data = cluster_data, apply_context = apply_context,
                context_clusters = context_clusters, context_name = context_name, create_separate=create_separate, generate_union=generate_union, cluster_dependencies=clust_dep)
        else: 
            generate_cluster_gtiffs(data_reader = reader, data_reader_kwargs = data_reader_kwargs, subset_inds = subset_inds,
                cluster_data = cluster_data, gtiff_data = gtiff_data, apply_context = apply_context,
                context_clusters = context_clusters, context_name = context_name, compare = compare, 
                    create_separate = create_separate, generate_union=generate_union, cluster_dependencies=clust_dep)
    else:
        generate_cluster_masks_no_geo(data_reader = reader, data_reader_kwargs = data_reader_kwargs, subset_inds = subset_inds,
                cluster_data = cluster_data, apply_context = apply_context, context_clusters = context_clusters, 
                context_name = context_name, compare = compare, create_separate = create_separate, generate_union=generate_union, cluster_dependencies=clust_dep)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
    args = parser.parse_args()
    main(args.yaml)
