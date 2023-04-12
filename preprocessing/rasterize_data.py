"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

from osgeo import gdal, ogr



def rasterize(nodata, vector_fn, raster_fn, pixel_size):
    source_ex = gdal.OpenEx(vector_fn)
    
    gdal.Rasterize(raster_fn, source_ex, format='GTIFF', outputType=gdal.GDT_Byte, creationOptions=["COMPRESS=DEFLATE"], noData=NoData_value, initValues=NoData_value, xRes=pixel_size, yRes=-pixel_size, allTouched=True, burnValues=[1])    


def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    tile_data(yml_conf["nodata"], yml_conf["vector_fn"], yml_conf["raster_fn"], yml_conf["pixel_size"])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)



