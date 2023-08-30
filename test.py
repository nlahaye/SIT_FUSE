import glob
import os
import numpy as np
import yaml
from osgeo import gdal, osr
from scipy.ndimage import uniform_filter, variance

def lee_filter(img, size):
    """
        Lee Speckle Filter for synthetic aperature radar data.
        
        img: image data
        size: size of Lee Speckle Filter window (optimal size is usually 5)
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def read_uavsar(in_fps, desc_out=None, type_out=None, search_out=None, **kwargs):
    """
    Reads UAVSAR data. 

    Args:
        in_fps (list(string) or string):  list of strings (each file will be treated as a separate channel)
                                          or string of data file paths
        desc_out (optional): if specified, is set to the annotation description of the files converted 
        type_out (optional): if specified, is set to the filetype of the files converted 
        search_out (optional): if specified, is set to the search keyword used to search the .ann file
        kwargs:
            ann_fps (list(string) or string): list of or string of UAVSAR annotation file paths,
                                          ann files will be automatically matched to data files
            pol_modes (list(string)) (optional): list of allowed polarization modes 
                                                    to filter for (e.g. ['HHHH', 'HVHV', 'VVVV'])
            linear_to_dB (bool) (optional): convert linear amplitude units to decibels

    Returns:
        data: numpy array of shape (channels, lines, samples) 
              Complex-valued (unlike polarization) data will be split into separate phase and amplitude channels. 
    """
    
    if "ann_fps" in kwargs:
        ann_fps = kwargs["ann_fps"]
    else:
        raise Exception("No annotation files specified.")
    
    if "pol_modes" in kwargs:
        pol_modes = list(kwargs["pol_modes"])
    else:
        pol_modes = None
    if "linear_to_dB" in kwargs:
        linear_to_dB = kwargs["linear_to_dB"]
    else:
        linear_to_dB = False

    if isinstance(in_fps, str):
        in_fps = [in_fps]
    if isinstance(ann_fps, str):
        ann_fps = [ann_fps]
    
    data = []
    
    print("Reading UAVSAR files...")
    
    # Filter allowed polarization modes
    if pol_modes:
        tmp = []
        for fp in in_fps:
            if any(mode in os.path.basename(fp) for mode in pol_modes):
                tmp.append(fp)
        in_fps = tmp
    
    for fp in in_fps:
        
        # Locate file and matching annotation
        if not os.path.os.path.exists(fp):
            raise Exception(f"Failed to find file: {fp}")
        fname = os.path.basename(fp)
        id = "_".join(fname.split("_")[0:4])
        ann_fp = None
        for ann in ann_fps:
            if id in os.path.basename(ann):
                ann_fp = ann
        if not ann_fp:
            raise Exception(f"File {fname} does not have an associated annotation file.")
        
        print(f"file: {fp}")
        print(f"matching ann file: {ann_fp}")
    
        exts = fname.split('.')[1:]

        if len(exts) == 2:
            ext = exts[1]
            type = exts[0]
        elif len(exts) == 1:
            type = ext = exts[0]
        else:
            raise ValueError('Unable to parse extensions')
        
        # Check for compatible extensions
        if type == 'zip':
            raise Exception('Cannot convert zipped directories. Unzip first.')
        if type == 'dat' or type == 'kmz' or type == 'kml' or type == 'png' or type == 'tif':
            raise Exception(f"Cannot handle {type} products")
        if type == 'ann':
            raise Exception('Cannot convert annotation files.')
            
        # Check for slant range files and ancillary files
        anc = None
        if type == 'slope' or type == 'inc':
            anc = True

        # Read in annotation file
        desc = read_annotation(ann_fp)

        if 'start time of acquisition for pass 1' in desc.keys():
            mode = 'insar'
            raise Exception('INSAR data currently not supported.')
        else:
            mode = 'polsar'

        # Determine the correct file typing for searching data dictionary
        if not anc:
            if mode == 'polsar':
                if type == 'hgt':
                    search = type
                else:
                    polarization = os.path.os.path.basename(fp).split('_')[5][-4:]
                    if polarization == 'HHHH' or polarization == 'HVHV' or polarization == 'VVVV':
                            search = f'{type}_pwr'
                    else:
                        search = f'{type}_phase'
                    type = polarization

            elif mode == 'insar':
                if ext == 'grd':
                    if type == 'int':
                        search = f'grd_phs'
                    else:
                        search = 'grd'
                else:
                    if type == 'int':
                        search = 'slt_phs'
                    else:
                        search = 'slt'
                pass
        else:
            search = type

        # Pull the appropriate values from our annotation dictionary
        nrow = desc[f'{search}.set_rows']['value']
        ncol = desc[f'{search}.set_cols']['value']

        # Set up datatypes
        com_des = desc[f'{search}.val_frmt']['value']
        com = False
        if 'COMPLEX' in com_des:                                    
            com = True
        if com:
            dtype = np.complex64
        else:
            dtype = np.float32

        # Read in binary data
        dat = np.fromfile(fp, dtype = dtype)
        if com:
            dat = np.abs(dat)
            phase = np.angle(dat)
            
        # Change zeros and -10,000 to fillvalue and convert linear units to dB if specified
        fillvalue = -9999.0
        dat[dat==0] = fillvalue
        dat[dat==-10000] = fillvalue
                
        if linear_to_dB:
            dat = 10.0 * np.log10(dat)
            
        # Reshape it to match what the text file says the image is
        if type == 'slope':
            slopes = {}
            slopes['east'] = dat[::2].reshape(nrow, ncol)
            slopes['north'] = dat[1::2].reshape(nrow, ncol)
            dat = slopes
        else:
            slopes = None
            dat = dat.reshape(nrow, ncol)
            if com:
                phase = phase.reshape(nrow, ncol)
        
        # Apply 5x5 Lee Speckle Filter
        if not anc and type != 'hgt':
            if com:
                dat = lee_filter(np.real(dat), 5) + np.imag(dat)
            else:
                lee_filter(dat, 5)

        data.append(dat)
        if com:
            data.append(phase)
            
        dat = None
        phase = None
    
    data = np.array(data)
    
    if "start_line" in kwargs and "end_line" in kwargs and "start_sample" in kwargs and "end_sample" in kwargs:
        data = data[:, kwargs["start_line"]:kwargs["end_line"], kwargs["start_sample"]:kwargs["end_sample"]]
    
    if search_out:
        search_out = search
    if desc_out:
        desc_out = desc
    if type_out:
        type_out = type
    
    return data


def read_annotation(ann_file):
    """
    Reads a UAVSAR annotation file.

    Args:
        ann_file: path to the annotation file

    Returns:
        data: a dictionary of the annotation's contents, 
              data[key] = {'value': value, 'units': units, 'comment': comment}
    """
    with open(ann_file) as fp:
        lines = fp.readlines()
        fp.close()
    data = {}

    # loop through the data and parse
    for line in lines:

        # Filter out all comments and remove any line returns
        info = line.strip().split(';')
        comment = info[-1].strip().lower()
        info = info[0]
        
        # Ignore empty strings
        if info and "=" in info:
            d = info.strip().split('=')
            name, value = d[0], d[1]
            name_split = name.split('(')
            key = name_split[0].strip().lower()
            
            # Isolate units encapsulated between '(' and ')'
            if len(name_split) > 1:
                lidx = name_split[-1].find('(') + 1
                ridx = name_split[-1].find(')')
                units = name_split[-1][lidx:ridx]
            else:
                units = None

            value = value.strip()

            # Cast the values that can be to numbers ###
            if value.strip('-').replace('.', '').isnumeric():
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)

            # Assign each entry as a dictionary with value and units
            data[key] = {'value': value, 'units': units, 'comment': comment}

    return data

def uavsar_to_geotiff(in_fps, out_dir, **kwargs):
    """
    Converts UAVSAR file(s) to geotiff.
    Args:
        in_fps (list(string) or string):  list of strings (each file will be treated as a separate channel)
                                          or string of data file paths
        out_dir (string): directory to which the geotiffs will be saved
        kwargs:
            ann_fps (list(string) or string): list of or string of UAVSAR annotation file paths,
                                            ann files will be automatically matched to data files

    Returns:
        data: numpy array of shape (channels, lines, samples) 
              Complex-valued (unlike polarization) data will be split into separate phase and amplitude channels. 
    """
    
    if "ann_fps" in kwargs:
        ann_fps = kwargs["ann_fps"]

    if not out_dir:
        out_dir = os.path.dirname(in_fps)
    if os.path.isfile(out_dir):
        raise Exception('Provide a directory, not a filepath.')
    
    desc = None
    type = None
    search = None
    
    data = read_uavsar(in_fps, desc, type, search, **kwargs)
    out_fps = []
    for dat, fp in zip(data, in_fps):
        
        fname = os.path.os.path.basename(fp)
        out_fp = os.path.join(out_dir, fname) + '.tiff'
        exts = fname.split('.')[1:]
        dtype = dat.dtype
        if dtype == np.complex64:
            bands = 2
        else:
            bands = 1

        driver = gdal.GetDriverByName("GTiff")
            
        # If ground projected image, north up...
        if type in {'grd', 'slope', 'inc'}: 
            # Delta latitude and longitude
            dlat = float(desc[f'{search}.row_mult']['value'])
            dlon = float(desc[f'{search}.col_mult']['value'])
            # Upper left corner coordinates
            lat1 = float(desc[f'{search}.row_addr']['value'])
            lon1 = float(desc[f'{search}.col_addr']['value'])
            # Set up geotransform for gdal
            srs = osr.SpatialReference()
            # WGS84 Projection, spatial reference using the EPSG code (4326)
            srs.ImportFromEPSG(4326)
            t = [lon1, dlon, 0.0, lat1, 0.0, dlat]

        if type == 'slope':
                out_fps = []
                for direction, arr in data.items():
                    slope_fp = out_fp.replace('.tiff',f'.{direction}.tiff')
                    print(f"saving to {slope_fp}.")
                    ds = driver.Create(slope_fp, 
                                        ysize=arr.shape[0], 
                                        xsize=arr.shape[1], 
                                        bands=bands, 
                                        eType=gdal.GDT_Float32)
                    ds.SetProjection(srs.ExportToWkt())
                    ds.SetGeoTransform(t)
                    ds.GetRasterBand(1).WriteArray(np.abs(dat))
                    ds.GetRasterBand(1).SetNoDataValue(np.nan)
                    ds.FlushCache()
                    ds = None
                    out_fps.append(slope_fp)
        else:
            ds = driver.Create(out_fp, 
                                    ysize=dat.shape[0], 
                                    xsize=dat.shape[1], 
                                    bands=bands, 
                                    eType=gdal.GDT_Float32)
            if type in {'grd', 'inc'}:
                ds.SetProjection(srs.ExportToWkt())
                ds.SetGeoTransform(t)
            if bands == 2:
                ds.GetRasterBand(1).WriteArray(np.abs(dat))
                ds.GetRasterBand(1).SetNoDataValue(np.nan)
                ds.GetRasterBand(2).WriteArray(np.angle(dat))
                ds.GetRasterBand(2).SetNoDataValue(np.nan)
            else:
                ds.GetRasterBand(1).WriteArray(dat)
                ds.GetRasterBand(1).SetNoDataValue(np.nan)
            out_fps.append(out_fp)
            
        ds.FlushCache() # save tiffs
        ds = None  # close the dataset
    
    print("Saved geotiffs to:")
    print(out_fps, sep='\n')
    return out_fps

def read_yaml(fpath_yaml):
    yml_conf = None
    with open(fpath_yaml) as f_yaml:
        yml_conf = yaml.load(f_yaml, Loader=yaml.FullLoader)
    return yml_conf


def main():
    # dir = "/work/09562/nleet/ls6/output"
    # # insearch = glob.glob(os.path.join(dir, "caldor_26200_21048_013_210825_L090VVVV*.input"))
    # dsearch = glob.glob(os.path.join(dir, "caldor_26200_21048_013_210825_L090VVVV*.data"))
    # isearch = glob.glob(os.path.join(dir, "caldor_26200_21048_013_210825_L090VVVV*.indices"))
    # d = load(dsearch[0]).numpy()
    # i = load(isearch[1])
    # print(d.shape)
    # print(i.shape)

    # f1 = "/work/09562/nleet/ls6/data/uavsar/fire-scenes/L-band/caldor_08200_21049_026_210831_L090_CX_01/caldor_08200_21049_026_210831_L090HHHH_CX_01.grd"
    # a1 = "/work/09562/nleet/ls6/data/uavsar/fire-scenes/L-band/caldor_08200_21049_026_210831_L090_CX_01/uavsar.asf.alaska.edu_UA_caldor_08200_21049_026_210831_L090_CX_01_caldor_08200_21049_026_210831_L090_CX_01.ann"
    # f2 = "/work/09562/nleet/ls6/data/uavsar/fire-scenes/L-band/caldor_26200_21048_013_210825_L090_CX_01/caldor_26200_21048_013_210825_L090HHHH_CX_01.grd"
    # a2 = "/work/09562/nleet/ls6/data/uavsar/fire-scenes/L-band/caldor_26200_21048_013_210825_L090_CX_01/uavsar.asf.alaska.edu_UA_caldor_26200_21048_013_210825_L090_CX_01_caldor_26200_21048_013_210825_L090_CX_01.ann"
    # d1 = read_uavsar(f1, ann_fps=a1)
    # d2 = read_uavsar(f2, ann_fps=a2)
    # print(d1.shape)
    # print(d2.shape)
    
    dir = "/home/niklas/JSIP/uavsar-data/fire-scenes/L-band/SanAnd_26526_17122_004_171102_L090_CX_01"
    in_fps = glob.glob(os.path.join(dir, "*.grd"))
    print(in_fps)
    yml = "/vol/JIFRESSE/SIT_FUSE/config/dbn/uavsar_dbn_test.yaml"
    yml_conf = read_yaml(yml)
    reader_kwargs = yml_conf['data']['reader_kwargs']
    uavsar_to_geotiff(in_fps, dir, **reader_kwargs)
    
if __name__ == '__main__':
    main()
 


