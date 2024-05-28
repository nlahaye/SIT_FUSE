"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import MisrToolkit as Mtk
import numpy as np
import cv2

BAND_NAMES = ["RedBand", "GreenBand", "BlueBand", "NIRBand"]
RADIANCE_NAMES = ['Red Radiance', 'Green Radiance', 'Blue Radiance', 'NIR Radiance']
 

def preprocess(fname, blocks):
 
        bands = BAND_NAMES
        radianceNames = RADIANCE_NAMES

	datRet = None 
	f =  Mtk.MtkFile(fname)
	r = Mtk.MtkRegion(f.path, blocks[0], blocks[1])
	dat = [0] * len(bands)
	val = [0] * len(bands)
	valmask = [0] * len(bands)
	mask = [0] * len(bands)
	for i in range(len(bands)):
		print(bands[i], radianceNames[i])
		print("READING")
		g = f.grid(bands[i])
		print("GRID")
		fld = g.field(radianceNames[i])
		scale = g.attr_get('Scale factor')
		print("FIELD")
		tmp = fld.read(r)
		dat[i] = np.array(tmp.data())
		rdqi = dat[i].astype(np.uint16) & (2**15 | 2**14)
		dat[i] = dat[i].astype(np.uint16) & (2**14 - 1)
		dat[i][np.where(rdqi > 1)] = 0.0

		dat[i] = dat[i].astype(np.float64) * scale

		print(dat[i].mean(), dat[i].min(), dat[i].max())
		if "_AN_" in fname or "Red" in bands[i]:
			print("MASKING")
			mask[i] = np.zeros(dat[i].shape, dtype=np.uint8)
			mask[i][np.where(dat[i] > 0.0000005)] = 1
			print("DOWNSAMPLING")
			val[i] = cv2.GaussianBlur(dat[i],(5,5),0)
			val[i], valmask[i] =  resample(val[i], mask[i], 4)
			print("DOWNSAMPLED")
			x = np.array(val[i])
		else:
			x = np.array(dat[i])

		if datRet is None:
			datRet = np.zeros((len(bands), x.shape[0], x.shape[1]))
		
		print(x.mean(), x.min(), x.max())
		print(x.shape)
		datRet[i,:,:] = x

	np.save(fname + ".gauss.npy", datRet)
	print("RETURNING")


def resample(data, mask, sze):

	datRet = np.zeros((int(data.shape[0]/sze), int(data.shape[1]/sze)))
	maskRet = np.zeros((int(data.shape[0]/sze), int(data.shape[1]/sze)))

	for i in range(datRet.shape[0]):
		for j in range(datRet.shape[1]):
			strtLine = i * sze
			endLine = strtLine + sze

			strtSamp = j * sze
			endSamp = strtSamp + sze

			count = 0
			sm = 0.0

			for line in range(strtLine, endLine):
				for samp in range(strtSamp, endSamp):
					if mask[line, samp] == 1:
						sm = sm + data[line, samp]
						count = count + 1
				
			if count > 0:
				maskRet[i,j] = 1
				datRet[i,j] = sm / float(count)	
	

	return datRet, maskRet	


def smooth(data, mask, windowLine, windowSamp):

	dt = np.zeros(data.shape)
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if mask[i,j] == 0:
				continue
			sm = 0
			count = 0
			for wl in range(int(i - (windowLine - 1) / 2), int(i + (windowLine - 1) / 2) + 1):
				for ws in range(int(i - (windowSamp - 1) / 2), int(i + (windowSamp - 1) / 2) + 1):

					if wl >= 0 and wl < data.shape[0] and ws >= 0 and ws < data.shape[1] and mask[wl,ws] == 1:
						sm = sm + data[wl,ws]
					else:
						sm = sm + data[i,j]
					count = count + 1


			dt[i,j] = sm / float(count)

def preprocess_latlon(fname, blocks):

    f =  Mtk.MtkFile(fname)
    r = Mtk.MtkRegion(f.path, blocks[0], blocks[1])
    mp = r.snap_to_grid(f.path, 1100)
    lat, lon = mp.create_latlon()
    dat = np.array([lat, lon])
    np.save(fname + ".gauss.latlon.npy", dat)



def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    blocks = yml_conf["blocks"]
    fnames = yml_conf["filenames"]
    
    for i in range(len(filenames)):
        preprocess(filenames[i], blocks)
        preprocess_latlon(filenames[i], blocks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for MISR data info.")
    args = parser.parse_args()
    main(args.yaml)



