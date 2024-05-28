"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt

import numpy as np
import healpy as hp



def map_to_hpx(nside, data, location, output):

    data = data[0].ravel()
    theta = np.radians(location[:,:,0]).ravel()
    phi = np.radians(location[:,:,1]).ravel()

    pixel_indices = hp.ang2pix(nside, theta, phi)
    m = np.zeros(hp.nside2npix(nside))-1
  
    print(data.shape, theta.shape, phi.shape, pixel_indices.shape, m.shape)
   
    m[pixel_indices] = data
    hp.write_map(output + ".fits", m, overwrite=True)

    hp.mollview(m)
    plt.savefig(output + ".png", dpi=400) 
   

data = np.load("/data/nlahaye/remoteSensing/VIIRS/ALGAE/npp_viirs_m01_20170920_173000_wgs84_fit.tif.npy")
location = np.load("/data/nlahaye/remoteSensing/VIIRS/ALGAE/npp_viirs_m01_20170920_173000_wgs84_fit.tif.lonlat.npy")

print(location.shape, data.shape)
print(location[:,:,0].min(), location[:,:,0].max())
print(location[:,:,1].min(), location[:,:,1].max())

location = np.flipud(np.fliplr(location))

location[:,:,1] = location[:,:,1] + 180
#location[:,:,0] = location[:,:,0] + 90

map_to_hpx(128, data, location)

