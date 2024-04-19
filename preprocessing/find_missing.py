import datetime
import os

#input_init = "/data/nlahaye/remoteSensing/Multi_Year_WQ/SNPP_VIIRS."
#input_end = ".L3m.DAY.RRS.angstrom.4km.nc"

input_init = "/data/nlahaye/remoteSensing/TROPOMI_MODIS_HAB/TROPO_redSIF_"
input_end = "_ungridded.nc"

trop_init = "/data/nlahaye/remoteSensing/TROPOMI_MODIS_HAB/TROPO_redSIF_"
trop_end = "_ungridded.nc"



date_start = datetime.datetime.strptime("20180601", "%Y%m%d")
date_end = datetime.datetime.strptime("20191231", "%Y%m%d")

while date_start <= date_end:
    fname = input_init + date_start.strftime("%Y-%m-%d") + input_end #date_start.strftime("%Y%m%d") + input_end 
    
    if not os.path.exists(fname) or not os.path.isfile(fname):
        print(input_init + date_start.strftime("%Y-%m-%d") + input_end)
        #print("https://oceandata.sci.gsfc.nasa.gov/cgi/getfile/" + os.path.basename(fname))

    #print("[\"" + input_init + date_start.strftime("%Y%m%d") + input_end + "\",\"" + trop_init + date_start.strftime("%Y-%m-%d") + trop_end + "\"],")
    date_start = date_start + datetime.timedelta(days=1) 



