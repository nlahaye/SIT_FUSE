import datetime


input_init = "/data/nlahaye/remoteSensing/Multi_Year_WQ/JPSS1_VIIRS."
input_end = ".L3m.DAY."

trop_init = "/data/nlahaye/remoteSensing/TROPOMI_MODIS_HAB/TROPO_redSIF_"
trop_end = "_ungridded.nc"



date_start = datetime.datetime.strptime("20180601", "%Y%m%d")
date_end = datetime.datetime.strptime("20191231", "%Y%m%d")

while date_start <= date_end:
    print("[\"" + input_init + date_start.strftime("%Y%m%d") + input_end + "\",\"" + trop_init + date_start.strftime("%Y-%m-%d") + trop_end + "\"],")
    date_start = date_start + datetime.timedelta(days=1) 



