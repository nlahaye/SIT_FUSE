
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from utils import read_mspi 

data_files = ["/data/nlahaye/remoteSensing/MSPI/2019-08-06/181938Z_WA-Keller/AirMSPI_ER2_GRP_ELLIPSOID_20190806_181938Z_WA-Keller_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/200215Z_WA-Valley/AirMSPI_ER2_GRP_ELLIPSOID_20190806_200215Z_WA-Valley_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-07/205008Z_WA-Mansfield/AirMSPI_ER2_GRP_ELLIPSOID_20190807_205008Z_WA-Mansfield_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/200125Z_WA-Valley/AirMSPI_ER2_GRP_ELLIPSOID_20190806_200125Z_WA-Valley_SWPA_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/205524Z_WA-ElectricCity/AirMSPI_ER2_GRP_ELLIPSOID_20190806_205524Z_WA-ElectricCity_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-07/193122Z_WA-Ford/AirMSPI_ER2_GRP_ELLIPSOID_20190807_193122Z_WA-Ford_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-07/193951Z_WA-Mansfield/AirMSPI_ER2_GRP_ELLIPSOID_20190807_193951Z_WA-Mansfield_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-07/200625Z_WA-Ford/AirMSPI_ER2_GRP_ELLIPSOID_20190807_200625Z_WA-Ford_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-08/182451Z_ID-Athol/AirMSPI_ER2_GRP_ELLIPSOID_20190808_182451Z_ID-Athol_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/212231Z_WA-CouleeCity/AirMSPI_ER2_GRP_ELLIPSOID_20190806_212231Z_WA-CouleeCity_SWPF_F01_V006.hdf"
"/data/nlahaye/remoteSensing/MSPI/2019-08-07/193122Z_WA-Ford/AirMSPI_ER2_GRP_ELLIPSOID_20190807_193122Z_WA-Ford_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/190830Z_WA-Republic/AirMSPI_ER2_GRP_ELLIPSOID_20190806_190830Z_WA-Republic_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-08/182400Z_ID-Rathdrum/AirMSPI_ER2_GRP_ELLIPSOID_20190808_182400Z_ID-Rathdrum_SWPA_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/203638Z_WA-Valley/AirMSPI_ER2_GRP_ELLIPSOID_20190806_203638Z_WA-Valley_SWPA_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/193440Z_WA-ElectricCity/AirMSPI_ER2_GRP_ELLIPSOID_20190806_193440Z_WA-ElectricCity_SWPF_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/205433Z_WA-CouleeCity/AirMSPI_ER2_GRP_ELLIPSOID_20190806_205433Z_WA-CouleeCity_SWPA_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/182241Z_WA-Keller/AirMSPI_ER2_GRP_ELLIPSOID_20190806_182241Z_WA-Keller_SWPA_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-06/192738Z_WA-Davenport/AirMSPI_ER2_GRP_ELLIPSOID_20190806_192738Z_WA-Davenport_SWPA_F01_V006.hdf",
"/data/nlahaye/remoteSensing/MSPI/2019-08-07/182311Z_WA-GrandCoulee/AirMSPI_ER2_GRP_ELLIPSOID_20190807_182311Z_WA-GrandCoulee_SWPF_F01_V006.hdf"]

 

 
for i in range(len(data_files)):
    filename = data_files[i]
    data = read_mspi(filename)
 
    print(data.shape)
    for c in range(data.shape[0]):
        mask_gray = cv.normalize(src=data[c,100:-100,100:-100], dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        equ = cv.equalizeHist(mask_gray)
        plt.imshow(equ)
        plt.savefig("MSPI_IMAGE_" + str(i) + "C" + str(c) + ".png")



