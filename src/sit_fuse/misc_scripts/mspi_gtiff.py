from osgeo import gdal
import numpy as np




fnames = [["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_183632Z_WA-Keller_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_183726Z_WA-Keller_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_183815Z_WA-Keller_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_183904Z_WA-Keller_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_184044Z_WA-Keller_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_184133Z_WA-Keller_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_184223Z_WA-Keller_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_184317Z_WA-Keller_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_190054Z_WA-Keller_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_190147Z_WA-Keller_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_190237Z_WA-Keller_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_190326Z_WA-Keller_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_190506Z_WA-Keller_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_190555Z_WA-Keller_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_190645Z_WA-Keller_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_190738Z_WA-Keller_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_194520Z_WA-Keller_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_194614Z_WA-Keller_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_194704Z_WA-Keller_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_194753Z_WA-Keller_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_194932Z_WA-Keller_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_195021Z_WA-Keller_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_195112Z_WA-Keller_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_195205Z_WA-Keller_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_200308Z_WA-Keller_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_200401Z_WA-Keller_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_200451Z_WA-Keller_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_200540Z_WA-Keller_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_200720Z_WA-Keller_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_200809Z_WA-Keller_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_200859Z_WA-Keller_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_200953Z_WA-Keller_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_202139Z_WA-Creston_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_202233Z_WA-Creston_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_202322Z_WA-Creston_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_202411Z_WA-Creston_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_202551Z_WA-Creston_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_202640Z_WA-Creston_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_202730Z_WA-Creston_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_202824Z_WA-Creston_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_203821Z_WA-Keller_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_203914Z_WA-Keller_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_204004Z_WA-Keller_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_204053Z_WA-Keller_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_204233Z_WA-Keller_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_204322Z_WA-Keller_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_204412Z_WA-Keller_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_204506Z_WA-Keller_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_205617Z_WA-Wilbur_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_205710Z_WA-Wilbur_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_205800Z_WA-Wilbur_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_205849Z_WA-Wilbur_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_210029Z_WA-Wilbur_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_210118Z_WA-Wilbur_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_210208Z_WA-Wilbur_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_210301Z_WA-Wilbur_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_211454Z_WA-Wilbur_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_211547Z_WA-Wilbur_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_211637Z_WA-Wilbur_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_211726Z_WA-Wilbur_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_211906Z_WA-Wilbur_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_211955Z_WA-Wilbur_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_212045Z_WA-Wilbur_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190806_212139Z_WA-Wilbur_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_182404Z_WA-Inchelium_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_182457Z_WA-Inchelium_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_182547Z_WA-Inchelium_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_182636Z_WA-Inchelium_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_182816Z_WA-Inchelium_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_182905Z_WA-Inchelium_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_182955Z_WA-Inchelium_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_183048Z_WA-Inchelium_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_193214Z_WA-Inchelium_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_193308Z_WA-Inchelium_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_193358Z_WA-Inchelium_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_193447Z_WA-Inchelium_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_193626Z_WA-Inchelium_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_193715Z_WA-Inchelium_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_193805Z_WA-Inchelium_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_193859Z_WA-Inchelium_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_195226Z_WA-Inchelium_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_195319Z_WA-Inchelium_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_195409Z_WA-Inchelium_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_195458Z_WA-Inchelium_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_195638Z_WA-Inchelium_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_195727Z_WA-Inchelium_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_195817Z_WA-Inchelium_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_195911Z_WA-Inchelium_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_200717Z_WA-Lincoln_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_200811Z_WA-Lincoln_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_200901Z_WA-Lincoln_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_200950Z_WA-Lincoln_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_201129Z_WA-Lincoln_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_201218Z_WA-Lincoln_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_201308Z_WA-Lincoln_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_201402Z_WA-Lincoln_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_202502Z_WA-Inchelium_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_202556Z_WA-Inchelium_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_202645Z_WA-Inchelium_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_202734Z_WA-Inchelium_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_202914Z_WA-Inchelium_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_203003Z_WA-Inchelium_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_203053Z_WA-Inchelium_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_203147Z_WA-Inchelium_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_204231Z_WA-Inchelium_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_204325Z_WA-Inchelium_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_204415Z_WA-Inchelium_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_204504Z_WA-Inchelium_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_204643Z_WA-Inchelium_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_204732Z_WA-Inchelium_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_204822Z_WA-Inchelium_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190807_204916Z_WA-Inchelium_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_174846Z_WA-Connell_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_174939Z_WA-Connell_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_175029Z_WA-Connell_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_175118Z_WA-Connell_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_175258Z_WA-Connell_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_175347Z_WA-Connell_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_175437Z_WA-Connell_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_175530Z_WA-Connell_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_181151Z_WA-Inchelium_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_181244Z_WA-Inchelium_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_181334Z_WA-Inchelium_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_181423Z_WA-Inchelium_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_181603Z_WA-Inchelium_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_181652Z_WA-Inchelium_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_181742Z_WA-Inchelium_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190808_181835Z_WA-Inchelium_645A_F01_V006.tif"],
["/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190821_224247Z_AZ-Flagstaff_645F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190821_224341Z_AZ-Flagstaff_571F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190821_224431Z_AZ-Flagstaff_458F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190821_224520Z_AZ-Flagstaff_274F_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190821_224700Z_AZ-Flagstaff_274A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190821_224749Z_AZ-Flagstaff_458A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190821_224839Z_AZ-Flagstaff_571A_F01_V006.tif",
"/data/nlahaye/remoteSensing/MSPI/step_and_stare/AirMSPI_ER2_GRP_TERRAIN_20190821_224932Z_AZ-Flagstaff_645A_F01_V006.tif"]]



out_dir = "/data/nlahaye/remoteSensing/MSPI/step_and_stare/"



for i in range(len(fnames)):
    dat_full = None
    for j in range(len(fnames[i])):
        dat = gdal.Open(fnames[i][j])
        arr = dat.ReadAsArray()
        print(arr.shape)
        if j == 0:
            dat_full = arr[6:,:,:]
        elif j == 3:
            dat_full = np.concatenate((dat_full,arr), axis=0)
        else:
            dat_full = np.concatenate((dat_full,arr[6:,:,:]), axis=0)
        print(dat_full.shape)


    fname = (fnames[i][0]).replace("645F_", "")
    dat = gdal.Open(fnames[i][0])     
    metadata = dat.GetMetadata()
    geoTransform = dat.GetGeoTransform()
    wkt = dat.GetProjection()

    dat_full = np.array(dat_full)
    print(dat_full.shape)
    out_ds = gdal.GetDriverByName("GTiff").Create(fname, dat_full.shape[2], dat_full.shape[1], dat_full.shape[0], gdal.GDT_Float32)

    out_ds.SetMetadata(metadata)
    out_ds.SetGeoTransform(geoTransform)
    out_ds.SetProjection(wkt)

    print(fname)
    for j in range(dat_full.shape[0]):
        out_ds.GetRasterBand((j+1)).WriteArray(dat_full[j,:,:])
    out_ds.FlushCache()
    out_ds = None
    del dat_full





