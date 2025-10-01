#!/bin/bash
 
python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/modis_only_s_ca_multi_layer_pl.yaml
mkdir ww/modis_only_s_ca
mv img/* ww/modis_only_s_ca/
mv ww-img/* ww/modis_only_s_ca/
 
python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/modis_only_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/modis_only_gom/
mv img/* ww/modis_only_gom/
mv ww-img/* ww/modis_only_gom/

python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/modis_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/modis_sif_gom/
mv img/* ww/modis_sif_gom/
mv ww-img/* ww/modis_sif_gom/

python3 ww_test.py  -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/modis_s_ca_multi_layer_pl.yaml
mkdir ww/modis_sif_s_ca/
mv img/* ww/modis_sif_s_ca/
mv ww-img/* ww/modis_sif_s_ca/



python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/jpss1_viirs_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/jpss1_sif_gom/
mv img/* ww/jpss1_sif_gom/
mv ww-img/* ww/jpss1_sif_gom/


python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/jpss1_viirs_only_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/jpss1_only_gom/
mv img/* ww/jpss1_only_gom/
mv ww-img/* ww/jpss1_only_gom/


python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/jpss1_viirs_only_s_ca_multi_layer_pl.yaml
mkdir ww/jpss1_only_s_ca/
mv img/* ww/jpss1_only_s_ca/
mv ww-img/* ww/jpss1_only_s_ca/


python3 ww_test.py  -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/jpss1_viirs_s_ca_multi_layer_pl.yaml
mkdir ww/jpss1_sif_s_ca/
mv img/* ww/jpss1_sif_s_ca/
mv ww-img/* ww/jpss1_sif_s_ca/

python3 ww_test.py  -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/viirs_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/snpp_viirs_sif_gom/
mv img/* ww/snpp_viirs_sif_gom/
mv ww-img/* ww/snpp_viirs_sif_gom/
 
python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/viirs_only_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/snpp_viirs_only_gom/
mv img/* ww/snpp_viirs_only_gom/
mv ww-img/* ww/snpp_viirs_only_gom/

python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/viirs_only_s_ca_multi_layer_pl.yaml
mkdir ww/snpp_viirs_only_s_ca/
mv img/* ww/snpp_viirs_only_s_ca/
mv ww-img/* ww/snpp_viirs_only_s_ca/

python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model/viirs_s_ca_multi_layer_pl.yaml
mkdir ww/snpp_viirs_s_ca/
mv img/* ww/snpp_viirs_s_ca/
mv ww-img/* ww/snpp_viirs_s_ca/ 

python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//pace_only_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/pace_only_gom/
mv img/* ww/pace_only_gom/
mv ww-img/* ww/pace_only_gom/
 
python3 ww_test.py  -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//pace_only_gulf_of_mexico_pca_s_ca_embed.yaml
mkdir ww/pace_only_s_ca/
mv img/* ww/pace_only_s_ca/
mv ww-img/* ww/pace_only_s_ca/


python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3a_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/s3a_sif_gom/
mv img/* ww/s3a_sif_gom/
mv ww-img/* ww/s3a_sif_gom/

python3 ww_test.py  -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3a_only_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/s3a_only_gom/
mv img/* ww/s3a_only_gom/
mv ww-img/* ww/s3a_only_gom/

python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3a_only_s_cal_multi_layer_pl.yaml
mkdir ww/s3a_only_s_ca/
mv img/* ww/s3a_only_s_ca/
mv ww-img/* ww/s3a_only_s_ca/ 


python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3a_s_cal_multi_layer_pl.yaml
mkdir ww/s3a_sif_s_ca/
mv img/* ww/s3a_sif_s_ca/
mv ww-img/* ww/s3a_sif_s_ca/


python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3b_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/s3b_sif_gom/
mv img/* ww/s3b_sif_gom/
mv ww-img/* ww/s3b_sif_gom/

python3 ww_test.py  -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3b_only_gulf_of_mexico_multi_layer_pl.yaml 
mkdir ww/s3b_only_gom/
mv img/* ww/s3b_only_gom/
mv ww-img/* ww/s3b_only_gom/

python3 ww_test.py   -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3b_only_s_cal_multi_layer_pl.yaml
mkdir ww/s3b_only_s_ca/
mv img/* ww/s3b_only_s_ca/
mv ww-img/* ww/s3b_only_s_ca/


python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3b_s_cal_multi_layer_pl.yaml
mkdir ww/s3b_sif_s_ca/
mv img/* ww/s3b_sif_s_ca/
mv ww-img/* ww/s3b_sif_s_ca/



