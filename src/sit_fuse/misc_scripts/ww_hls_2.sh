#!/bin/bash
 

python3 ww_test.py -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//pace_only_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/pace_only_gom/
mv img/* ww/pace_only_gom/
mv ww-img/* ww/pace_only_gom/

python3 ww_test.py  -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//pace_only_gulf_of_mexico_pca_s_ca_embed.yaml
mkdir ww/pace_only_s_ca/
mv img/* ww/pace_only_s_ca/
mv ww-img/* ww/pace_only_s_ca/

python3 ww_test.py  -y /home/nlahaye/SIT_FUSE/src/sit_fuse/config/model//s3a_only_gulf_of_mexico_multi_layer_pl.yaml
mkdir ww/s3a_only_gom/
mv img/* ww/s3a_only_gom/
mv ww-img/* ww/s3a_only_gom/



