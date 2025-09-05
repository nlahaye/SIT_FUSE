
import json
import os
import copy
import argparse
from sit_fuse.utils import read_yaml

entry = {"PARAMETERS": {"INPUT_RASTER": "", "RASTER_BAND": "1", "INPUT_VECTOR": "", "COLUMN_PREFIX": "''"}, "OUTPUTS": {"OUTPUT": ""}}

input_vector_str_1 = "QgsProcessingFeatureSourceDefinition('"
input_vector_str_2 = "', selectedFeaturesOnly=False, featureLimit=-1, flags=QgsProcessingFeatureSourceDefinition.FlagOverrideDefaultGeometryCheck, geometryCheck=QgsFeatureRequest.GeometryNoCheck)"

 
def main(yml_fpath):

    #Translate config to dictionary·
    yml_conf = read_yaml(yml_fpath)
    #Run·
    shps = yml_conf["shp_files"]
    files = yml_conf["input_files"]
    out_file = yml_conf["out_file"]

    entries = []
 
    for i in range(len(shps)):
        for j in range(len(files[i])):
            for k in range(len(shps[i])):
                ent = copy.deepcopy(entry)
                ent["PARAMETERS"]["INPUT_RASTER"] = files[i][j]
                ent["PARAMETERS"]["INPUT_VECTOR"] = input_vector_str_1 + shps[i][k] + input_vector_str_2
                ent["OUTPUTS"]["OUTPUT"] = os.path.splitext(shps[i][k])[0] + "_OUTPUT_" + str(j) + ".shp"
                entries.append(ent)

 

    with open(out_file, 'w') as file:
        json.dump(entries, file)



if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument("-y", "--yaml", help="YAML file for DBN and output config.")
     args = parser.parse_args()
     main(args.yaml)


