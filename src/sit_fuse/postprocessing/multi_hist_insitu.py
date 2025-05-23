


"""
Copyright [2022-23], by the California Institute of Technology and Chapman University. 
ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the 
Office of Technology Transfer at the California Institute of Technology and Chapman University.
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all 
applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be 
required before exporting such information to foreign countries or providing access to foreign persons.
"""

import numpy as np
from sit_fuse.utils import numpy_to_torch, read_yaml, insitu_hab_to_multi_hist
from osgeo import gdal, osr
import argparse
import os
import pytz
from pandas import DataFrame as df
from skimage.util import view_as_windows
from copy import deepcopy
import datetime





def build_lookup_dict(labels, lookup):

    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if (str(labels[i][j]) not in lookup) or (lookup[str(labels[i][j])] < i):
                lookup[str(labels[i][j])] = i
    print(lookup)
    return lookup

def build_final_list(lookup, final_lst):
    for key in lookup.keys():
        final_lst[lookup[key]].append(float(key))
    for i in range(len(final_lst)):
        final_lst[i] = sorted(final_lst[i])

 
    return final_lst
    

def main(yml_fpath):

    #Translate config to dictionary 
    yml_conf = read_yaml(yml_fpath)
    #Run 
    start_date = datetime.datetime.strptime( yml_conf['start_date'], '%Y-%m-%d') #.replace(tzinfo=datetime.timezone.utc)#.tz_localize(None)
    end_date = datetime.datetime.strptime( yml_conf['end_date'], '%Y-%m-%d') #.replace(tzinfo=datetime.timezone.utc) #.tz_localize(None)

    #start_date = pytz.utc.localize(start_date)
    #end_date = pytz.utc.localize(end_date)

    karenia = False
    if 'Karenia' in yml_conf['xl_fname'] or 'karenia' in yml_conf['xl_fname']:
        karenia = True


    arr_tmp = [[] for x in range(0,(len( yml_conf['ranges'])-1))] 
    labels = []
 
    use_key  = 'Total_Phytoplankton'
    if 'use_key' in yml_conf:
        use_key = yml_conf['use_key']
    lookup = {}
    final_lst = []
    for i in range(len(yml_conf["ranges"])):
        final_lst.append([])
    for i in range(len(yml_conf['radius_degrees'])):
        d_lower = (i > 0)
        labels.append(insitu_hab_to_multi_hist(yml_conf['xl_fname'], start_date, end_date,
    		yml_conf['clusters_dir'], yml_conf['clusters'], yml_conf['radius_degrees'][i],
                    yml_conf['ranges'], yml_conf['global_max'], yml_conf['input_file_type'], karenia, discard_lower = d_lower, use_key = use_key)) #, lookup = lookup))

        #lookup = build_lookup_dict(labels[-1], lookup)


    #final_lst = build_final_list(lookup, final_lst)


    arr_tmp = [[] for x in range(0,(len( yml_conf['ranges'])-1))]
    arr_init = labels[0]
     
    for n in range(1, len(labels)):
        for i in range(len(arr_init)):
            for j in range(len(labels[n][i])):
                ind = -1
                for k in range(len(arr_init)):
                    if labels[n][i][j] in arr_init[k]:
                        ind = k
                if ind < i:
                    arr_tmp[i].append(labels[n][i][j])

        for i in range(len(arr_init)):
            for j in range(len(arr_init[i])):
                ind = -1
                for k in range(len(arr_init)):
                    if arr_init[i][j] in arr_tmp[k]:
                        ind = k  
                if ind < 0:
                    arr_tmp[i].append(arr_init[i][j])

        arr_init = deepcopy(arr_tmp)
        arr_tmp = [[] for x in range(0,(len( yml_conf['ranges'])-1))]
    

    for i in range(len(arr_init)):
        arr_init[i] = sorted(arr_init[i])  
    print(arr_init)
    #print(labels)
    #print(final_lst)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML file for fusion info.")
    args = parser.parse_args()
    main(args.yaml)





