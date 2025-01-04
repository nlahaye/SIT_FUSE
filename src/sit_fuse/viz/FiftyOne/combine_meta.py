
import csv
import os
import re 

no_heir_labels = "/data/nlahaye/output/Learnergy/GENETICS_POC_COMBINED/genetics_poc_combined_no_heir_wass.npy.csv"
heir_labels = "/data/nlahaye/output/Learnergy/GENETICS_POC_COMBINED/genetics_poc_combined_wass.npy.csv"
csv_file = "/data/nlahaye/output/Learnergy/GENETICS_POC_COMBINED/genetics_combined_labels_poc_combined.csv"
data_labels = "/data/nlahaye/remoteSensing/Genetics/train/meta.csv" 
#"/data/nlahaye/remoteSensing/Genetics/train/meta_DAPI_MASK.csv"
track_labels = "/data/nlahaye/remoteSensing/Genetics/train/Tracks-NoTracksLabels.csv"

fname_re = r'(.*)_proj\.tif'


full_dict = {} #[data_fnames[i], labels[i]]

def csv_to_list(fname):
    data = []
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data
 

data = csv_to_list(no_heir_labels)
for i in range(len(data)):
    zarr_fname = data[i][0]
    no_heir_label = data[i][1]


    key = (re.search(fname_re, os.path.basename(zarr_fname)).groups(1))[0]
    print(key)
    if key not in full_dict:
        full_dict[key] = [key, no_heir_label, -1, -1, -1, -1, -1]

    else:
        full_dict[key][1] = no_heir_label

              
data = csv_to_list(heir_labels)
for i in range(len(data)):
    zarr_fname = data[i][0]
    heir_label = data[i][1]
 
    key = (re.search(fname_re, os.path.basename(zarr_fname)).groups(1))[0]
    print(key)
    if key not in full_dict:
        full_dict[key] = [key, -1, heir_label, -1, -1, -1, -1]
    else:
        full_dict[key][2] = heir_label



data = csv_to_list(data_labels)
for i in range(1, len(data)):
    fname = data[i][0]
    dose = data[i][1]
    part_type = data[i][2]
    hr_post = data[i][3]

    key = (re.search(fname_re, os.path.basename(fname)).groups(1))[0]
    fpath = os.path.join(os.path.dirname(data_labels), fname)
    print(key)
    if key not in full_dict:
        full_dict[key] = [fpath, -1, -1, -1, hr_post, part_type, dose] 
    else:
        full_dict[key][0] = fpath
        full_dict[key][4] = hr_post
        full_dict[key][5] = part_type
        full_dict[key][6] = dose


 
data = csv_to_list(track_labels)
for i in range(1, len(data)):
    fname = data[i][0]
    tracks = 0 if data[i][1] == "NoTracks" else 1

    key = (re.search(fname_re, os.path.basename(fname)).groups(1))[0]
    print(key)
    fpath = os.path.join(os.path.dirname(data_labels), fname)
    if key not in full_dict:
        full_dict[key] = [fpath, -1, -1, tracks, -1, -1, -1]
    else:
        full_dict[key][3] = tracks




with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the data rows (values of the dictionary)
        writer.writerows([["filepath", "label_seg_1", "label_seg_2", "label_tracks", "label_hr_post", "label_type", "label_dose"]])
        writer.writerows(full_dict.values())




