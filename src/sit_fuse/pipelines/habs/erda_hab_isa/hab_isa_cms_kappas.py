# Purpose of this script is to batch-compare SIT FUSE severity classification rasters (.tif) across sensors, producing per-comparison confusion matrices (as CSVs), weighted Cohen's kappa scores, and confusion matrix heatmap plots.
#
# Structure of input folder:
#   parent_folder is expected to be organized like this:
#   parent_folder/sizeclass/sensor_name/*.tif
#
#   Each .tif under a given sizeclass/sensor_name folder represents one scene's pixel-wise severity classification (values 0-5) for that sensor. Two sensors are compared for a given sizeclass by matching .tif files with identical filenames
#
# Terminology in this script:
#   sizeclass = the phytoplankton size class / product being classified (top-level subfolder)
#
# There are two comparison modes in this script:
#   1. sensor vs. sensor (pairwise) - currently disabled, see the commented-out block below
#   2. sensor vs. all other sensors combined - currently active (default)
#   To switch modes, comment/uncomment the corresponding block


import os
import itertools
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score


# parent_folder path structure = parent_folder/sizeclass/sensorname/*.tif
parent_folder = r"D:\Users\rachel.jiang\Desktop\HAB\outputsbyspecies"

# output directories for csv results and confusion matrix plots
output_csv_dir = r"D:\Users\rachel.jiang\Desktop\HAB\csv_results_standardized_2"
output_plots_dir = r"D:\Users\rachel.jiang\Desktop\HAB\confusionmatrix_plots_standardized_2"
os.makedirs(output_csv_dir, exist_ok=True)
os.makedirs(output_plots_dir, exist_ok=True)

# class values that correspond to bloom severity (0 = not present to 5 = very high)
labels = [0, 1, 2, 3, 4, 5]



def get_confusion_matrix(raster_1_path, raster_2_path):
    """Read two severity rasters and return their aligned, valid pixel values.
        raster_1_path = path to the first sensor's raster
        raster_2_path = path to the second sensor's raster
    """
    with rasterio.open(raster_1_path) as dataset1:
        data1 = dataset1.read(1)
        nodata1 = dataset1.nodata
    with rasterio.open(raster_2_path) as dataset2:
        data2 = dataset2.read(1)
        nodata2 = dataset2.nodata

    # masking nodata
    valid_mask = (data1 > -1) & (data2 > -1)
    if nodata1 is not None:
        valid_mask &= data1 != nodata1
    if nodata2 is not None:
        valid_mask &= data2 != nodata2

    data1_flat = data1[valid_mask].flatten()
    data2_flat = data2[valid_mask].flatten()

    if data1_flat.size == 0:
        return None
    return data1_flat, data2_flat


def kappa_from_confusion_matrix(cm):
    """Quadratic-weighted Cohen's kappa from an aggregated confusion matrix."""
    cm = np.array(cm, dtype=np.int64)
    y_true, y_pred = [], []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            if count > 0:
                y_true.extend([labels[i]] * count)
                y_pred.extend([labels[j]] * count)
    if len(y_true) == 0:
        return None
    return cohen_kappa_score(y_true, y_pred, labels=labels, weights='quadratic')


def plot_confusion_matrix(cm, out_name, output_dir, title='Confusion matrix', cmap=plt.cm.coolwarm):
    """Render and save a confusion matrix heatmap with counts and row-percentages."""
    # convert the raw counts into %s (what % of each row falls into each column)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = np.divide(cm, row_sums, where=row_sums != 0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.matshow(cm_normalized, cmap=cmap, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel(sensor_1_name)
    plt.xlabel(sensor_2_name)
    plt.title(title)
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n({cm_normalized[i, j]:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color="white" if cm_normalized[i, j] > 50 else "black")

    plot_path = os.path.join(output_dir, out_name)
    fig.savefig(plot_path, dpi=150)
    fig.tight_layout()
    plt.close(fig)
    return plot_path


# loop through the sizeclass and sensor combinations
sizeclass_list = [f for f in os.listdir(parent_folder)
                   if os.path.isdir(os.path.join(parent_folder, f))]

# 
# MODE 1 (currently OFF): sensor vs. sensor, pairwise
# To use this mode instead of MODE 2 below: uncomment this block and comment out the "MODE 2" block that follows it.
#
"""
for sizeclass in sizeclass_list:
    sizeclass_folder = os.path.join(parent_folder, sizeclass)

    sensor_names = [f for f in os.listdir(sizeclass_folder)
                     if os.path.isdir(os.path.join(sizeclass_folder, f))]

    for sensor_1_name, sensor_2_name in itertools.combinations(sorted(sensor_names), 2):
        folder_1 = os.path.join(sizeclass_folder, sensor_1_name)
        folder_2 = os.path.join(sizeclass_folder, sensor_2_name)

        files_1 = {f for f in os.listdir(folder_1) if f.endswith(".tif")}
        files_2 = {f for f in os.listdir(folder_2) if f.endswith(".tif")}

        results = []
        results_total = None

        for filename in sorted(files_1.intersection(files_2)):
            raster_1 = os.path.join(folder_1, filename)
            raster_2 = os.path.join(folder_2, filename)

            result = get_confusion_matrix(raster_1, raster_2)
            if result is None:
                continue
            y_1, y_2 = result

            cm = confusion_matrix(y_1, y_2, labels=labels)
            results.append((filename, cm))

            if results_total is None:
                results_total = cm
            else:
                results_total = results_total + cm

        if results_total is None:
            print(f"Skipped, No valid scenes for {sensor_1_name} vs {sensor_2_name} ({sizeclass})")
            continue

        label = f"{sensor_1_name}_vs_{sensor_2_name}_{sizeclass}"

        # make a csv
        csv_path = os.path.join(output_csv_dir, f"{label}_confusionmatrix_results.csv")
        with open(csv_path, 'w') as f:
            f.write("Filename,ConfusionMatrix\n")
            for filename, cm in results:
                f.write(f"{filename},{cm}\n")
            f.write(f"TOTAL,{results_total}\n")

        kappa = kappa_from_confusion_matrix(results_total)

        print(f"Total confusion matrix: {sensor_1_name} vs. {sensor_2_name} ({sizeclass}):")
        print(results_total)
        print(f"Kappa: {kappa:.4f}\n" if kappa is not None else "Kappa: N/A\n")

        plot_title = f"{sensor_1_name} vs. {sensor_2_name}"
        if kappa is not None:
            plot_title += f" \nκ={kappa:.3f}"
            plot_title += f" \nOverlapping Dates: {len(files_1.intersection(files_2))}"

        plot_confusion_matrix(
            results_total,
            f"{label}_TOTAL_cm.png",
            output_plots_dir,
            title=plot_title
        )
"""

# MODE 2 (currently ON): sensor vs. all other sensors combined

for sizeclass in sizeclass_list:
    sizeclass_folder = os.path.join(parent_folder, sizeclass)

    sensor_names = [f for f in os.listdir(sizeclass_folder)
                     if os.path.isdir(os.path.join(sizeclass_folder, f))]

    sensor_names_sorted = sorted(sensor_names)

    for sensor_1_name in sensor_names_sorted:
        folder_1 = os.path.join(sizeclass_folder, sensor_1_name)
        files_1 = {f for f in os.listdir(folder_1) if f.endswith(".tif")}

        other_sensors = [s for s in sensor_names_sorted if s != sensor_1_name]

        results = []
        results_total = None
        overlap_count = 0

        for sensor_2_name in other_sensors:
            folder_2 = os.path.join(sizeclass_folder, sensor_2_name)
            files_2 = {f for f in os.listdir(folder_2) if f.endswith(".tif")}

            overlap_files = files_1.intersection(files_2)
            overlap_count += len(overlap_files)

            for filename in sorted(overlap_files):
                raster_1 = os.path.join(folder_1, filename)
                raster_2 = os.path.join(folder_2, filename)

                result = get_confusion_matrix(raster_1, raster_2)
                if result is None:
                    continue
                y_1, y_2 = result

                cm = confusion_matrix(y_1, y_2, labels=labels)
                results.append((filename, sensor_2_name, cm))

                if results_total is None:
                    results_total = cm
                else:
                    results_total = results_total + cm

        if results_total is None:
            print(f"Skipped, no valid scenes for {sensor_1_name} vs all others ({sizeclass})")
            continue

        label = f"{sensor_1_name}_vs_AllOthers_{sizeclass}"

        # make a csv
        csv_path = os.path.join(output_csv_dir, f"{label}_confusionmatrix_results.csv")
        with open(csv_path, 'w') as f:
            f.write("Filename,ComparedToSensor,ConfusionMatrix\n")
            for filename, other_sensor, cm in results:
                f.write(f"{filename},{other_sensor},{cm}\n")
            f.write(f"TOTAL,ALL,{results_total}\n")

        kappa = kappa_from_confusion_matrix(results_total)

        print(f"Total confusion matrix: {sensor_1_name} vs. ALL OTHERS ({sizeclass}):")
        print(results_total)
        print(f"Kappa: {kappa:.4f}\n" if kappa is not None else "Kappa: N/A\n")

        sensor_2_name = "All Other Sensors"

        plot_title = f"{sensor_1_name} vs. All Other Sensors"
        if kappa is not None:
            plot_title += f" \nκ={kappa:.4f}"
            plot_title += f" \nTotal Comparisons: {overlap_count}"

        plot_confusion_matrix(
            results_total,
            f"{label}_comparison_cm.png",
            output_plots_dir,
            title=plot_title
        )
