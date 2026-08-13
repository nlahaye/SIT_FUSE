# Purpose of this script is to bin in-situ observations into classes (0-5) and plot how many samples fall into each bin, phytoplankton cell-counts (seriata, delicatissima, total) and domoic acid (pDA).
#
# Notes on input CSV file:
#   CSV_PATH is expected to be a combined in-situ dataset with one row per sample and columns matching the names listed in CELL_COUNT_COLUMNS and DA_COLUMN below.
#
# Terminology in this script:
#   cell count = phytoplankton concentration, measured in cells/L
#   pDA        = particulate domoic acid concentration, measured in ng/L


import os
import pandas as pd
import matplotlib.pyplot as plt

#file path
CSV_PATH = r"Z:\2026\Summer\Southern California Bight HABs\in_situ_nick_data\combined_full.csv"
OUTPUT_DIR = r"D:\Users\rachel.jiang\Desktop\HAB"

#standardize y axis max
Y_AXIS_MAX = 1500

#phytoplankton cell counts (measured in cells/L)
CELL_COUNT_COLUMNS = [
    "Pseudo_nitzschia_delicatissima_group",
    "Pseudo_nitzschia_seriata_group",
    "Total_Phytoplankton",
]

#domoic acid (measured in ng/L)
DA_COLUMN = "pDA"

# EDIT THESE if you want to change the bin thresholds
CELL_CLASS_NAMES = {
    -1: "N/A (no data)",
    0: "Not Present (<1,000 cells/L)",
    1: "Very Low (1,000-10,000 cells/L)",
    2: "Low (10,000-100,000 cells/L)",
    3: "Medium (100,000-1,000,000 cells/L)",
    4: "High (1,000,000-10,000,000 cells/L)",
    5: "Very High (>=10,000,000 cells/L)",
}

DA_CLASS_NAMES = {
    -1: "N/A (no data)",
    0: "Not Present (0 ng/L)",
    1: "Very Low (0-0.1 ng/L)",
    2: "Low (0.1-1 ng/L)",
    3: "Medium (1-10 ng/L)",
    4: "High (10-100 ng/L)",
    5: "Very High (100-1000 ng/L)",
}


def classify_cell_count(value):
    """Classify a single cells/L measurement into a severity bin."""
    if pd.isna(value):
        return -1
    elif value < 1000:
        return 0
    elif value < 10000:
        return 1
    elif value < 100000:
        return 2
    elif value < 1000000:
        return 3
    elif value < 10000000:
        return 4
    else:
        return 5


def classify_da(value):
    """Classify a single pDA (ng/L) measurement into a severity bin."""
    if pd.isna(value):
        return -1
    elif value < 0.1:
        return 0
    elif value < 1:
        return 1
    elif value < 10:
        return 2
    elif value < 100:
        return 3
    elif value < 1000:
        return 4
    else:
        return 5


def count_samples_per_class(class_numbers, class_names):
    """Count how many samples fall into each severity bin."""
    labels = []
    counts = []
    for class_number in sorted(class_names.keys()):
        label = f"{class_number}: {class_names[class_number]}"
        count = (class_numbers == class_number).sum()
        labels.append(label)
        counts.append(count)
    return labels, counts


def make_histogram(labels, counts, title, save_path):
    """Draw and save bar-chart histogram of sample counts per severity bin."""
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, counts, color="steelblue")

    # write the count above each bar
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.ylim(0, Y_AXIS_MAX)
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print("Saved plot:", save_path)


def main() -> None:
    """Main function for classifying and plotting."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    for column_name in CELL_COUNT_COLUMNS:
        if column_name not in df.columns:
            print("column not found, skipping:", column_name)
            continue

        class_column_name = column_name + "_Class"
        df[class_column_name] = df[column_name].apply(classify_cell_count)

        labels, counts = count_samples_per_class(df[class_column_name], CELL_CLASS_NAMES)

        # plot and save the histogram
        save_path = os.path.join(OUTPUT_DIR, column_name + "_histogram.png")
        make_histogram(labels, counts, column_name + " - Class Distribution", save_path)

    # for pDA histogram
    if DA_COLUMN in df.columns:
        class_column_name = DA_COLUMN + "_Class"
        df[class_column_name] = df[DA_COLUMN].apply(classify_da)

        labels, counts = count_samples_per_class(df[class_column_name], DA_CLASS_NAMES)

        save_path = os.path.join(OUTPUT_DIR, DA_COLUMN + "_histogram.png")
        make_histogram(labels, counts, DA_COLUMN + " - Class Distribution", save_path)
    else:
        print("column not found, skipping:", DA_COLUMN)

    #save the full table with the new class columns added
    out_csv = os.path.join(OUTPUT_DIR, "classified_data_new.csv")
    df.to_csv(out_csv, index=False)
    print("Saved classified data:", out_csv)


if __name__ == "__main__":
    main()
