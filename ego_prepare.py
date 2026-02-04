from __future__ import print_function, division

import sys
import glob
import pandas as pd
import os

# =======================
# CONFIG
# =======================
path_to_dataset = "/home/mislab/Charlene/frames/"

paths = sorted(
    glob.glob(os.path.join("/home/mislab/Charlene/", "labels-final-revised1/*/*/*"))
)

# =======================
# SUBJECT LISTS
# =======================
subject_ids = ["{:02d}".format(i) for i in range(1, 51)]

subject_ids_train = ["{:02d}".format(i) for i in [
    3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23,
    25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44,
    45, 46, 48, 49, 50
]]

subject_ids_val = ["{:02d}".format(i) for i in [
    1, 7, 12, 13, 24, 29, 33, 34, 35, 37
]]

subject_ids_test = ["{:02d}".format(i) for i in [
    2, 9, 11, 14, 18, 19, 28, 31, 41, 47
]]


# =======================
# MAIN FUNCTION
# =======================
def create_trainlist(subset, file_name, class_types="all"):
    folder1 = "Color"
    folder2 = "rgb"

    if subset == "training":
        subjects_to_process = subject_ids_train
    elif subset == "validation":
        subjects_to_process = subject_ids_val
    elif subset == "testing":
        subjects_to_process = subject_ids_test
    else:
        raise ValueError("Subset must be training, validation, or testing")

    print("Preparing lines...")
    new_lines = []

    for path in paths:
        df = pd.read_csv(path, index_col=False, header=None)
        x = path.rsplit(os.sep, 4)
        subject = x[2]

        if subject[-2:] not in subjects_to_process:
            continue

        index = x[-1].split(".")[0][-1]
        folder_path = os.path.join(
            subject.title(),
            x[3],
            folder1,
            folder2 + index
        )

        full_path = os.path.join("/" + x[0], "images", folder_path)
        n_images = len(sorted(glob.glob(full_path + "/*")))

        df_val = df.values
        start = 1
        end = df_val[1, 1] - 1
        len_lines = df_val.shape[0]

        for i in range(len_lines):
            line = df_val[i, :]

            if class_types == "all":
                if (line[1] - start) >= 8:
                    new_lines.append(
                        f"{folder_path} 84 {start} {line[1] - 1}"
                    )
                new_lines.append(
                    f"{folder_path} {line[0]} {line[1]} {line[2]}"
                )

            elif class_types == "all_but_None":
                new_lines.append(
                    f"{folder_path} {line[0]} {line[1]} {line[2]}"
                )

            elif class_types == "binary":
                if (line[1] - start) >= 8:
                    new_lines.append(
                        f"{folder_path} 1 {start} {line[1] - 1}"
                    )
                new_lines.append(
                    f"{folder_path} 2 {line[1]} {line[2]}"
                )

            start = line[2] + 1

        if (n_images - start) > 8:
            if class_types == "all":
                new_lines.append(
                    f"{folder_path} 84 {start} {n_images}"
                )
            elif class_types == "binary":
                new_lines.append(
                    f"{folder_path} 1 {start} {n_images}"
                )

    print("Writing file...")
    os.makedirs("annotation_EgoGesture", exist_ok=True)
    file_path = os.path.join("annotation_EgoGesture", file_name)

    with open(file_path, "w") as f:
        for line in new_lines:
            f.write(line + "\n")

    print("Successfully wrote:", file_path)


# =======================
# ENTRY POINT
# =======================
if __name__ == "__main__":
    subset = sys.argv[1]
    file_name = sys.argv[2]
    class_types = sys.argv[3]

    create_trainlist(subset, file_name, class_types)

    # Example:
    # python ego_prepare.py training trainlistall.txt all
