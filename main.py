# IMPORTS=
import time
import matplotlib.pyplot as plt
import numpy as np

# MODELS=
from src.Measurement import *
from src.VisualOdometry import *

# DATA=
input_dir = "src/data/input/kitti/"
skitti_indexes = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
kitti_indexes = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
dataset_indexes = skitti_indexes

# SETTINGS=
methods = ["mono", "stereo"][0:1]
features = ["earth", "grass", "sidewalk", "road", "building"].clear()
showMatches = False
saveData = False

# MAIN:
def main():
    # SAVE DATA:
    xy_values_csv = "dataset, method, x_final_diff, y_final_diff, x_mean_diff, y_mean_diff, x_max_diff, y_max_diff\n"
    # FOR EACH DATASET:∂
    for dataset_path in [os.path.join(input_dir, dataset_index) for dataset_index in dataset_indexes]:
        for method in methods:
            # VISUAL ODOMETRY:
            vo = VisualOdometry(dataset_path, method=method)   # Initialize the Visual Odometry class
            gt_path, est_path = vo.estimate_path(show_matches=showMatches, features=features)     # Estimate the path
            # MEASUREMENTS:
            x_final_diff, y_final_diff, x_mean_diff, y_mean_diff, x_max_diff, y_max_diff = get_xy_values(gt_path, est_path, print_values=True)  # Get the measurements
            # SAVE DATA:
            xy_values_csv += dataset_path.split("/")[-1] + ", " + method + ", " + str(x_final_diff) + ", " + str(y_final_diff) + ", " + str(x_mean_diff) + ", " + str(y_mean_diff) + ", " + str(x_max_diff) + ", " + str(y_max_diff) + "\n"

        # Create an array in which each element is a tuple of the non cumulative error of both x and y
        xy_values = [[est_path[i][0] - gt_path[i][0] - est_path[i - 1][0] + gt_path[i - 1][0],
                      est_path[i][1] - gt_path[i][1] - est_path[i - 1][1] + gt_path[i - 1][1]] for i in
                     range(1, len(est_path))]
        # 2D PLOT OF THE ERROR OF X AND Y (LINK CONSECUTIVE POINTS):
        plt.plot([xy_value[0] for xy_value in xy_values], [xy_value[1] for xy_value in xy_values], 'o-')
        plt.xlabel("x error")
        plt.ylabel("y error")
        plt.show()

    # SAVE CSV:
    if saveData:
        with open("src/data/output/" + time.strftime("%Y-%m-%d_%H-%M") + ".csv", "w") as f:
            f.write(xy_values_csv)

if __name__ == "__main__":
    main()
