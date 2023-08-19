# MODELS=
import numpy as np

from src.Measurement import *
from src.VisualOdometry import *

# SETTINGS=
# _ DATA=
input_dir = "src/data/input/kitti/"
skitti_indexes = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
kitti_indexes = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
dataset_indexes = skitti_indexes
# _ METHOD=
methods = ["mono"]
# _ VIEW/MONITOR/SAVE PARAMETERS=
view = True
monitor = True
saveData = False

# MAIN:
def main():
    # FOR EACH DATASET:
    print("dataset,method,pmor,ate,nc_ate,ate_percent,nc_ate_percent")
    for pmor in [False, True]:
        for dataset_path in [os.path.join(input_dir, dataset_index) for dataset_index in dataset_indexes]:
            for method in methods:
                try:
                    # VISUAL ODOMETRY: Initialize the Visual Odometry class
                    vo = VisualOdometry(
                        dataset_path,
                        method=method,
                        do_PMOR=pmor
                        )
                    gt_path, est_path = vo.estimate_path(monitor=monitor, view=view)     # Estimate the path
                    # MEASUREMENTS:
                    ate, nc_ate, ate_percent, nc_ate_percent = get_ate(gt_path, est_path)    # Absolute Trajectory Error
                    # PRINT:
                    print(f"{dataset_path.split('/')[-1]},{method},{pmor},{ate},{nc_ate},{ate_percent},{nc_ate_percent}")
                except:
                    print(f"{dataset_path.split('/')[-1]},{method},{pmor},CRASHED,CRASHED,CRASHED,CRASHED")


if __name__ == "__main__":
    main()
