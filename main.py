# MODELS=
import numpy as np

from src.Measurement import *
from src.VisualOdometry import *

# SETTINGS=
# ⌞ DATA=
input_dir = "src/data/input/kitti/"
skitti_indexes = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
kitti_indexes = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
dataset_indexes = skitti_indexes
# ⌞ METHOD=
methods = ["mono", "stereo"][1:2]
# ⌞ SEMANTIC SEGMENTATION=
do_SS = False
model_path = "src/models/deeplabv3_xception65_ade20k.h5"
features = ["earth", "grass", "sidewalk", "road", "building"]
# ⌞ FRAME TILE OPTIMIZATION (FTO)=
do_FTO = True
GRID_H = 10
GRID_W = 20
PATCH_MAX_FEATURES = 10
# ⌞ VIEW=
monitor = False
view = True
saveData = False

# MAIN:
def main():
    # FOR EACH DATASET:
    print("dataset,method,ss,fto,GRID_H,GRID_W,PATCH_MAX_FEATURES,ate,nc_ate")
    for dataset_path in [os.path.join(input_dir, dataset_index) for dataset_index in dataset_indexes]:
        for method in methods:
            # VISUAL ODOMETRY: Initialize the Visual Odometry class
            vo = VisualOdometry(dataset_path,
                                method=method,
                                semantic_segmentation_parameters={
                                    "segmentate": do_SS,
                                    "model_path": model_path,
                                    "features": features
                                },
                                fto_parameters={
                                    "do_FTO": do_FTO,
                                    "grid_h": GRID_H,
                                    "grid_w": GRID_W,
                                    "patch_max_features": PATCH_MAX_FEATURES
                                }
                                )
            gt_path, est_path = vo.estimate_path(monitor=monitor, view=view)     # Estimate the path
            # MEASUREMENTS:
            ate, nc_ate = get_ate(gt_path, est_path)    # Absolute Trajectory Error
            print(f"{dataset_path.split('/')[-1]},{method},{do_SS},{do_FTO},{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{ate},{nc_ate}")


if __name__ == "__main__":
    main()
