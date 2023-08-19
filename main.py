# MODELS=
import numpy as np

from src.Measurement import *
from src.VisualOdometry import *

# SETTINGS=
# ⌞ DATA=
input_dir = "src/data/input/"
output_dir = "src/data/output/"
datasets_paths = [os.path.join(input_dir,"kitti/", dataset_index) for dataset_index in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]]
# ⌞ METHOD=
methods = ["mono", "stereo"]
# ⌞ POST MATCHING OUTLIER REMOVAL (PMOR)=
do_PMOR = True
# ⌞ FEATURE DETECTION=
feature_detection_parameters = \
    {
        "method": ["surf", "fast", "orb"][2],
        "nfeatures": 3000
    }
# ⌞ SEMANTIC SEGMENTATION=
ss_parameters = \
    {
        "do_SS": False,
        "model_path": "src/models/deeplabv3_xception65_ade20k.h5",
        "features_to_ignore": ["sky", "person", "car"]
    }
# ⌞ FRAME TILE OPTIMIZATION (FTO)=
fto_parameters = \
    {
        "do_FTO": True,
        "grid_h": 40,
        "grid_w": 20,
        "patch_max_features": 10
    }
# ⌞ VIEW/MONITOR/SAVE PARAMETERS=
view = True
monitor = True
saveData = False


# MAIN:
def main():
    # FOR EACH DATASET:
    print("dataset,method,ss,fto,GRID_H,GRID_W,PATCH_MAX_FEATURES,ate,nc_ate,ate_percent,nc_ate_percent")
    for dataset_path in datasets_paths:
        for method in methods:
            # VISUAL ODOMETRY: Initialize the Visual Odometry class
            vo = VisualOdometry(
                data_dir=dataset_path,
                method=method,
                do_PMOR=do_PMOR,
                ss_parameters=ss_parameters,
                fto_parameters=fto_parameters
                )
            gt_path, est_path = vo.estimate_path(monitor=monitor, view=view)     # Estimate the path
            # MEASUREMENTS:
            ate, nc_ate, ate_percent, nc_ate_percent = get_ate(gt_path, est_path)    # Absolute Trajectory Error


if __name__ == "__main__":
    main()
