# IMPORTS:
from datetime import *
# MODELS:
from src.Measurement import *
from src.VisualOdometry import *

# SETTINGS:
# _ DATA:
input_dir = "src/data/input/kitti/"
output_dir = "src/data/output/kitti/"
datasets_paths = \
    [
        os.path.join(input_dir, dataset_index)
        for dataset_index in
        ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"][4:6]
    ]
# _ METHOD:
methods = ["mono", "stereo"]
# _ FEATURE DETECTION:
fd_parameters = \
    {
        "fda": ["surf", "fast", "orb"][2],   # The feature detection method
        "nfeatures": 3000                       # The number of features to detect
    }
# _ POST MATCHING OUTLIER REMOVAL (PMOR):
pmor_parameters = \
    {
        "do_PMOR": True,        # The main switch (to enable/disable PMOR)
        "do_xyMeanDist": True,  # The sub-switch for the mean distance method
        "do_xyImgDist": True,   # The sub-switch for the image dimension method
        "do_RANSAC": True,      # The sub-switch for the RANSAC method
    }
# _ SEMANTIC SEGMENTATION (SS):
ss_parameters = \
    {
        "do_SS": True,
        "model_path": "src/models/deeplabv3_xception65_ade20k.h5",
        "features_to_ignore": ["sky", "person", "car"]
    }
# _ FRAME TILE OPTIMIZATION (FTO):
fto_parameters = \
    {
        "do_FTO": True,             # The main switch (to enable/disable FTO)
        "grid_h": 40,               # The number of tiles in the horizontal direction
        "grid_w": 20,               # The number of tiles in the vertical direction
        "patch_max_features": 10    # The maximum number of features to keep in each tile
    }
# _ VIEW/MONITOR/SAVE PARAMETERS:
view = True                         # Visualize the results
monitor = True                      # Show the progress bar (tqdm)
save = False                         # Save the data to a .csv file


# MAIN:
def main():
    csv = "dataset,method,fda,doPMOR,doxyMeanDist,doxyImgDist,doRANSAC,doSS,doFTO,grid_h,grid_w,patch_max_features,ate,ate%,nc_ate,nc_ate%"
    print(csv)
    # FOR EACH DATASET:
    for dataset_path in datasets_paths:
        for method in methods:
            # VISUAL ODOMETRY: Initialize the Visual Odometry class
            vo = VisualOdometry(
                data_dir=dataset_path,
                method=method,
                fd_parameters=fd_parameters,
                pmor_parameters=pmor_parameters,
                ss_parameters=ss_parameters,
                fto_parameters=fto_parameters
                )
            gt_path, est_path = vo.estimate_path(monitor=monitor, view=view)
            # MEASUREMENTS:
            ate, ate_percent, nc_ate, nc_ate_percent = get_ate(gt_path, est_path)
            # PRINT:
            print(
                f"{dataset_path.split('/')[-1]},"
                f"{method},"
                f"{fd_parameters['fda']},"
                f"{pmor_parameters['do_PMOR']},{pmor_parameters['do_xyMeanDist']},{pmor_parameters['do_xyImgDist']},{pmor_parameters['do_RANSAC']},"
                f"{ss_parameters['do_SS']},"
                f"{fto_parameters['do_FTO']},{fto_parameters['grid_h']},{fto_parameters['grid_w']},{fto_parameters['patch_max_features']},"
                f"{ate},{ate_percent},{nc_ate},{nc_ate_percent}"
            )
            if save:
                csv += "\n" \
                    f"{dataset_path.split('/')[-1]},"\
                    f"{method},"\
                    f"{fd_parameters['fda']},"\
                    f"{pmor_parameters['do_PMOR']},{pmor_parameters['do_xyMeanDist']},{pmor_parameters['do_xyImgDist']},{pmor_parameters['do_RANSAC']},"\
                    f"{ss_parameters['do_SS']},"\
                    f"{fto_parameters['do_FTO']},{fto_parameters['grid_h']},{fto_parameters['grid_w']},{fto_parameters['patch_max_features']},"\
                    f"{ate},{ate_percent},{nc_ate},{nc_ate_percent}"

    # SAVE:
    if save:
        output_path = os.path.join(output_dir, "RESULTS_" + datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + ".csv")
        with open(output_path, 'w') as f:
            f.write(csv)
        print("Finished the computation and saved results to: " + output_path)


if __name__ == "__main__":
    main()
