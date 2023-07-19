# MODELS=
from src.Measurement import *
from src.VisualOdometry import *

# SETTINGS=
#   - DATA=
input_dir = "src/data/input/kitti/"
skitti_indexes = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"][6:7]
kitti_indexes = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
dataset_indexes = skitti_indexes
#   - METHOD=
methods = ["mono", "stereo"][0:1]
#   - SEMANTIC SEGMENTATION=
doSegmentate = False
model_path = "src/models/deeplabv3_xception65_ade20k.h5"
features = ["earth", "grass", "sidewalk", "road", "building"][3:4]
#   - VIEW=
view = False
saveData = False

# MAIN:
def main():
    # SAVE DATA:
    xy_values_csv = "dataset, method, x_final_diff, y_final_diff, x_mean_diff, y_mean_diff, x_max_diff, y_max_diff\n"
    # FOR EACH DATASET:∂
    for dataset_path in [os.path.join(input_dir, dataset_index) for dataset_index in dataset_indexes]:
        for method in methods:
            # VISUAL ODOMETRY: Initialize the Visual Odometry class
            vo = VisualOdometry(dataset_path,
                                method=method,
                                semantic_segmentation_parameters={
                                    "segmentate": doSegmentate,
                                    "model_path": model_path,
                                    "features": features
                                })
            gt_path, est_path = vo.estimate_path(view=view)     # Estimate the path
            # MEASUREMENTS:
            x_final_diff, y_final_diff, x_mean_diff, y_mean_diff, x_max_diff, y_max_diff = get_xy_values(gt_path, est_path, print_values=True)  # Get the measurements
            # SAVE DATA:
            xy_values_csv += dataset_path.split("/")[-1] + ", " + method + ", " + str(x_final_diff) + ", " + str(y_final_diff) + ", " + str(x_mean_diff) + ", " + str(y_mean_diff) + ", " + str(x_max_diff) + ", " + str(y_max_diff) + "\n"

    # SAVE CSV:
    if saveData:
        with open("src/data/output/" + time.strftime("%Y-%m-%d_%H-%M") + ".csv", "w") as f:
            f.write(xy_values_csv)


if __name__ == "__main__":
    main()
