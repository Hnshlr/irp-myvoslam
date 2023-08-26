# IMPORTS:
from datetime import *
from io import StringIO

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
        ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
    ]
# _ METHOD:
method = "stereo"
# _ FEATURE DETECTION:
fd_parameters = \
    {
        "fda": ["surf", "fast", "orb"][2],   # The feature detection method
        "nfeatures": 3000                       # The number of features to detect
    }
# _ FRAME TILE OPTIMIZATION (FTO):
GRID_H_values = [4, 8, 10]
GRID_W_values = [4, 8, 10, 20, 30]
PATCH_MAX_FEATURES = 10  # So that the number of features never exceeds 3000
# _ VIEW/MONITOR/SAVE PARAMETERS:
view = False                         # Visualize the results
monitor = True                      # Show the progress bar (tqdm)
save = True                         # Save the data to a .csv file


# MAIN:
def main():
    csv = "dataset,method,GRID_H,GRID_W,PATCH_MAX_FEATURES,GRID_MAX_FEATURES,ate,nc_ate,ate_percent,nc_ate_percent"
    print("dataset,method,GRID_H,GRID_W,PATCH_MAX_FEATURES,GRID_MAX_FEATURES,ate,nc_ate,ate_percent,nc_ate_percent")
    # FOR EACH DATASET:
    for dataset_path in datasets_paths:
        # VISUAL ODOMETRY: Initialize the Visual Odometry class
        vo = VisualOdometry(
            data_dir=dataset_path,
            method=method,
            fd_parameters=fd_parameters,
            fto_parameters={
                "do_FTO": False,
                "grid_h": 0,
                "grid_w": 0,
                "patch_max_features": 0
            }
            )
        gt_path, est_path = vo.estimate_path(monitor=monitor, view=view)
        # MEASUREMENTS:
        ate, ate_percent, nc_ate, nc_ate_percent = get_ate(gt_path, est_path)
        # PRINT:
        print(f"{dataset_path.split('/')[-1]},SVO,NULL,NULL,NULL,3000,{ate},{nc_ate},{ate_percent},{nc_ate_percent}")
        # SAVE:
        if save:
            csv += f"\n{dataset_path.split('/')[-1]},SVO,NULL,NULL,NULL,3000,{ate},{nc_ate},{ate_percent},{nc_ate_percent}"

        # FOR EACH COMBO OF FTO PARAMETERS:
        for GRID_H in GRID_H_values:
            for GRID_W in GRID_W_values:
                try:
                    vo = VisualOdometry(
                        data_dir=dataset_path,
                        method=method,
                        fd_parameters=fd_parameters,
                        fto_parameters={
                            "do_FTO": True,
                            "grid_h": GRID_H,
                            "grid_w": GRID_W,
                            "patch_max_features": PATCH_MAX_FEATURES
                        }
                    )
                    gt_path, est_path = vo.estimate_path(monitor=monitor, view=view)
                    # MEASUREMENTS:
                    ate, ate_percent, nc_ate, nc_ate_percent = get_ate(gt_path, est_path)
                    # PRINT:
                    print(f"{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},{ate},{nc_ate},{ate_percent},{nc_ate_percent}")
                    # SAVE:
                    if save:
                        csv += f"\n{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},{ate},{nc_ate},{ate_percent},{nc_ate_percent}"
                except:
                    print(f"{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},CRASHED,CRASHED,CRASHED,CRASHED")
                    if save:
                        csv += f"\n{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},CRASHED,CRASHED,CRASHED,CRASHED"

    # SAVE:
    if save:
        curr_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        output_path = os.path.join(output_dir, "BULKFTO_" + curr_time + ".csv")
        with open(output_path, 'w') as f:
            f.write(csv)

        dataframe = pd.read_csv(StringIO(csv))
        df_ate, df_nc_ate = svo_fto_improvements_v2(dataframe=dataframe)
        output_path_ate = os.path.join(output_dir, "BULKFTO_ATE_COMPARISON_" + curr_time + ".csv")
        output_path_nc_ate = os.path.join(output_dir, "BULKFTO_NCATE_COMPARISON_" + curr_time + ".csv")
        df_ate.to_csv(output_path_ate, index=False)
        df_nc_ate.to_csv(output_path_nc_ate, index=False)
        print(f"Finished the BULK FTO computation and saved all three results files to:")
        print(f"{output_path}\n{output_path_ate}\n{output_path_nc_ate}")


if __name__ == "__main__":
    main()
