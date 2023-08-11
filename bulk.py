# IMPORTS=
from io import StringIO

# MODELS=
from src.Measurement import *
from src.VisualOdometry import *


# SETTINGS=
#   - DATA=
input_dir = "src/data/input/kitti/"
output_dir = "src/data/output/kitti/"
skitti_indexes = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
dataset_indexes = skitti_indexes
#   - VIEW=
monitor = True
view = True

# Combos of 2000 features:
def main():
    csv = "dataset,method,GRID_H,GRID_W,PATCH_MAX_FEATURES,GRID_MAX_FEATURES,ate,nc_ate,ate_percent,nc_ate_percent"
    print("dataset,method,GRID_H,GRID_W,PATCH_MAX_FEATURES,GRID_MAX_FEATURES,ate,nc_ate,ate_percent,nc_ate_percent")
    GRID_H_values = [4, 8, 10]
    GRID_W_values = [4, 8, 10, 20, 30]
    PATCH_MAX_FEATURES = 10 # So that the number of features never exceeds 3000
    for dataset_path in [os.path.join(input_dir, dataset_index) for dataset_index in dataset_indexes]:
        # 1ST: 3000 FEATURES (STEREO VO WITHOUT FTO):
        vo = VisualOdometry(
            dataset_path,
            method="stereo",
            fto_parameters={
                "do_FTO": False,
                "grid_h": 0,
                "grid_w": 0,
                "patch_max_features": 0
            }
        )
        gt_path, est_path = vo.estimate_path(monitor=monitor, view=view)     # Estimate the path
        # MEASUREMENTS:
        ate, nc_ate, ate_percent, nc_ate_percent = get_ate(gt_path, est_path)    # Absolute Trajectory Error
        print(f"{dataset_path.split('/')[-1]},SVO,NULL,NULL,NULL,3000,{ate},{nc_ate},{ate_percent},{nc_ate_percent}")
        csv += f"\n{dataset_path.split('/')[-1]},SVO,NULL,NULL,NULL,3000,{ate},{nc_ate},{ate_percent},{nc_ate_percent}"
        # 2ND: MAX 3000 FEATURES COMBOS (STEREO VO WITH FTO):
        for GRID_H in GRID_H_values:
            for GRID_W in GRID_W_values:
                try:
                    vo = VisualOdometry(
                        dataset_path,
                        method="stereo",
                        fto_parameters={
                            "do_FTO": True,
                            "grid_h": GRID_H,
                            "grid_w": GRID_W,
                            "patch_max_features": PATCH_MAX_FEATURES
                        }
                    )
                    gt_path, est_path = vo.estimate_path(monitor=monitor, view=view)     # Estimate the path
                    # MEASUREMENTS:
                    ate, nc_ate, ate_percent, nc_ate_percent = get_ate(gt_path, est_path)    # Absolute Trajectory Error
                    csv += f"\n{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},{ate},{nc_ate},{ate_percent},{nc_ate_percent}"
                    print(f"{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},{ate},{nc_ate},{ate_percent},{nc_ate_percent}")
                except:
                    print(f"{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},CRASHED,CRASHED,CRASHED,CRASHED")
                    csv += f"\n{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},CRASHED,CRASHED,CRASHED,CRASHED"
    curr_time = time.strftime("%Y%m%d-%H%M%S")
    with open(output_dir + "SKITTI11_BULK-SVO-FTO_"+curr_time+".csv", 'w') as f:
        f.write(csv)

    dataframe = pd.read_csv(StringIO(csv))
    df_ate, df_nc_ate = svo_fto_improvements_v2(dataframe=dataframe)
    df_ate.to_csv(output_dir + "SKITTI11_BULK-SVO-FTO_"+curr_time+"_ATE_COMPARISON.csv", index=False)
    df_nc_ate.to_csv(output_dir + "SKITTI11_BULK-SVO-FTO_"+curr_time+"_NCATE_COMPARISON.csv", index=False)


if __name__ == "__main__":
    main()
    filenames = os.listdir(output_dir)
