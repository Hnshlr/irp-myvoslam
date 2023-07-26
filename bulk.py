# MODELS=
from src.Measurement import *
from src.VisualOdometry import *

# SETTINGS=
#   - DATA=
input_dir = "src/data/input/kitti/"
output_dir = "src/data/output/kitti/SVOFTO/"
skitti_indexes = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
dataset_indexes = skitti_indexes
#   - VIEW=
monitor = True
view = False

# Combos of 2000 features:
def main():
    csv = "dataset,method,GRID_H,GRID_W,PATCH_MAX_FEATURES,GRID_MAX_FEATURES,ate,nc_ate"
    print("dataset,method,GRID_H,GRID_W,PATCH_MAX_FEATURES,GRID_MAX_FEATURES,ate,nc_ate")
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
        ate, nc_ate = get_ate(gt_path, est_path)    # Absolute Trajectory Error
        print(f"{dataset_path.split('/')[-1]},SVO,NULL,NULL,NULL,3000,{ate},{nc_ate}")
        csv += f"\n{dataset_path.split('/')[-1]},SVO,NULL,NULL,NULL,3000,{ate},{nc_ate}"
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
                    ate, nc_ate = get_ate(gt_path, est_path)    # Absolute Trajectory Error
                    csv += f"\n{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},{ate},{nc_ate}"
                    print(f"{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},{ate},{nc_ate}")
                except:
                    print(f"{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},CRASHED,CRASHED")
                    csv += f"\n{dataset_path.split('/')[-1]},SVOFTO,{GRID_H},{GRID_W},{PATCH_MAX_FEATURES},{GRID_H * GRID_W * PATCH_MAX_FEATURES},CRASHED,CRASHED"
    with open(output_dir + "SKITTI11_BULK-SVO-FTO_"+ time.strftime("%Y%m%d-%H%M%S") +".csv", "w") as f:
        f.write(csv)


if __name__ == "__main__":
    for i in range(9):
        main()
    filenames = os.listdir(output_dir)
    for filename in filenames:
        if filename.endswith('.csv'):
            print(filename)
            df_ate, df_nc_ate = svo_fto_improvements(filepath=output_dir + filename)
            df_ate.to_csv(output_dir + filename[:-4] + '_ATE_COMPARISON.csv', index=False)
            df_nc_ate.to_csv(output_dir + filename[:-4] + '_NCATE_COMPARISON.csv', index=False)
