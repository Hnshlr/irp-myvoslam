# IMPORTS=
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time

# MODELS=
from src.VisualOdometry import VisualOdometry

def main():
    # SETTINGS=
    input_dir = "src/data/input/kitti/"
    dataset_indexes = [["S1", "S2", "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"][7]]
    datasets = [os.path.join(input_dir, dataset_index) for dataset_index in dataset_indexes]

    # MAIN=
    csv_string = "dataset, x_final_diff, y_final_diff, x_mean_diff, y_mean_diff, x_max_diff, y_max_diff\n"
    for dataset in datasets:
        # VISUAL ODOMETRY:
        vo = VisualOdometry(dataset)    # Initialize the Visual Odometry class
        gt_path, est_path = [], []
        for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose", desc="Processing dataset")):
            if i == 0:  # First pose is the origin
                cur_pose = gt_pose
            else:
                q1, q2 = vo.get_matches(i, show=False)  # Get the matches between the current and previous image
                transf = vo.get_pose(q1, q2)    # Get the transformation matrix between the current and previous image
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))   # Update the current pose
            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))  # Append the ground truth path
            est_path.append((cur_pose[0, 3], cur_pose[2, 3]))   # Append the estimated path

        # TESTS=
        x_final_diff = np.abs(np.round(gt_path[-1][0] - est_path[-1][0], 2))
        y_final_diff = np.abs(np.round(gt_path[-1][1] - est_path[-1][1], 2))
        x_mean_diff = np.abs(np.round(np.mean([gt_path[i][0] - est_path[i][0] for i in range(len(gt_path))]), 2))
        y_mean_diff = np.abs(np.round(np.mean([gt_path[i][1] - est_path[i][1] for i in range(len(gt_path))]), 2))
        x_max_diff = np.abs(np.round(np.max([gt_path[i][0] - est_path[i][0] for i in range(len(gt_path))]), 2))
        y_max_diff = np.abs(np.round(np.max([gt_path[i][1] - est_path[i][1] for i in range(len(gt_path))]), 2))
        print("x_final_diff:", x_final_diff, " | y_final_diff:", y_final_diff, " | x_mean_diff:", x_mean_diff, " | y_mean_diff:", y_mean_diff, " | x_max_diff:", x_max_diff, " | y_max_diff:", y_max_diff)
        csv_string += f"{dataset}, {x_final_diff}, {y_final_diff}, {x_mean_diff}, {y_mean_diff}, {x_max_diff}, {y_max_diff}\n"
        time.sleep(1)

    # SAVE RESULTS=
    with open("src/data/output/results.csv", "w") as f:
        f.write(csv_string)


if __name__ == "__main__":
    main()
