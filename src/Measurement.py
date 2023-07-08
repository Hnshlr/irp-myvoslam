import numpy as np

def get_xy_values(gt_path, est_path, print_values=False):
    x_final_diff = np.abs(np.round(gt_path[-1][0] - est_path[-1][0], 2))
    y_final_diff = np.abs(np.round(gt_path[-1][1] - est_path[-1][1], 2))
    x_mean_diff = np.abs(np.round(np.mean([gt_path[i][0] - est_path[i][0] for i in range(len(gt_path))]), 2))
    y_mean_diff = np.abs(np.round(np.mean([gt_path[i][1] - est_path[i][1] for i in range(len(gt_path))]), 2))
    x_max_diff = np.round(np.max([np.abs(gt_path[i][0] - est_path[i][0]) for i in range(len(gt_path))]), 2)
    y_max_diff = np.round(np.max([np.abs(gt_path[i][1] - est_path[i][1]) for i in range(len(gt_path))]), 2)
    if print_values:
        print("x_final_diff:", x_final_diff, " | y_final_diff:", y_final_diff, " | x_mean_diff:", x_mean_diff,
              " | y_mean_diff:", y_mean_diff, " | x_max_diff:", x_max_diff, " | y_max_diff:", y_max_diff)
    return x_final_diff, y_final_diff, x_mean_diff, y_mean_diff, x_max_diff, y_max_diff