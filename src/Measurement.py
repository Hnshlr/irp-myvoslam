import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_xy_values(gt_path, est_path):
    x_final_diff = np.abs(np.round(gt_path[-1][0] - est_path[-1][0], 2))
    y_final_diff = np.abs(np.round(gt_path[-1][1] - est_path[-1][1], 2))
    x_mean_diff = np.abs(np.round(np.mean([gt_path[i][0] - est_path[i][0] for i in range(len(gt_path))]), 2))
    y_mean_diff = np.abs(np.round(np.mean([gt_path[i][1] - est_path[i][1] for i in range(len(gt_path))]), 2))
    x_max_diff = np.round(np.max([np.abs(gt_path[i][0] - est_path[i][0]) for i in range(len(gt_path))]), 2)
    y_max_diff = np.round(np.max([np.abs(gt_path[i][1] - est_path[i][1]) for i in range(len(gt_path))]), 2)
    return x_final_diff, y_final_diff, x_mean_diff, y_mean_diff, x_max_diff, y_max_diff

def get_ate(gt_path, est_path):
    gt_path = np.array(gt_path)
    est_path = np.array(est_path)
    errors = np.linalg.norm(gt_path - est_path, axis=1)
    non_cumulative_errors = np.concatenate(([0], [errors[i] - errors[i - 1] for i in range(1, len(errors))]))
    ate = np.mean(errors)
    nc_ate = np.mean(non_cumulative_errors)
    ate = np.round(ate, 2 - int(np.floor(np.log10(abs(ate)))) - 1)
    nc_ate = np.round(nc_ate, 2 - int(np.floor(np.log10(abs(nc_ate)))) - 1)
    return ate, nc_ate


# PLOT THE MONO, STEREO AND GT PATHS:
def plot_poses(dataset_path, gt_path, est_paths):
    plt.plot([gt_path[i][0] for i in range(len(gt_path))], [gt_path[i][1] for i in range(len(gt_path))], label="Ground Truth", color="blue")
    plt.plot([est_paths[0][i][0] for i in range(len(est_paths[0]))], [est_paths[0][i][1] for i in range(len(est_paths[0]))], label="Mono", color="orange")
    plt.plot([est_paths[1][i][0] for i in range(len(est_paths[1]))], [est_paths[1][i][1] for i in range(len(est_paths[1]))], label="Stereo", color="green")
    # SET X AXIS MAX TO HALF OF Y AXIS MAX:
    plt.xlim(
        np.min(np.concatenate((
            np.array([gt_path[i][0] for i in range(len(gt_path))]),
            np.array([est_paths[0][i][0] for i in range(len(est_paths[0]))]),
            np.array([est_paths[1][i][0] for i in range(len(est_paths[1]))])))),
        np.max(np.concatenate((
            np.array([gt_path[i][0] for i in range(len(gt_path))]),
            np.array([est_paths[0][i][0] for i in range(len(est_paths[0]))]),
            np.array([est_paths[1][i][0] for i in range(len(est_paths[1]))]))))
    )
    plt.title("Pose Estimation of the " + dataset_path.split("/")[-1] + " Dataset")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.savefig("src/data/output/" + dataset_path.split("/")[-1] + "_" + "POSE_EST"
                + time.strftime("%Y-%m-%d_%H-%M") + ".png")
    plt.close()
    plt.show()

# PLOT THE ERROR EVOLUTION OF THE MONO AND STEREO PATHS COMPARED TO THE GT PATH (DISTANCE=SQRT(X^2+Y^2)):
def plot_errors(dataset_path, gt_path, est_paths):
    plt.plot(
        [np.sqrt((gt_path[i][0] - est_paths[0][i][0]) ** 2 + (gt_path[i][1] - est_paths[0][i][1]) ** 2) for
         i in range(len(gt_path))], label="Euclidean Distance Error (Mono)", color="orange")
    plt.plot(
        [np.sqrt((gt_path[i][0] - est_paths[1][i][0]) ** 2 + (gt_path[i][1] - est_paths[1][i][1]) ** 2) for
         i in range(len(gt_path))], label="Euclidean Distance Error (Stereo)", color="green")
    plt.title("Euclidean Distance Error of the " + dataset_path.split("/")[-1] + " Dataset")
    plt.xlabel("Frame Index")
    plt.ylabel("Euclidean Distance Error (m)")
    plt.legend()
    plt.savefig("src/data/output/" + dataset_path.split("/")[-1] + "_" + "ED_ERROR"
                + time.strftime("%Y-%m-%d_%H-%M") + ".png")
    plt.close()
    plt.show()

# PLOT THE ERROR EVOLUTION OF THE MONO AND STEREO PATHS COMPARED TO THE GT PATH (DISTANCE=SQRT(X^2+Y^2)) BUT FOR EACH STEP (NOT CUMULATIVE):`
def plot_ncerrors(dataset_path, gt_path, est_paths):
    plt.plot([np.sqrt(
        (gt_path[i][0] - est_paths[0][i][0]) ** 2 + (gt_path[i][1] - est_paths[0][i][1]) ** 2) - np.sqrt(
        (gt_path[i - 1][0] - est_paths[0][i - 1][0]) ** 2 + (
                    gt_path[i - 1][1] - est_paths[0][i - 1][1]) ** 2) for i in range(1, len(gt_path))],
             label="Non-cumulative Euclidean Distance Error (Mono)", color="orange")
    plt.plot([np.sqrt(
        (gt_path[i][0] - est_paths[1][i][0]) ** 2 + (gt_path[i][1] - est_paths[1][i][1]) ** 2) - np.sqrt(
        (gt_path[i - 1][0] - est_paths[1][i - 1][0]) ** 2 + (
                    gt_path[i - 1][1] - est_paths[1][i - 1][1]) ** 2) for i in range(1, len(gt_path))],
             label="Non-cumulative Euclidean Distance Error (Stereo)", color="green")
    plt.title("Non-cumulative Euclidean Distance Error of the " + dataset_path.split("/")[-1] + " Dataset")
    plt.xlabel("Frame Index")
    plt.ylabel("Non-cumulative Euclidean Distance Error (m)")
    plt.legend()
    plt.savefig("src/data/output/" + dataset_path.split("/")[-1] + "_" + "NC-ED_ERROR"
                + time.strftime("%Y-%m-%d_%H-%M") + ".png")
    plt.close()
    plt.show()