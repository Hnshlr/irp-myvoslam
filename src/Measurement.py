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
    ate_percent = np.round(ate / np.mean(np.linalg.norm(gt_path, axis=1)) * 100, 2)
    nc_ate_percent = np.round(nc_ate / np.mean(np.linalg.norm(gt_path, axis=1)) * 100, 2)
    return ate, nc_ate, ate_percent, nc_ate_percent


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

def svo_fto_improvements(filepath=None, dataframe=None):
    # Read the CSV file:
    if filepath is not None:
        df = pd.read_csv(filepath, sep=',', header=0)
    elif dataframe is not None:
        df = dataframe
    else:
        raise Exception("No CSV file or dataframe provided!")

    # Gather the ATE and NC-ATE values for Stereo Visual Odometry method:
    df_SVO = df[df['method'] == "SVO"]

    # Gather the best ATE for the SVO with FTO, and the grid combo used:
    df_best_ate = df\
        .sort_values(['dataset', 'ate']) \
        .drop_duplicates(subset=['dataset'], keep='first')

    # Gather the best NC-ATE for the SVO with FTO, and the grid combo used:
    df_best_nc_ate = df\
                .sort_values(['dataset', 'nc_ate'])\
                .drop_duplicates(subset=['dataset'], keep='first')

    # Add lines in a single dataframe:
    df_ = pd.concat([df_SVO, df_best_ate, df_best_nc_ate]) \
        .sort_values(['dataset', 'method']) \
        .reset_index()

    # Gather the dataset names:
    dataset_indexes = df_['dataset'].drop_duplicates()
    dataset_indexes = dataset_indexes.to_numpy()
    dataset_indexes.sort()
    dataset_indexes = dataset_indexes[np.argsort([len(x) for x in dataset_indexes])]

    # Gather the datasets whose ATE is better using FTO:
    dataset_whose_ate_is_better_using_fto = []
    for dataset in dataset_indexes:
        svo_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['ate'].values)
        svo_ate = float(svo_ate)
        try:
            fto_ate = np.min([float(value) for value in
                              df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['ate'].values])
        except:
            fto_ate = np.inf
        fto_ate = float(fto_ate)
        if fto_ate < svo_ate:
            dataset_whose_ate_is_better_using_fto.append(dataset)

    # Gather the datasets whose NC-ATE is better using FTO:
    dataset_whose_nc_ate_is_better_using_fto = []
    for dataset in dataset_indexes:
        svo_nc_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['nc_ate'].values)
        svo_nc_ate = float(svo_nc_ate)
        try:
            fto_nc_ate = np.min([float(value) for value in
                                 df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['nc_ate'].values])
        except:
            fto_nc_ate = np.inf
        fto_nc_ate = float(fto_nc_ate)
        if fto_nc_ate < svo_nc_ate:
            dataset_whose_nc_ate_is_better_using_fto.append(dataset)

    # Gather the datasets who benefited from FTO:
    data_who_FTO_improved_either_ate_or_nc_ate = []
    for dataset in dataset_indexes:
        if dataset in dataset_whose_ate_is_better_using_fto or dataset in dataset_whose_nc_ate_is_better_using_fto:
            data_who_FTO_improved_either_ate_or_nc_ate.append(dataset)

    # Gather the datasets who did not benefit from FTO:
    dataset_whose_FTO_did_not_improve_ate_nor_nc_ate = []
    for dataset in dataset_indexes:
        if dataset not in data_who_FTO_improved_either_ate_or_nc_ate:
            dataset_whose_FTO_did_not_improve_ate_nor_nc_ate.append(dataset)


    # For each dataset who FTO improved either the ATE or Non-cumulative ATE, print the % of improvement:
    for dataset in data_who_FTO_improved_either_ate_or_nc_ate:
        svo_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['ate'].values)
        svo_ate = float(svo_ate)
        fto_ate = np.min(
            [float(value) for value in df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['ate'].values])
        fto_ate = float(fto_ate)
        svo_nc_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['nc_ate'].values)
        svo_nc_ate = float(svo_nc_ate)
        fto_nc_ate = np.min(
            [float(value) for value in df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['nc_ate'].values])
        fto_nc_ate = float(fto_nc_ate)
        # print("For", dataset, "the Non-cumulative ATE improved by ", np.round((svo_nc_ate - fto_nc_ate) / svo_nc_ate * 100, 2), "%", " and the ATE improved by ", np.round((svo_ate - fto_ate) / svo_ate * 100, 2), "% using FTO.")

    # Create an array containing the % of improvement for each dataset of the ATE and NC-ATE, with the GRID combo used
    # (if FTO did not improve the ATE or NC-ATE, the value for the ate and nc_ate and the GRID combo used is np.nan):
    df_ate = pd.DataFrame(columns=['dataset', 'GRID_H', 'GRID_W', 'SVO_ATE', 'SVOFTO_ATE', 'ATE_improvement'])
    for dataset in dataset_indexes:
        svo_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['ate'].values)
        svo_ate = float(svo_ate)
        try:
            fto_ate = np.min([float(value) for value in
                              df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['ate'].values])
            fto_ate = float(fto_ate)
            grid_h = df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['GRID_H'].values[0]
            grid_w = df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['GRID_W'].values[0]
        except:
            fto_ate = np.nan
            grid_h = np.nan
            grid_w = np.nan
        if fto_ate < svo_ate:
            ate_improvement = np.round((svo_ate - fto_ate) / svo_ate * 100, 2)
        else:
            ate_improvement = np.nan
        df_ate = df_ate.append({'dataset': dataset, 'GRID_H': grid_h, 'GRID_W': grid_w, 'SVO_ATE': svo_ate,
                                'SVOFTO_ATE': fto_ate, 'ATE_improvement': ate_improvement}, ignore_index=True)

    df_nc_ate = pd.DataFrame(columns=['dataset', 'GRID_H', 'GRID_W', 'SVO_NC_ATE', 'SVOFTO_NC_ATE', 'NC_ATE_improvement'])

    for dataset in dataset_indexes:
        svo_nc_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['nc_ate'].values)
        svo_nc_ate = float(svo_nc_ate)
        try:
            fto_nc_ate = np.min([float(value) for value in
                                 df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['nc_ate'].values])
            fto_nc_ate = float(fto_nc_ate)
            grid_h = df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['GRID_H'].values[0]
            grid_w = df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['GRID_W'].values[0]
        except:
            fto_nc_ate = np.nan
            grid_h = np.nan
            grid_w = np.nan
        if fto_nc_ate < svo_nc_ate:
            nc_ate_improvement = np.round((svo_nc_ate - fto_nc_ate) / svo_nc_ate * 100, 2)
        else:
            nc_ate_improvement = np.nan
        df_nc_ate = df_nc_ate.append({'dataset': dataset, 'GRID_H': grid_h, 'GRID_W': grid_w, 'SVO_NC_ATE': svo_nc_ate,
                                      'SVOFTO_NC_ATE': fto_nc_ate, 'NC_ATE_improvement': nc_ate_improvement}, ignore_index=True)

    return df_ate, df_nc_ate

def svo_fto_improvements_v2(filepath=None, dataframe=None):
    # Read the CSV file:
    if filepath is not None:
        df = pd.read_csv(filepath, sep=',', header=0)
    elif dataframe is not None:
        df = dataframe
    else:
        raise Exception("No CSV file or dataframe provided!")

    # Gather the ATE and NC-ATE values for Stereo Visual Odometry method:
    df_SVO = df[df['method'] == "SVO"]

    # Gather the best ATE for the SVO with FTO, and the grid combo used:
    df_best_ate = df\
        .sort_values(['dataset', 'ate_percent'])\
        .drop_duplicates(subset=['dataset'], keep='first')

    # Gather the best NC-ATE for the SVO with FTO, and the grid combo used:
    df_best_nc_ate = df\
                .sort_values(['dataset', 'nc_ate_percent'])\
                .drop_duplicates(subset=['dataset'], keep='first')

    # Add lines in a single dataframe:
    df_ = pd.concat([df_SVO, df_best_ate, df_best_nc_ate]) \
        .sort_values(['dataset', 'method']) \
        .reset_index()

    # Gather the dataset names:
    dataset_indexes = df_['dataset'].drop_duplicates()
    dataset_indexes = dataset_indexes.to_numpy()
    dataset_indexes.sort()
    dataset_indexes = dataset_indexes[np.argsort([len(x) for x in dataset_indexes])]

    # Gather the datasets whose ATE is better using FTO:
    dataset_whose_ate_is_better_using_fto = []
    for dataset in dataset_indexes:
        svo_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['ate_percent'].values)
        svo_ate = float(svo_ate)
        try:
            fto_ate = np.min([float(value) for value in
                              df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['ate_percent'].values])
        except:
            fto_ate = np.inf
        fto_ate = float(fto_ate)
        if fto_ate < svo_ate:
            dataset_whose_ate_is_better_using_fto.append(dataset)

    # Gather the datasets whose NC-ATE is better using FTO:
    dataset_whose_nc_ate_is_better_using_fto = []
    for dataset in dataset_indexes:
        svo_nc_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['nc_ate_percent'].values)
        svo_nc_ate = float(svo_nc_ate)
        try:
            fto_nc_ate = np.min([float(value) for value in
                                 df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['nc_ate_percent'].values])
        except:
            fto_nc_ate = np.inf
        fto_nc_ate = float(fto_nc_ate)
        if fto_nc_ate < svo_nc_ate:
            dataset_whose_nc_ate_is_better_using_fto.append(dataset)

    # Gather the datasets who benefited from FTO:
    data_who_FTO_improved_either_ate_or_nc_ate = []
    for dataset in dataset_indexes:
        if dataset in dataset_whose_ate_is_better_using_fto or dataset in dataset_whose_nc_ate_is_better_using_fto:
            data_who_FTO_improved_either_ate_or_nc_ate.append(dataset)

    # Gather the datasets who did not benefit from FTO:
    dataset_whose_FTO_did_not_improve_ate_nor_nc_ate = []
    for dataset in dataset_indexes:
        if dataset not in data_who_FTO_improved_either_ate_or_nc_ate:
            dataset_whose_FTO_did_not_improve_ate_nor_nc_ate.append(dataset)


    # For each dataset who FTO improved either the ATE or Non-cumulative ATE, print the % of improvement:
    for dataset in data_who_FTO_improved_either_ate_or_nc_ate:
        svo_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['ate_percent'].values)
        svo_ate = float(svo_ate)
        fto_ate = np.min(
            [float(value) for value in df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['ate'].values])
        fto_ate = float(fto_ate)
        svo_nc_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['nc_ate_percent'].values)
        svo_nc_ate = float(svo_nc_ate)
        fto_nc_ate = np.min(
            [float(value) for value in df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['nc_ate_percent'].values])
        fto_nc_ate = float(fto_nc_ate)
        # print("For", dataset, "the Non-cumulative ATE improved by ", np.round((svo_nc_ate - fto_nc_ate) / svo_nc_ate * 100, 2), "%", " and the ATE improved by ", np.round((svo_ate - fto_ate) / svo_ate * 100, 2), "% using FTO.")

    # Create an array containing the % of improvement for each dataset of the ATE and NC-ATE, with the GRID combo used
    # (if FTO did not improve the ATE or NC-ATE, the value for the ate and nc_ate and the GRID combo used is np.nan):
    df_ate = pd.DataFrame(columns=['dataset', 'GRID_H', 'GRID_W', 'SVO_ATE', 'SVOFTO_ATE', 'ATE_improvement'])
    for dataset in dataset_indexes:
        svo_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['ate_percent'].values)
        svo_ate = float(svo_ate)
        try:
            fto_ate = np.min([float(value) for value in
                              df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['ate_percent'].values])
            fto_ate = float(fto_ate)
            grid_h = df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['GRID_H'].values[0]
            grid_w = df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['GRID_W'].values[0]
        except:
            fto_ate = np.nan
            grid_h = np.nan
            grid_w = np.nan
        if fto_ate < svo_ate:
            ate_improvement = np.round((svo_ate - fto_ate) / svo_ate * 100, 2)
        else:
            ate_improvement = np.nan
        df_ate = df_ate.append({'dataset': dataset, 'GRID_H': grid_h, 'GRID_W': grid_w, 'SVO_ATE': svo_ate,
                                'SVOFTO_ATE': fto_ate, 'ATE_improvement': ate_improvement}, ignore_index=True)

    df_nc_ate = pd.DataFrame(columns=['dataset', 'GRID_H', 'GRID_W', 'SVO_NC_ATE', 'SVOFTO_NC_ATE', 'NC_ATE_improvement'])

    for dataset in dataset_indexes:
        svo_nc_ate = np.min(df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVO')]['nc_ate'].values)
        svo_nc_ate = float(svo_nc_ate)
        try:
            fto_nc_ate = np.min([float(value) for value in
                                 df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['nc_ate'].values])
            fto_nc_ate = float(fto_nc_ate)
            grid_h = df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['GRID_H'].values[0]
            grid_w = df_[(df_['dataset'] == dataset) & (df_['method'] == 'SVOFTO')]['GRID_W'].values[0]
        except:
            fto_nc_ate = np.nan
            grid_h = np.nan
            grid_w = np.nan
        if fto_nc_ate < svo_nc_ate:
            nc_ate_improvement = np.round((svo_nc_ate - fto_nc_ate) / svo_nc_ate * 100, 2)
        else:
            nc_ate_improvement = np.nan
        df_nc_ate = df_nc_ate.append({'dataset': dataset, 'GRID_H': grid_h, 'GRID_W': grid_w, 'SVO_NC_ATE': svo_nc_ate,
                                      'SVOFTO_NC_ATE': fto_nc_ate, 'NC_ATE_improvement': nc_ate_improvement}, ignore_index=True)

    return df_ate, df_nc_ate
