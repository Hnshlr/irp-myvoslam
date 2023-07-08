import numpy as np
import cv2
import os
from tqdm import tqdm

def load_calib(filepath):
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P

def load_calib_v2(filepath):
    # Store all lines in a list of strings:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Extract the parameters from the first line:
    line0 = lines[0].split(' ')[1:]
    # Convert the parameters to a numpy array of floats:
    params = np.fromstring(' '.join(line0), dtype=np.float64, sep=' ')
    # Reshape the array to a 3x4 matrix (to obtain the projection matrix and the intrinsic matrix):
    P = np.reshape(params, (3, 4))
    K = P[0:3, 0:3]
    return K, P

def load_calib_LR(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Extract the parameters from the first line:
    line_left = lines[0].split(' ')[1:]
    line_right = lines[1].split(' ')[1:]
    # Convert the parameters to a numpy array of floats:
    params_left = np.fromstring(' '.join(line_left), dtype=np.float64, sep=' ')
    params_right = np.fromstring(' '.join(line_right), dtype=np.float64, sep=' ')
    # Reshape the array to a 3x4 matrix (to obtain the projection matrix and the intrinsic matrix):
    P_left = np.reshape(params_left, (3, 4))
    P_right = np.reshape(params_right, (3, 4))
    K_left = P_left[0:3, 0:3]
    K_right = P_right[0:3, 0:3]
    return K_left, P_left, K_right, P_right


def load_poses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses

def load_images(filepath):
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in tqdm(image_paths, desc="Loading images")]

def form_transf(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T