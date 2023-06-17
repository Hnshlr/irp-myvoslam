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
    line0 = lines[0].split(' ')[1:]
    # Extract the first two lines:
    params = np.fromstring(' '.join(line0), dtype=np.float64, sep=' ')
    # Reshape the array to a 3x4 matrix:
    P = np.reshape(params, (3, 4))
    K = P[0:3, 0:3]
    return K, P

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