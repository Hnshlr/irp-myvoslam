import numpy as np
import cv2
import os
from tqdm import tqdm


def load_calib(filepath):
    """
    Loads the calibration parameters from a file.
    :param filepath: Path to the calibration file.
    :return: The intrinsic matrix K and the projection matrix P.
    """
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
    """
    Loads the calibration parameters from a file.
    :param filepath: Path to the calibration file.
    :return: The intrinsic matrix K and the projection matrix P, for both the left and the right camera.
    """
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
    """
    Loads the poses from a file.
    :param filepath: Path to the file containing the poses.
    :return: A list of all the poses.
    """
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses


def load_images(filepath, monitor=False):
    """
    Loads all the images in a directory.
    :param filepath: Path to the directory containing the images.
    :param monitor: Whether to show the progress bar (tqdm).
    :return: A list of all the images in the directory.
    """
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
    if monitor:
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in tqdm(image_paths, desc="Loading images", leave=False)]
    else:
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]


def form_transf(R, t):
    """
    Forms a transformation matrix from a rotation matrix and a translation vector.
    :param R: The rotation matrix.
    :param t: The translation vector.
    :return: The transformation matrix.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes an image while maintaining the aspect ratio.
    :param image: The image to resize.
    :param width: The desired width.
    :param height: The desired height.
    :param inter: The interpolation method to use.
    :return:
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)