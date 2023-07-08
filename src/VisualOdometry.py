import os
import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import least_squares

from src.Utils import *
from src.SemanticSegmentation import *

class VisualOdometry():
    def __init__(self, data_dir, method="mono"):
        self.method = method
        self.dataset_dir_path = data_dir
        if method == "mono":
            self.K, self.P = load_calib_v2(os.path.join(data_dir, 'calib.txt'))
            self.gt_poses = load_poses(os.path.join(data_dir, 'poses.txt'))
            self.images = load_images(os.path.join(data_dir, "image_0"))
            self.orb = cv2.ORB_create(3000)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        elif method == "stereo":
            self.K_l, self.P_l, self.K_r, self.P_r = load_calib_LR(data_dir + '/calib.txt')
            self.gt_poses = load_poses(data_dir + '/poses.txt')
            self.images_l = load_images(data_dir + '/image_0')
            self.images_r = load_images(data_dir + '/image_1')

            # Disparity map creation:
            block = 11
            P1 = block * block * 8
            P2 = block * block * 32
            self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
            # disparity is a StereoSGBM object that can be used to compute the disparity map(s) between two images.
            # P1 and P2 are parameters that control the smoothness of the disparity map.
            # The block size is the size of the window used to match pixels between the two images.
            # The number of disparities is the maximum disparity minus the minimum disparity. The maximum disparity represents the maximum shift between the two images.

            # Create the disparities list and add the first disparity map (between the first left and right images):
            self.disparities = [
                np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]
            # disparities is a list of disparity maps, one for each frame. Each disparity map is a numpy array of shape (height, width),
            # same dimensions as the images. The values in the disparity map represent the disparity in pixels between the two images.

            self.fastFeatures = cv2.FastFeatureDetector_create()

            # Parameters for lucas kanade optical flow
            self.lk_params = dict(winSize=(15, 15),
                                  flags=cv2.MOTION_AFFINE,
                                  maxLevel=3,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        else:
            raise ValueError("Method must be 'mono' or 'stereo'.")

    # PATH ESTIMATION (MONO/STEREO):
    def estimate_path(self, show_matches=False, features=[None].clear()):
        # PATHS= (FOR SEMANTIC SEGMENTATION)
        images_dir_path = os.path.join(self.dataset_dir_path, "image_0")
        images_paths = [os.path.join(images_dir_path, file) for file in sorted(os.listdir(images_dir_path))]
        images_paths.sort()
        gt_path, est_path = [], []
        if self.method == "mono":
            for i, gt_pose in enumerate(tqdm(self.gt_poses, unit="pose", desc="Processing dataset")):
                if i == 0:  # First pose is the origin
                    cur_pose = gt_pose
                else:
                    q1, q2 = self.get_matches(i,
                                              show=show_matches,
                                              prev_mask=get_total_upscaled_mask(images_paths[i - 1], features=features),
                                              curr_mask=get_total_upscaled_mask(images_paths[i], features=features)
                                              )  # Get the matches between the current and previous image
                    transf = self.get_pose_from_matches(q1, q2)  # Get the transformation matrix between the current and previous image
                    cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))  # Update the current pose
                gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))  # Append the ground truth path
                est_path.append((cur_pose[0, 3], cur_pose[2, 3]))  # Append the estimated path
            return gt_path, est_path
        elif self.method == "stereo":
            for i, gt_pose in enumerate(tqdm(self.gt_poses, unit="poses", desc="Processing dataset")):
                if i < 1:
                    cur_pose = gt_pose
                else:
                    transf = self.get_pose(i)
                    cur_pose = np.matmul(cur_pose, transf)
                gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
                est_path.append((cur_pose[0, 3], cur_pose[2, 3]))
            return gt_path, est_path
        else:
            raise ValueError("Invalid method")

    # MONO METHODS=

    def get_matches(self, i, show=True, prev_mask=None, curr_mask=None):
        # VERSION 1: ORB + FLANN
        kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], mask=prev_mask)
        kp2, des2 = self.orb.detectAndCompute(self.images[i], mask=curr_mask)
        # The two previous lines serve to find the keypoints and descriptors of the two images
        # Then we match the descriptors, and only keep the good matches.
        matches = self.flann.knnMatch(des1, des2, k=2)

        # VERSION 2: FAST FEATURES DETECTOR + FLANN
        # fast = cv2.FastFeatureDetector_create()
        # kp1 = fast.detect(self.images[i - 1], mask=prev_mask)
        # kp2 = fast.detect(self.images[i], mask=curr_mask)
        # kp1, des1 = self.orb.compute(self.images[i - 1], kp1)
        # kp2, des2 = self.orb.compute(self.images[i], kp2)
        # matches = self.flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test:
        good = []   # Good matches
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)  # Then we keep the match. Otherwise, we discard it because the distance is too large.
        except ValueError:
            pass

        # Get the coordinates of the keypoints of the good matches:
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # 1. Filter out the matches whose distance is 3 times larger than the median distance:
        # median_y_distance = np.median(np.abs(q1[:, 1] - q2[:, 1]))
        # mask = np.array([np.abs(q1[:, 1] - q2[:, 1]) < 6 * median_y_distance]).squeeze()
        # q1 = q1[mask]
        # q2 = q2[mask]
        # good = [good[i] for i in range(len(mask)) if mask[i]]

        # 2. Filter out the matches whose distance is larger than 1/10th of the image height:
        image_height = self.images[i].shape[0]
        mask = np.array([np.abs(q1[:, 1] - q2[:, 1]) < image_height / 7]).squeeze()
        q1 = q1[mask]
        q2 = q2[mask]
        good = [good[i] for i in range(len(mask)) if mask[i]]

        # 3. Filter out the matches to keep only the matches that are in the 1/3th bottom of the image:
        # image_height = self.images[i].shape[0]
        # mask = np.array([q1[:, 1] > 2 * image_height / 3]).squeeze()
        # q1 = q1[mask]
        # q2 = q2[mask]
        # good = [good[i] for i in range(len(mask)) if mask[i]]

        # Show the matches:
        if show:
            draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
            img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1], kp2, good, None, **draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(200)

        return q1, q2

    def get_pose_from_matches(self, q1, q2):
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        R, t = self.decomp_essential_mat(E, q1, q2)

        transformation_matrix = form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]

    # STEREO METHODS=

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate the residuals

        Parameters
        ----------
        dof (ndarray): Transformation between the two frames. First 3 elements are the rotation vector and the last 3 is the translation. Shape (6)
        q1 (ndarray): Feature points in i-1'th image. Shape (n_points, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n_points, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n_points, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n_points, 3)

        Returns
        -------
        residuals (ndarray): The residuals. In shape (2 * n_points * 2)
        """
        # Get the rotation vector
        r = dof[:3]
        # Create the rotation matrix from the rotation vector
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = dof[3:]
        # Create the transformation matrix from the rotation matrix and translation vector
        transf = form_transf(R, t)

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # Project 3D points from i'th image to i-1'th image
        q1_pred = Q2.dot(f_projection.T)
        # Un-homogenize
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        # Project 3D points from i-1'th image to i'th image
        q2_pred = Q1.dot(b_projection.T)
        # Un-homogenize
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # Calculate the residuals
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        """
        Splits the image into tiles and detects the 10 best keypoints in each tile

        Parameters
        ----------
        img (ndarray): The image to find keypoints in. Shape (height, width)
        tile_h (int): The tile height
        tile_w (int): The tile width

        Returns
        -------
        kp_list (ndarray): A 1-D list of all keypoints. Shape (n_keypoints)
        """

        def get_kps(x, y):
            # Get the image tile
            impatch = img[y:y + tile_h, x:x + tile_w]

            # Detect keypoints
            keypoints = self.fastFeatures.detect(impatch)

            # Correct the coordinate for the point
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)

            # Get the 10 best keypoints
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda
                    x: -x.response)  # The reponse parameter is the strength of the keypoint (strengh means how well the keypoint can be described)
                return keypoints[:10]
            return keypoints

        # Get the image height and width
        h, w, *_ = img.shape

        # Get the keypoints for each of the tiles
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]

        # Flatten the keypoint list
        kp_list_flatten = np.concatenate(kp_list)
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames

        Parameters
        ----------
        img1 (ndarray): i-1'th image. Shape (height, width)
        img2 (ndarray): i'th image. Shape (height, width)
        kp1 (ndarray): Keypoints in the i-1'th image. Shape (n_keypoints)
        max_error (float): The maximum acceptable error

        Returns
        -------
        trackpoints1 (ndarray): The tracked keypoints for the i-1'th image. Shape (n_keypoints_match, 2)
        trackpoints2 (ndarray): The tracked keypoints for the i'th image. Shape (n_keypoints_match, 2)
        """
        # Convert the keypoints into a vector of points and expand the dims so we can select the good ones
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)

        # Use optical flow to find tracked counterparts
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        # Convert the status vector to boolean so we can use it as a mask
        trackable = st.astype(bool)

        # Create a maks there selects the keypoints there was trackable and under the max error
        under_thresh = np.where(err[trackable] < max_error, True, False)

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th left image. In shape (n_points, 2)
        q2 (ndarray): Feature points in i'th left image. In shape (n_points, 2)
        disp1 (ndarray): Disparity i-1'th image per. Shape (height, width)
        disp2 (ndarray): Disparity i'th image per. Shape (height, width)
        min_disp (float): The minimum disparity
        max_disp (float): The maximum disparity

        Returns
        -------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n_in_bounds, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n_in_bounds, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n_in_bounds, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n_in_bounds, 2)
        """

        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)

        # Combine the masks
        in_bounds = np.logical_and(mask1, mask2)

        # Get the feature points and disparity's there was in bounds
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]

        # Calculate the right feature points
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images

        Parameters
        ----------
        q1_l (ndarray): Feature points in i-1'th left image. In shape (n, 2)
        q1_r (ndarray): Feature points in i-1'th right image. In shape (n, 2)
        q2_l (ndarray): Feature points in i'th left image. In shape (n, 2)
        q2_r (ndarray): Feature points in i'th right image. In shape (n, 2)

        Returns
        -------
        Q1 (ndarray): 3D points seen from the i-1'th image. In shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. In shape (n, 3)
        """
        # Triangulate points from i-1'th image
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # Un-homogenize
        Q1 = np.transpose(Q1[:3] / Q1[3])

        # Triangulate points from i'th image
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # Un-homogenize
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): Feature points in i-1'th image. Shape (n, 2)
        q2 (ndarray): Feature points in i'th image. Shape (n, 2)
        Q1 (ndarray): 3D points seen from the i-1'th image. Shape (n, 3)
        Q2 (ndarray): 3D points seen from the i'th image. Shape (n, 3)
        max_iter (int): The maximum number of iterations

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix. Shape (4,4)
        """
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        # Get the rotation vector
        r = out_pose[:3]
        # Make the rotation matrix
        R, _ = cv2.Rodrigues(r)
        # Get the translation vector
        t = out_pose[3:]
        # Make the transformation matrix
        transformation_matrix = form_transf(R, t)
        return transformation_matrix

    def get_pose(self, i):
        """
        "(SCRATCH) - HOW TO STEPS:
        Calculates the transformation matrix for the i'th frame
        1. Get the left i-1'th image and i'th image (left only)
        2. Get the tiled keypoints using the i-1'th left image
        3. Track the keypoints from the i-1'th left image to the i'th left image using the kp1_l keypoints.
        4. Calculate the disparity between the i-1'th left image and the i'th right image, and add it to
        the list of disparities.
        5. Calculate the right i-1'th ad i'th keypoints using the both i-1'th and i'th left images and the
        i-1'th and i'th disparities.
        6. Calculate the 3D points using the four sets of keypoints.
        7. Obtain the transformation matrix using both the 3D points two sets of keypoints (either the left or right)
        of the i-1'th and i'th image.
        """

        # Get the i-1'th image and i'th image
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        # Get the tiled keypoints (top 10 best keypoints per tile)
        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)

        # Track the keypoints
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

        # Calculate the disparities
        self.disparities.append(np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

        # Calculate the right keypoints
        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])

        # Calculate the 3D points
        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

        # Estimate the transformation matrix
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)

        return transformation_matrix
