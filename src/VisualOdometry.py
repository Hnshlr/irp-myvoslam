import numpy as np
import cv2
import os

from src.Utils import load_calib, load_calib_v2, load_poses, load_images, form_transf

class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = load_calib_v2(os.path.join(data_dir, 'calib.txt'))
        dataset_index = data_dir.split("/")[-1]
        poses_dir_path = data_dir.replace(dataset_index, "poses")
        poses_file = os.path.join(poses_dir_path, dataset_index + ".txt")
        self.gt_poses = load_poses(poses_file)
        self.images = load_images(os.path.join(data_dir, "image_0"))
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def get_matches(self, i, show=True, prev_mask=None, curr_mask=None):
        # VERSION 1: ORB + FLANN
        # kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], mask=prev_mask)
        # kp2, des2 = self.orb.detectAndCompute(self.images[i], mask=curr_mask)
        # The two previous lines serve to find the keypoints and descriptors of the two images
        # Then we match the descriptors, and only keep the good matches.
        # matches = self.flann.knnMatch(des1, des2, k=2)

        # VERSION 2: FAST FEATURES DETECTOR + FLANN
        fast = cv2.FastFeatureDetector_create()
        kp1 = fast.detect(self.images[i - 1], mask=prev_mask)
        kp2 = fast.detect(self.images[i], mask=curr_mask)
        kp1, des1 = self.orb.compute(self.images[i - 1], kp1)
        kp2, des2 = self.orb.compute(self.images[i], kp2)
        matches = self.flann.knnMatch(des1, des2, k=2)

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
        mask = np.array([np.abs(q1[:, 1] - q2[:, 1]) < image_height / 10]).squeeze()
        q1 = q1[mask]
        q2 = q2[mask]
        good = [good[i] for i in range(len(mask)) if mask[i]]

        # 3. Filter out the matches to keep only the matches that are in the 1/3th bottom of the image:
        image_height = self.images[i].shape[0]
        mask = np.array([q1[:, 1] > 2 * image_height / 3]).squeeze()
        q1 = q1[mask]
        q2 = q2[mask]
        good = [good[i] for i in range(len(mask)) if mask[i]]

        # Show the matches:
        if show:
            draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
            img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1], kp2, good, None, **draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(200)

        return q1, q2

    def get_pose(self, q1, q2):
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