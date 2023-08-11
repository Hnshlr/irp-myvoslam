# IMPORTS=
from scipy.optimize import least_squares
from tkinter import *
win= Tk()

# MODELS=
from src.Utils import *
from src.SemanticSegmentation import *

class VisualOdometry():
    def __init__(self, data_dir, method="mono", semantic_segmentation_parameters=None, fto_parameters=None):
        self.method = method
        self.dataset_dir_path = data_dir

        # CALIBRATION, GROUND TRUTH POSES AND IMAGES:
        self.K_l, self.P_l, self.K_r, self.P_r = load_calib_LR(data_dir + '/calib.txt')
        self.gt_poses = load_poses(data_dir + '/poses.txt')
        self.images_l = load_images(data_dir + '/image_0')
        self.images_r = load_images(data_dir + '/image_1')

        # SEMANTIC SEGMENTATION:
        self.semantic_segmentation = None
        if semantic_segmentation_parameters is not None:
            if semantic_segmentation_parameters["segmentate"]:
                self.semantic_segmentation = SemanticSegmentation(semantic_segmentation_parameters["model_path"], semantic_segmentation_parameters["features_to_ignore"])

        # FRAME TILE OPTIMIZATION (FTO):
        self.fto_parameters = fto_parameters
        self.do_FTO = fto_parameters["do_FTO"]
        self.GRID_H = fto_parameters["grid_h"]
        self.GRID_W = fto_parameters["grid_w"]
        self.PATCH_MAX_FEATURES = fto_parameters["patch_max_features"]

        # PATHS= (FOR SEMANTIC SEGMENTATION)
        images_dir_path = os.path.join(self.dataset_dir_path, "image_0")
        self.images_paths = [os.path.join(images_dir_path, file) for file in sorted(os.listdir(images_dir_path))]
        self.images_paths.sort()

        if method == "mono":
            # ORB+FLANN:
            self.orb = cv2.ORB_create(3000)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        elif method == "stereo":
            block = 11
            P1 = block * block * 8
            P2 = block * block * 32
            self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
            self.disparities = [np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]

            # FAST FEATURES DETECTOR + LK OPTICAL FLOW:
            self.fastFeatures = cv2.FastFeatureDetector_create()
            self.lk_params = dict(winSize=(15, 15), flags=cv2.MOTION_AFFINE, maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        else:
            raise ValueError("Method must be 'mono' or 'stereo'.")

    # PATH ESTIMATION (MONO/STEREO):
    def estimate_path(self, monitor=False, view=False, features=[None].clear()):
        gt_path, est_path = [], []
        cur_pose = None
        iterable = enumerate(tqdm(self.gt_poses, unit="pose", desc="Processing dataset", leave=False)) if monitor else enumerate(self.gt_poses)
        for i, gt_pose in iterable:
            if i == 0:  # First pose is the origin
                cur_pose = gt_pose
            else:
                transf = self.get_pose(i,
                                       show=view,
                                       prev_mask=self.semantic_segmentation.get_total_upscaled_mask(self.images_paths[i - 1]) if self.semantic_segmentation is not None else None,
                                       curr_mask=self.semantic_segmentation.get_total_upscaled_mask(self.images_paths[i]) if self.semantic_segmentation is not None else None
                                       )  # Get the transformation matrix between the current and previous image
                cur_pose = np.matmul(cur_pose, transf)  # Update the current pose
            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))  # Append the ground truth path
            est_path.append((cur_pose[0, 3], cur_pose[2, 3]))  # Append the estimated path
        return gt_path, est_path

    # MONO METHODS=

    def get_matches(self, i, show=True, prev_mask=None, curr_mask=None):
        # VERSION 1: ORB
        # kp1, des1 = self.orb.detectAndCompute(self.images_l[i - 1], mask=prev_mask)
        # kp2, des2 = self.orb.detectAndCompute(self.images_l[i], mask=curr_mask)

        # VERSION 2: FAST (USING ORB TO COMPUTE DESCRIPTORS, AS FAST DOES NOT SUPPORT IT)
        # fast = cv2.FastFeatureDetector_create()
        # kp1 = fast.detect(self.images_l[i - 1], mask=prev_mask)
        # kp2 = fast.detect(self.images_l[i], mask=curr_mask)
        # kp1, des1 = self.orb.compute(self.images_l[i - 1], kp1)
        # kp2, des2 = self.orb.compute(self.images_l[i], kp2)

        # VERSION 3: SURF (USING ORB TO COMPUTE DESCRIPTORS, AS SURF DOES NOT SUPPORT IT)
        surf = cv2.xfeatures2d.SURF_create()
        kp1 = surf.detect(self.images_l[i - 1], mask=prev_mask)
        kp2 = surf.detect(self.images_l[i], mask=curr_mask)
        kp1, des1 = self.orb.compute(self.images_l[i - 1], kp1)
        kp2, des2 = self.orb.compute(self.images_l[i], kp2)

        # FLANN:
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test:
        good = []   # Good matches
        try:
            for m, n in matches:    # m is the best match, n is the second best match
                if m.distance < 0.8 * n.distance:
                    good.append(m)  # Then we keep the match. Otherwise, we discard it because the distance is too large.
                    # 0.8 is
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
        image_height = self.images_l[i].shape[0]
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

        # 4. Filter out the matches whose x distance is 3 times larger than the median x distance:
        median_x_distance = np.median(np.abs(q1[:, 0] - q2[:, 0]))
        mask = np.array([np.abs(q1[:, 0] - q2[:, 0]) < 3 * median_x_distance]).squeeze()
        q1 = q1[mask]
        q2 = q2[mask]
        good = [good[i] for i in range(len(mask)) if mask[i]]

        if show:
            # Show the keypoints of the good matches:
            img1 = cv2.drawKeypoints(self.images_l[i - 1], kp1, None, color=(255, 0, 0))
            img2 = cv2.drawKeypoints(self.images_l[i], kp2, None, color=(255, 0, 0))
            img23 = np.concatenate((img1, img2), axis=1)
            # cv2.imshow("image23", img23)

            # Show the good matches:
            draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
            img3 = cv2.drawMatches(self.images_l[i], kp1, self.images_l[i-1], kp2, good, None, **draw_params)
            # cv2.imshow("image3", img3)

            # Draw the optical flow:
            img4 = self.images_l[i].copy()
            for i in range(len(q1)):
                cv2.arrowedLine(img4, tuple(map(int, q1[i])), tuple(map(int, q2[i])), (255, 0, 0), 1)
            img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)
            # cv2.imshow("image4", img4)

            empty = np.zeros(img4.shape, dtype=np.uint8)
            third = np.hstack((img4, empty))
            final_image = np.vstack((img23, img3, third))

            screen_width = win.winfo_screenwidth()
            final_image = ResizeWithAspectRatio(final_image, width=screen_width)

            cv2.imshow('final_image', final_image)
            cv2.waitKey(1)

        return q1, q2

    def decomp_essential_mat(self, E, q1, q2):
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K_l, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P_l, P, q1.T, q2.T)
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

    def get_keypoints(self, img, do_FTO=True, GRID_H=10, GRID_W=20, max_kp_per_patch=10):

        def get_kps(x, y):
            impatch = img[y:y + tile_h, x:x + tile_w]
            keypoints = self.fastFeatures.detect(impatch)
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
            if len(keypoints) > max_kp_per_patch:
                keypoints = sorted(keypoints, key=lambda
                    x: -x.response)  # The reponse parameter is the strength of the keypoint (strengh means how well the keypoint can be described)
                return keypoints[:10]
            return keypoints

        if do_FTO:
            h, w, *_ = img.shape
            tile_h = h // GRID_H
            tile_w = w // GRID_W
            kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]
            kp_list_flatten = np.concatenate(kp_list)
            return kp_list_flatten

        else:
            max_total_kp_amount = 3000
            keypoints = self.fastFeatures.detect(img)
            keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_total_kp_amount]
            for pt in keypoints:
                pt.pt = (pt.pt[0], pt.pt[1])
            return keypoints

    def track_keypoints(self, img1,  img2, kp1, max_error=4):
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


        for i in range(max_iter):
            # Choose 6 random feature points
            # sample_idx = np.random.choice(range(q1.shape[0]), 6)

            # Choose 6 feature points using the iterator i and the number of feature points, starting from 0:
            sample_idx = (np.arange(i, i + 6) % q1.shape[0]) if q1.shape[0] > 6 else np.arange(q1.shape[0])

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

            # Check if the error is less than the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                # If we have not found any better result in early_termination_threshold iterations
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

    # COMMON:

    def get_pose(self, i, show=True, prev_mask=None, curr_mask=None):

        if self.method == "mono":
            q1, q2 = self.get_matches(i,
                                      show=show,
                                      prev_mask=prev_mask,
                                      curr_mask=curr_mask
                                      )  # Get the matches between the current and previous image

            E, _ = cv2.findEssentialMat(q1, q2, self.K_l, threshold=1)

            R, t = self.decomp_essential_mat(E, q1, q2)

            transformation_matrix = form_transf(R, np.squeeze(t))
            transformation_matrix = np.linalg.inv(transformation_matrix)
            return transformation_matrix

        elif self.method == "stereo":

            # Get the i-1'th image and i'th image
            img1_l, img2_l = self.images_l[i - 1:i + 1]
            img1_r, img2_r = self.images_r[i - 1:i + 1]     # Right images, solely for viewing/drawing purposes

            # Get the tiled keypoints (top 10 best keypoints per tile)
            kp1_l = self.get_keypoints(img1_l, do_FTO=self.do_FTO, GRID_H=self.GRID_H, GRID_W=self.GRID_W, max_kp_per_patch=self.PATCH_MAX_FEATURES)

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

            if show:
                # Show the matches and the disparity map:
                draw_params = dict(matchColor=-1, singlePointColor=None, matchesMask=None, flags=2)
                tp1_l = [cv2.KeyPoint(x, y, 1) for x, y in tp1_l]
                tp2_l = [cv2.KeyPoint(x, y, 1) for x, y in tp2_l]
                tp1_r = [cv2.KeyPoint(x, y, 1) for x, y in tp1_r]
                tp2_r = [cv2.KeyPoint(x, y, 1) for x, y in tp2_r]
                matches = [cv2.DMatch(i, i, 0) for i in range(len(tp1_l))]
                img = cv2.drawMatches(img1_l, tp1_l, img2_l, tp2_l, matches, None, **draw_params)
                disparity = self.disparities[i - 1]
                disparity = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # Denormalize the disparity map:
                disparity = np.uint8(disparity * 255)
                disparity = cv2.cvtColor(disparity, cv2.COLOR_GRAY2BGR)
                # Show the 2D keypoints:
                img1_l = cv2.drawKeypoints(img1_l, kp1_l, None, color=(255, 0, 0), flags=0)
                if self.do_FTO:
                    grid_img = 255 * np.ones((img1_l.shape[0], img1_l.shape[1]), dtype=np.uint8)
                    tile_height = int(img1_l.shape[0] / self.GRID_H)
                    tile_width = int(img1_l.shape[1] / self.GRID_W)
                    # Draw a black and white tile grid:
                    for row in range(self.GRID_H):
                        for col in range(self.GRID_W):
                            top_left = (col * tile_width, row * tile_height)
                            bottom_right = ((col + 1) * tile_width, (row + 1) * tile_height)
                            if (row + col) % 2 == 0:
                                cv2.rectangle(grid_img, top_left, bottom_right, 0, -1)
                else:
                    grid_img = 255 * np.ones((img1_l.shape[0], img1_l.shape[1]), dtype=np.uint8)
                grid_img = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)

                # Draw the image for the TRACKED keypoints, and stack them:
                four_images = np.zeros((img1_l.shape[0]*2, img1_l.shape[1]*2, 3), dtype=np.uint8)
                t_img1_l = cv2.drawKeypoints(img1_l, tp1_l, None, color=(255, 0, 0), flags=0)
                t_img2_l = cv2.drawKeypoints(img2_l, tp2_l, None, color=(255, 0, 0), flags=0)
                t_img1_r = cv2.drawKeypoints(img1_r, tp1_r, None, color=(255, 0, 0), flags=0)
                t_img2_r = cv2.drawKeypoints(img2_r, tp2_r, None, color=(255, 0, 0), flags=0)
                four_images = np.vstack((np.hstack((t_img2_l, t_img2_r)), np.hstack((t_img1_l, t_img1_r))))
                four_images = cv2.resize(four_images, (img1_l.shape[1], img1_l.shape[0]))

                first = img
                second = np.hstack((img1_l, four_images))
                third = np.hstack((grid_img, disparity))
                final_image = np.vstack((first, second, third))

                screen_width = win.winfo_screenwidth()
                final_image = ResizeWithAspectRatio(final_image, width=screen_width)

                cv2.imshow('final_image', final_image)

                cv2.waitKey(1)

            return transformation_matrix
