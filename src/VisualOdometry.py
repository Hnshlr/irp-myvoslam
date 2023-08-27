# IMPORTS=
from scipy.optimize import least_squares
from tkinter import *

# MODELS=
from src.Utils import *
from src.SemanticSegmentation import *

# THE TKINTER WINDOW (USED TO GATHER USER SCREEN WIDTH, FOR BETTER VISUALIZATION):
win = Tk()


# THE MAIN VISUAL ODOMETRY CLASS:
class VisualOdometry:
    def __init__(self, data_dir, method="mono", fd_parameters=None, pmor_parameters=None, ss_parameters=None,
                 fto_parameters=None):
        """
        Creates a Visual Odometry object, which can be used to estimate the path of an object using a dataset of
        consecutive images.
        :param data_dir: The path to the dataset directory
        :param method: The method to use (either "mono" or "stereo")
        :param fd_parameters: The feature detection dictionary parameters (see main.py for an example)
        :param pmor_parameters: The post matching outlier removal dictionary parameters (see main.py for an example)
        :param ss_parameters: The semantic segmentation dictionary parameters (see main.py for an example)
        :param fto_parameters: The frame tile optimization dictionary parameters (see main.py for an example)
        """

        # METHOD:
        self.method = method
        self.dataset_dir_path = data_dir

        # CALIBRATION, GROUND TRUTH POSES AND IMAGES:
        self.K_l, self.P_l, self.K_r, self.P_r = load_calib_LR(data_dir + '/calib.txt')
        self.gt_poses = load_poses(data_dir + '/poses.txt')
        self.images_l = load_images(data_dir + '/image_0')
        self.images_r = load_images(data_dir + '/image_1')

        # FEATURE DETECTION:
        self.feature_detection_parameters = fd_parameters \
            if fd_parameters is not None \
            else {"fda": "orb", "nfeatures": 3000}

        # PMOR:
        self.pmor_parameters = pmor_parameters \
            if pmor_parameters is not None \
            else {"do_PMOR": False, "do_xyMeanDist": False, "do_xyImgDist": False, "do_RANSAC": False}

        # SEMANTIC SEGMENTATION:
        self.semantic_segmentation = None
        if ss_parameters is not None:
            if ss_parameters["do_SS"]:
                self.semantic_segmentation = SemanticSegmentation(ss_parameters["model_path"],
                                                                  ss_parameters["features_to_ignore"])

        # FRAME TILE OPTIMIZATION (FTO):
        if fto_parameters is not None:
            self.fto_parameters = fto_parameters
            self.do_FTO = fto_parameters["do_FTO"]
            self.GRID_H = fto_parameters["grid_h"]
            self.GRID_W = fto_parameters["grid_w"]
            self.PATCH_MAX_FEATURES = fto_parameters["patch_max_features"]

        # PATHS= (FOR SEMANTIC SEGMENTATION)
        images_dir_path = os.path.join(self.dataset_dir_path, "image_0")
        self.images_paths = [os.path.join(images_dir_path, file) for file in sorted(os.listdir(images_dir_path))]
        self.images_paths.sort()

        # INITIALIZATION:
        if method == "mono":
            # FLANN MATCHER
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        elif method == "stereo":
            block = 11
            P1 = block * block * 8
            P2 = block * block * 32
            self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2)
            self.disparities = [
                np.divide(self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32), 16)]

            # FAST FEATURES DETECTOR + LK OPTICAL FLOW:
            self.fastFeatures = cv2.FastFeatureDetector_create()
            self.lk_params = dict(winSize=(15, 15), flags=cv2.MOTION_AFFINE, maxLevel=3,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))
        else:
            raise ValueError("Method must be 'mono' or 'stereo'.")

    # COMMON METHODS:
    # _ PATH ESTIMATION:
    def estimate_path(self, monitor=False, view=False, features=[None].clear()):
        """
        Main path estimation pipeline, which estimates the path of the object in the dataset using various methods
        defined later in the class.
        :param monitor: Boolean, whether to show the progress bar (tqdm).
        :param view: Boolean, whether to visualise the computation in real time.
        :param features: The features to use for the path estimation. If None, the features are computed from scratch.
        :return: The ground truth path and the estimated path.
        """

        gt_path, est_path = [], []
        cur_pose = None
        iterable = enumerate(
            tqdm(self.gt_poses, unit="pose", desc="Processing dataset", leave=False)) if monitor else enumerate(
            self.gt_poses)
        for i, gt_pose in iterable:
            if i == 0:  # First pose is the origin
                cur_pose = gt_pose
            else:
                transf = self.get_transformation(i,
                                                 show=view,
                                                 prev_mask=self.semantic_segmentation.get_total_upscaled_mask(
                                                     self.images_paths[
                                                         i - 1]) if self.semantic_segmentation is not None else None,
                                                 curr_mask=self.semantic_segmentation.get_total_upscaled_mask(
                                                     self.images_paths[
                                                         i]) if self.semantic_segmentation is not None else None
                                                 )  # Get the transformation matrix between the current and previous image
                cur_pose = np.matmul(cur_pose, transf)  # Update the current pose
            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))  # Append the ground truth path
            est_path.append((cur_pose[0, 3], cur_pose[2, 3]))  # Append the estimated path
        return gt_path, est_path

    # _ GET POSE:
    def get_transformation(self, i, show=True, prev_mask=None, curr_mask=None):
        """
        The main method for estimating the transformation matrix between two consecutive frames, which varies
        depending on the method used (either "mono" or "stereo").
        :param i: The index of the current image.
        :param show: Boolean, whether to visualise the computation in real time.
        :param prev_mask: The mask to apply to the previous image, if semantic segmentation is used.
        :param curr_mask: The mask to apply to the current image, if semantic segmentation is used.
        :return: The transformation matrix between the current and previous image.
        """

        # MONO:
        if self.method == "mono":
            q1, q2 = self.get_matches(i,
                                      show=show,
                                      prev_mask=prev_mask,
                                      curr_mask=curr_mask
                                      )  # Get the matches between the current and previous image

            transformation_matrix = self.estimate_pose_mono(q1, q2)

            return transformation_matrix

        # STEREO:
        elif self.method == "stereo":
            # Get the i-1'th image and i'th image
            img1_l, img2_l = self.images_l[i - 1:i + 1]
            img1_r, img2_r = self.images_r[i - 1:i + 1]  # Right images, solely for viewing/drawing purposes

            # Get the tiled keypoints (top 10 best keypoints per tile)
            kp1_l = self.get_keypoints(img1_l, do_FTO=self.do_FTO, GRID_H=self.GRID_H, GRID_W=self.GRID_W,
                                       max_kp_per_patch=self.PATCH_MAX_FEATURES)

            # Track the keypoints
            tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)

            # Calculate the disparities
            self.disparities.append(
                np.divide(self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32), 16))

            # Calculate the right keypoints
            tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1],
                                                                 self.disparities[i])

            # Calculate the 3D points
            Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)

            # Estimate the transformation matrix
            transformation_matrix = self.estimate_pose_stereo(tp1_l, tp2_l, Q1, Q2)

            if show:
                # Show the matches and the disparity map:
                draw_params = dict(singlePointColor=None, matchesMask=None, flags=2)
                tp1_l = [cv2.KeyPoint(x, y, 1) for x, y in tp1_l]
                tp2_l = [cv2.KeyPoint(x, y, 1) for x, y in tp2_l]
                tp1_r = [cv2.KeyPoint(x, y, 1) for x, y in tp1_r]
                tp2_r = [cv2.KeyPoint(x, y, 1) for x, y in tp2_r]
                matches = [cv2.DMatch(i, i, 0) for i in range(len(tp1_l))]
                img = cv2.drawMatches(img1_l, tp1_l, img2_l, tp2_l, matches, None, **draw_params)
                disparity = self.disparities[i - 1]
                disparity = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_32F)
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
                four_images = np.zeros((img1_l.shape[0] * 2, img1_l.shape[1] * 2, 3), dtype=np.uint8)
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

    # MONO METHODS:
    # _ GET MATCHES:
    def get_matches(self, i, show=False, prev_mask=None, curr_mask=None):
        """
        The main method for getting the matches between two consecutive frames, which varies depending on the feature
        detection algorithm used (either "orb", "fast" or "surf").
        :param i: The index of the current image.
        :param show: Boolean, whether to visualise the computation in real time.
        :param prev_mask: The mask to apply to the previous image, if semantic segmentation is used.
        :param curr_mask: The mask to apply to the current image, if semantic segmentation is used.
        :return: The keypoints of the previous and current image.
        """

        # CREATE ORB DETECTOR (MANDATORY FOR ALL VERSIONS):
        orb = cv2.ORB_create(nfeatures=self.feature_detection_parameters["nfeatures"])

        # VERSION 1: ORB
        if self.feature_detection_parameters["fda"] == "orb":
            kp1 = orb.detect(self.images_l[i - 1], mask=prev_mask)
            kp2 = orb.detect(self.images_l[i], mask=curr_mask)

        # VERSION 2: FAST (USING ORB TO COMPUTE DESCRIPTORS, AS FAST DOES NOT SUPPORT IT)
        elif self.feature_detection_parameters["fda"] == "fast":
            fast = cv2.FastFeatureDetector_create()
            kp1 = fast.detect(self.images_l[i - 1], mask=prev_mask)
            kp2 = fast.detect(self.images_l[i], mask=curr_mask)

        # VERSION 3: SURF (USING ORB TO COMPUTE DESCRIPTORS, AS SURF DOES NOT SUPPORT IT)
        elif self.feature_detection_parameters["fda"] == "surf":
            try:
                surf = cv2.xfeatures2d.SURF_create()
                kp1 = surf.detect(self.images_l[i - 1], mask=prev_mask)
                kp2 = surf.detect(self.images_l[i], mask=curr_mask)
            except AttributeError:
                print("In Python 3.9, there are no OpenCV/OpenCV-Contrib versions that support SURF, "
                      "as it is patented. Please use Python 3.7 or lower, or choose another "
                      "feature detection algorithm.")
                exit()

        # COMPUTE DESCRIPTORS:
        kp1, des1 = orb.compute(self.images_l[i - 1], kp1)
        kp2, des2 = orb.compute(self.images_l[i], kp2)

        # FLANN:
        matches = self.flann.knnMatch(des1, des2, k=2)

        # LOWE'S RATIO TEST:
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        # GET THE KEYPOINTS:
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # POST MATCHING OUTLIER REMOVAL (PMOR):
        if self.pmor_parameters["do_PMOR"]:
            # MATCHES DIMENSIONS STRATEGY:
            if self.pmor_parameters["do_xyMeanDist"]:
                # USING THE MEAN X-DISTANCE:
                median_x_distance = np.mean(np.abs(q1[:, 0] - q2[:, 0]))
                mask = np.array([np.abs(q1[:, 0] - q2[:, 0]) < 5 * median_x_distance]).squeeze()
                q1 = q1[mask]
                q2 = q2[mask]
                good = [good[i] for i in range(len(mask)) if mask[i]]

                # USING THE MEAN Y-DISTANCE:
                median_y_distance = np.mean(np.abs(q1[:, 1] - q2[:, 1]))
                mask = np.array([np.abs(q1[:, 1] - q2[:, 1]) < 5 * median_y_distance]).squeeze()
                q1 = q1[mask]
                q2 = q2[mask]
                good = [good[i] for i in range(len(mask)) if mask[i]]

            # IMAGE DIMENSION STRATEGY:
            if self.pmor_parameters["do_xyImgDist"]:
                # USING THE X DIMENSION:
                mask = np.array([np.abs(q1[:, 0] - q2[:, 0]) < self.images_l[i].shape[1] / 5]).squeeze()
                q1 = q1[mask]
                q2 = q2[mask]
                good = [good[i] for i in range(len(mask)) if mask[i]]

                # USING THE Y DIMENSION:
                mask = np.array([np.abs(q1[:, 1] - q2[:, 1]) < self.images_l[i].shape[0] / 5]).squeeze()
                q1 = q1[mask]
                q2 = q2[mask]
                good = [good[i] for i in range(len(mask)) if mask[i]]

            # RANSAC:
            if self.pmor_parameters["do_RANSAC"]:
                transf, mask = cv2.findHomography(q1, q2, cv2.RANSAC, 5.0)
                mask = np.array([True if mask[i][0] == 1 else False for i in range(len(mask))])
                q1 = q1[mask]
                q2 = q2[mask]
                good = [good[i] for i in range(len(mask)) if mask[i]]

        if show:
            # KEYPOINTS:
            img1 = cv2.drawKeypoints(self.images_l[i - 1], kp1, None, color=(255, 0, 0))
            img2 = cv2.drawKeypoints(self.images_l[i], kp2, None, color=(255, 0, 0))
            img23 = np.concatenate((img1, img2), axis=1)

            # MATCHES:
            draw_params = dict(singlePointColor=None, matchesMask=None, flags=2)
            img3 = cv2.drawMatches(self.images_l[i], kp1, self.images_l[i - 1], kp2, good, None, **draw_params)

            # OPTICAL FLOW:
            img4 = self.images_l[i].copy()
            for i in range(len(q1)):
                cv2.arrowedLine(img4, tuple(map(int, q1[i])), tuple(map(int, q2[i])), (255, 0, 0), 1)
            img4 = cv2.cvtColor(img4, cv2.COLOR_GRAY2BGR)

            # STACK THE IMAGES:
            empty = np.zeros(img4.shape, dtype=np.uint8)
            third = np.hstack((img4, empty))
            final_image = np.vstack((img23, img3, third))

            # IMAGE RESIZING AND FITTING (USING TKINTER):
            screen_width = win.winfo_screenwidth()
            final_image = ResizeWithAspectRatio(final_image, width=screen_width)

            # SHOW THE IMAGE:
            cv2.imshow('final_image', final_image)
            cv2.waitKey(1)
        return q1, q2

    # _ DECOMPOSE ESSENTIAL MATRIX:
    def estimate_pose_mono(self, q1, q2):
        """
        Estimates the transformation matrix between two consecutive frames using the 5-point algorithm.
        :param q1: The keypoints of the previous image.
        :param q2: The keypoints of the current image.
        :return: The transformation matrix between the current and previous image.
        """
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
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) /
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # FIND ESSENTIAL MATRIX:
        E, _ = cv2.findEssentialMat(q1, q2, self.K_l, threshold=1)

        # DECOMPOSE ESSENTIAL MATRIX:
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # POSSIBLE SOLUTIONS:
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # SOLUTION SELECTION:
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        # FORM TRANSFORMATION MATRIX:
        transformation_matrix = form_transf(R1, np.squeeze(t))
        transformation_matrix = np.linalg.inv(transformation_matrix)
        return transformation_matrix

    # STEREO METHODS:
    # _ RE-PROJECTION RESIDUALS:
    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculates the reprojection residuals between the current and previous image using set of matches and the 3D
        points.
        :param dof: The degrees of freedom (the rotation and translation vector).
        :param q1: The keypoints of the previous image.
        :param q2: The keypoints of the current image.
        :param Q1: The 3D points of the previous image.
        :param Q2: The 3D points of the current image.
        :return: The reprojection residuals.
        """
        # CONSTRUCT THE TRANSFORMATION MATRIX FROM THE ROTATION MATRIX AND TRANSLATION VECTOR:
        r = dof[:3]
        R, _ = cv2.Rodrigues(r)
        t = dof[3:]
        transf = form_transf(R, t)

        # PROJECTION MATRICES:
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # HOMOGENIZE THE 3D POINTS:
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        # 3D-PROJECTION AND UN-HOMOGENIZATION:
        q1_pred = Q2.dot(f_projection.T)
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]
        q2_pred = Q1.dot(b_projection.T)
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        # CALCULATE THE RESIDUALS:
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    # _ GET KEYPOINTS:
    def get_keypoints(self, img, do_FTO=True, GRID_H=10, GRID_W=20, max_kp_per_patch=10):
        """
        Gets the keypoints of the image, either using the frame tiling optimization (FTO) or not.
        :param img: The image to get the keypoints from.
        :param do_FTO: Boolean, whether to use FTO.
        :param GRID_H: The amount of tiles in along the height of the image.
        :param GRID_W: The amount of tiles in along the width of the image.
        :param max_kp_per_patch: The maximum amount of keypoints per tile.
        :return: The keypoints of the image.
        """
        def get_kps(x, y):
            """
            Gets the keypoints of a tile.
            :param x: The x-coordinate of the tile.
            :param y: The y-coordinate of the tile.
            :return: The keypoints of the tile.
            """
            # PATCH/TILE THE IMAGE:
            impatch = img[y:y + tile_h, x:x + tile_w]
            # DETECT THE KEYPOINTS ON THE PATCH/TILE:
            keypoints = self.fastFeatures.detect(impatch)
            # SHIFT THE KEYPOINTS TO THE CORRECT POSITION:
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
            # SORT THE KEYPOINTS BY STRENGTH:
            if len(keypoints) > max_kp_per_patch:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                # RETURN THE TOP 10 KEYPOINTS:
                return keypoints[:10]
            return keypoints

        # IF FTO IS USED, GET THE KEYPOINTS OF EACH TILE AND CONCATENATE THEM:
        if do_FTO:
            # GET THE TILE HEIGHT AND WIDTH:
            h, w, *_ = img.shape
            tile_h = h // GRID_H
            tile_w = w // GRID_W
            # GET THE KEYPOINTS OF EACH TILE:
            kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]
            kp_list_flatten = np.concatenate(kp_list)
            return kp_list_flatten

        # OTHERWISE, JUST GET THE KEYPOINTS OF THE IMAGE:
        else:
            max_total_kp_amount = 3000
            keypoints = self.fastFeatures.detect(img)
            keypoints = sorted(keypoints, key=lambda x: -x.response)[:max_total_kp_amount]
            for pt in keypoints:
                pt.pt = (pt.pt[0], pt.pt[1])
            return keypoints

    # _ TRACK KEYPOINTS:
    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between two consecutive frames using the Lucas-Kanade optical flow algorithm.
        :param img1: The previous image.
        :param img2: The current image.
        :param kp1: The keypoints of the previous image.
        :param max_error: The maximum error allowed for a tracked keypoint.
        :return: The tracked keypoints of the previous and current image.
        """
        # CONVERT THE KEYPOINTS TO TRACKABLE POINTS:
        tp1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
        # TRACK THE KEYPOINTS:
        tp2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, tp1, None, **self.lk_params)
        trackable = st.astype(bool)
        # FILTER OUT THE KEYPOINTS THAT ARE NOT TRACKED OR HAVE A TOO HIGH ERROR:
        under_thresh = np.where(err[trackable] < max_error, True, False)
        tp1 = tp1[trackable][under_thresh]
        tp2 = np.around(tp2[trackable][under_thresh])
        # FILTER OUT THE KEYPOINTS THAT ARE OUT OF BOUNDS:
        h, w = img1.shape
        in_bounds = np.where(np.logical_and(tp2[:, 1] < h, tp2[:, 0] < w), True, False)
        tp1 = tp1[in_bounds]
        tp2 = tp2[in_bounds]
        return tp1, tp2

    # _ CALCULATE RIGHT KEYPOINTS:
    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints of the previous and current image, using the disparity maps, and return the
        left and right keypoints of the previous and current image.
        :param q1: The keypoints of the previous image.
        :param q2: The keypoints of the current image.
        :param disp1: The disparity map of the previous image.
        :param disp2: The disparity map of the current image.
        :param min_disp: The minimum disparity.
        :param max_disp: The maximum disparity.
        :return: The left and right keypoints of the previous and current image.
        """
        def get_idxs(q, disp):
            """
            Gets the disparity's for the feature points and mask for min_disp & max_disp.
            :param q: The keypoints.
            :param disp: The disparity map.
            :return: The disparity's for the feature points and mask for min_disp & max_disp.
            """
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)
        # GET THE DISPARITY'S FOR THE FEATURE POINTS AND MASK FOR MIN_DISP & MAX_DISP:
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)
        # COMBINE THE MASKS:
        in_bounds = np.logical_and(mask1, mask2)
        # FILTER OUT THE KEYPOINTS THAT ARE OUT OF BOUNDS:
        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]
        # CALCULATE THE RIGHT KEYPOINTS:
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        return q1_l, q1_r, q2_l, q2_r

    # _ CALCULATE 3D POINTS:
    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Calculates the 3D points of the previous and current image using the left and right keypoints, of both the
        previous and current image (tracked keypoints).
        :param q1_l: The left keypoints of the previous image.
        :param q1_r: The right keypoints of the previous image.
        :param q2_l: The left keypoints of the current image.
        :param q2_r: The right keypoints of the current image.
        :return: The 3D points of the previous and current image.
        """
        # TRIANGULATE POINTS:
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        # UN-HOMOGENIZE:
        Q1 = np.transpose(Q1[:3] / Q1[3])
        # TRIANGULATE POINTS:
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        # UN-HOMOGENIZE:
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    # _ ESTIMATE POSE:
    def estimate_pose_stereo(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix between two consecutive frames using the least squares optimization
        method (Levenberg-Marquardt algorithm). The method is repeated until the error does not improve for a certain
        amount of iterations.
        :param q1: The matched keypoints of the previous image.
        :param q2: The matched keypoints of the current image.
        :param Q1: The 3D points of the previous image.
        :param Q2: The 3D points of the current image.
        :param max_iter: The maximum amount of iterations.
        :return: The transformation matrix between the current and previous image.
        """
        # SET THE PARAMETERS:
        early_termination_threshold = 5
        min_error = float('inf')
        early_termination = 0

        # ITERATE UNTIL MAX_ITER IS REACHED, OR THE ERROR DOES NOT IMPROVE FOR A CERTAIN AMOUNT OF ITERATIONS:
        for i in range(max_iter):
            # SAMPLE 6 RANDOM POINTS:
            sample_idx = (np.arange(i, i + 6) % q1.shape[0]) if q1.shape[0] > 6 else np.arange(q1.shape[0])
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # INITIAL GUESS:
            in_guess = np.zeros(6)
            # OPTIMIZE THE TRANSFORMATION MATRIX USING LEAST SQUARES:
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                    args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

            # CALCULATE THE ERROR:
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # CHECK IF THE ERROR IMPROVED:
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_threshold:
                break

        # CONSTRUCT THE TRANSFORMATION MATRIX FROM THE ROTATION MATRIX AND TRANSLATION VECTOR:
        r = out_pose[:3]
        R, _ = cv2.Rodrigues(r)
        t = out_pose[3:]
        transformation_matrix = form_transf(R, t)
        return transformation_matrix
