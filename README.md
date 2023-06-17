# AY22 Individual Research Project (A) 2022-2023

Autonomous Vehicle Self Awareness through use of SLAM algorithms.

© Copyright 2023, All rights reserved to Hans Haller, CSTE-CIDA Student at Cranfield Uni. SATM, Cranfield, UK.

https://www.github.com/Hnshlr

## Computer Vision: Visual Odometry with Monocular Camera

1. The necessary imports are made, including modules for visualization, file handling, and OpenCV.


2. The VisualOdometry class is defined, which initializes the object with the data directory. It loads the camera calibration, ground truth poses, images, and initializes the ORB feature detector and FLANN matcher.
- The ORB algorithm combines the FAST keypoint detector and the BRIEF descriptor to detect and describe distinctive features in images. FAST identifies potential keypoints by finding pixels with significant intensity differences from their surroundings. ORB then assigns a dominant orientation to each keypoint and computes a binary feature descriptor using pixel intensity comparisons. This combination of speed and efficiency makes ORB a popular choice for real-time applications where robust feature matching is required.


- FLANN is a library that provides efficient approximate nearest neighbor search algorithms. In the context of the code you provided, the FLANN-based matcher is used to find matches between the ORB descriptors of keypoints in consecutive image frames. It performs approximate nearest neighbor search to efficiently match descriptors based on their binary feature representations. Additionally, a distance ratio test is applied to filter out ambiguous matches by comparing the distances to the nearest and second-nearest neighbors. This combination of efficient matching and filtering helps improve the accuracy and robustness of feature matching for tasks such as visual odometry.

3. Several static methods are defined to load calibration data, ground truth poses, and images from files. The calib.txt file contains the calibration parameters of the camera. The format of the file is typically structured as each line represents a row of the 3x4 projection matrix, where:

f_x 0 c_x 0

0 f_y c_y 0

0 0 1 0


f_x and f_y are the focal lengths along the x-axis and y-axis, respectively; and c_x and c_y are the principal point coordinates (the optical center) along the x-axis and y-axis, respectively.

4. The form_transf method constructs a transformation matrix from a rotation matrix and translation vector. In computer vision and robotics, a transformation matrix is commonly used to represent the pose (position and orientation) of an object or camera in a 3D space.
A transformation matrix is a 4x4 matrix that combines rotation and translation. It has the following structure:

[R | t]

[0 0 0 1]

where R is a 3x3 rotation matrix representing the rotation part of the transformation, t is a 3-element translation vector representing the translation part of the transformation, and [0 0 0 1] is a row vector used for homogeneous coordinates.

5. The get_matches method is called for each consecutive pair of images. It detects keypoints and computes descriptors using the ORB feature detector for the previous and current images. It then matches the keypoints using the FLANN matcher, discards poor matches based on distance, and returns the positions of the good matches in the previous and current images.


6. The get_pose method is called with the matched keypoints. It calculates the essential matrix between the two sets of keypoints, decomposes it into rotation and translation using the decomp_essential_mat method, and returns the transformation matrix.


7. The decomp_essential_mat method decomposes the essential matrix and selects the correct rotation and translation based on the number of points with positive z-coordinate in both cameras. It returns the rotation and translation as a list.


8. A loop is executed for each ground truth pose in the gt_poses list. In the first iteration (i.e., when i == 0), the current pose is set as the ground truth pose. In subsequent iterations, the get_matches method is called to get the keypoints matches for the current frame. The get_pose method is then called to calculate the transformation matrix based on the matches. The current pose is updated by multiplying it with the inverse of the transformation matrix. 


9. The ground truth and estimated paths are collected in gt_path and estimated_path lists, respectively, for visualization.


This script performs visual odometry by estimating camera poses based on feature matches between consecutive frames. It accumulates the relative poses to compute the camera trajectory.