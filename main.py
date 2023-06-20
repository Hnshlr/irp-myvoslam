# IMPORTS=
import numpy as np
from tqdm import tqdm
import os
import time

# SEMANTIC SEGMENTATION=
import pixellib
from pixellib.semantic import semantic_segmentation     # second model

# MODELS=
from src.VisualOdometry import VisualOdometry
from src.Utils import load_images

def main():
    # SETTINGS=
    input_dir = "src/data/input/kitti/"
    dataset_indexes = [["S1", "S2", "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"][0]]
    datasets = [os.path.join(input_dir, dataset_index) for dataset_index in dataset_indexes]

    # MAIN=
    csv_string = "dataset, x_final_diff, y_final_diff, x_mean_diff, y_mean_diff, x_max_diff, y_max_diff\n"
    for dataset in datasets:
        # PATHS=
        images_path = os.path.join(dataset, "image_0")
        images_paths = [os.path.join(images_path, file) for file in sorted(os.listdir(images_path))]
        images_paths.sort()

        # SEMANTIC SEGMENTATION:
        seg = semantic_segmentation()
        seg.load_ade20k_model("src/models/deeplabv3_xception65_ade20k.h5")
        def get_upscaled_mask(image_path, features):
            segvalues, masks, output = seg.segmentAsAde20k(image_path, overlay=True, extract_segmented_objects=True)
            # From the masks list, get the element where class_name = "building" (segvalues is a dictionary with the class names):
            masks = [mask for mask in masks if mask['class_name'] in features]
            masks = [mask['masks'] for mask in masks]
            dimx, dimy = masks[0].shape
            mask = np.zeros((dimx, dimy), dtype=bool)
            for i in range(dimx):
                for j in range(dimy):
                    mask[i, j] = False
                    for m in masks:
                        if m[i, j]:
                            mask[i, j] = True
                            break

            # The mask (true/false) is 512x154, so we need to resize it to the image size (1226x370):
            upscaled_mask = np.zeros((370, 1226), dtype=np.uint8)
            ratio = 370/154
            for i in range(upscaled_mask.shape[0]):
                for j in range(upscaled_mask.shape[1]):
                    upscaled_mask[i, j] = 255 if mask[int(i/ratio), int(j/ratio)] else 0
            return upscaled_mask

        # VISUAL ODOMETRY:
        vo = VisualOdometry(dataset)    # Initialize the Visual Odometry class
        gt_path, est_path = [], []
        for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose", desc="Processing dataset")):
            if i == 0:  # First pose is the origin
                cur_pose = gt_pose
            else:
                features = ["earth", "grass", "sidewalk", "road", "car", "building"]
                q1, q2 = vo.get_matches(i, show=True, prev_mask=get_upscaled_mask(images_paths[i-1], features=features), curr_mask=get_upscaled_mask(images_paths[i], features=features))  # Get the matches between the current and previous image
                transf = vo.get_pose(q1, q2)    # Get the transformation matrix between the current and previous image
                cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))   # Update the current pose
            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))  # Append the ground truth path
            est_path.append((cur_pose[0, 3], cur_pose[2, 3]))   # Append the estimated path

        # TESTS=
        x_final_diff = np.abs(np.round(gt_path[-1][0] - est_path[-1][0], 2))
        y_final_diff = np.abs(np.round(gt_path[-1][1] - est_path[-1][1], 2))
        x_mean_diff = np.abs(np.round(np.mean([gt_path[i][0] - est_path[i][0] for i in range(len(gt_path))]), 2))
        y_mean_diff = np.abs(np.round(np.mean([gt_path[i][1] - est_path[i][1] for i in range(len(gt_path))]), 2))
        x_max_diff = np.abs(np.round(np.max([gt_path[i][0] - est_path[i][0] for i in range(len(gt_path))]), 2))
        y_max_diff = np.abs(np.round(np.max([gt_path[i][1] - est_path[i][1] for i in range(len(gt_path))]), 2))
        print("x_final_diff:", x_final_diff, " | y_final_diff:", y_final_diff, " | x_mean_diff:", x_mean_diff, " | y_mean_diff:", y_mean_diff, " | x_max_diff:", x_max_diff, " | y_max_diff:", y_max_diff)
        csv_string += f"{dataset}, {x_final_diff}, {y_final_diff}, {x_mean_diff}, {y_mean_diff}, {x_max_diff}, {y_max_diff}\n"
        time.sleep(1)

    # SAVE RESULTS=
    with open("src/data/output/results.csv", "w") as f:
        f.write(csv_string)


if __name__ == "__main__":
    main()
