import numpy as np
import os
import cv2
from pixellib.semantic import semantic_segmentation     # second model

class SemanticSegmentation:
    def __init__(self, model_path, features_to_ignore):
        self.seg = semantic_segmentation()
        self.seg.load_ade20k_model(model_path)
        self.features_to_ignore = features_to_ignore

    def get_total_upscaled_mask(self, image_path):
        """
        Returns the upscaled mask of the image, containing all the features in the list.
        :param image_path: Path to the image to segment.
        :param features: List of features to segment (ie. ["person", "car", "road"]).
        :return: The upscaled mask (meaning that it has the same dimensions as the original image) of the image,
        corresponding to the region for the feature detection algorithm to look for the features in.
        """
        if not self.features_to_ignore:
            return None
        # Remove the last two directories from the path:
        masks_dir_path = os.path.join(*image_path.split("/")[:-2], "masks/")
        masks_path = os.path.join(masks_dir_path, image_path.split("/")[-1].replace(".png", ".npy"))
        # If the masks directory does not exist, create it:
        if not os.path.exists(masks_dir_path):
            os.makedirs(masks_dir_path)
        # If the masks file already exists, load it:
        if os.path.exists(masks_path):
            masks = np.load(masks_path, allow_pickle=True)
        # Otherwise, create it:
        else:
            # Extract the masks, and filter them by the features wanted:
            temp_path = os.path.join(*image_path.split("/")[:-2], "temp.png")
            # Convert the image to a 3-channel image, and save it to a temporary file:
            # NB: Required because the segmentation algorithm only works with 3-channel images.
            image = cv2.imread(image_path)
            cv2.imwrite(temp_path, image)
            # Segment the image:
            segvalues, masks, output = self.seg.segmentAsAde20k(temp_path, overlay=True, extract_segmented_objects=True)
            # Remove the temporary file:
            os.remove(temp_path)
            # Save the masks to a file:
            np.save(masks_path, masks)
        masks = [mask['masks'] for mask in masks if mask['class_name'] not in self.features_to_ignore]
        mask = np.any(masks, axis=0).astype(np.uint8)
        image = cv2.imread(image_path)
        upscaled_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) * 255
        return upscaled_mask