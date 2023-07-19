import numpy as np
import os
from pixellib.semantic import semantic_segmentation     # second model

class SemanticSegmentation:
    def __init__(self, model_path, features):
        self.seg = semantic_segmentation()
        self.model = self.seg.load_ade20k_model(model_path)
        self.features = features

    def get_total_upscaled_mask(self, image_path):
        """
        Returns the upscaled mask of the image, containing all the features in the list.
        :param image_path: Path to the image to segment.
        :param features: List of features to segment (ie. ["person", "car", "road"]).
        :return: The upscaled mask (meaning that it has the same dimensions as the original image) of the image,
        corresponding to the region for the feature detection algorithm to look for the features in.
        """
        if not self.features:
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
            segvalues, masks, output = self.model.segmentAsAde20k(image_path, overlay=True, extract_segmented_objects=True)
            # Save the masks to a file:
            np.save(masks_path, masks)
        masks = [mask for mask in masks if mask['class_name'] in self.features]
        masks = [mask['masks'] for mask in masks]
        dimx, dimy = masks[0].shape
        mask = np.zeros((dimx, dimy), dtype=bool)
        # Merge the masks:
        for i in range(dimx):
            for j in range(dimy):
                mask[i, j] = any([mask[i, j] for mask in masks])    # Returns True if at least one mask is True
        # Upscale the mask:
        upscaled_mask = np.zeros((370, 1226), dtype=np.uint8)
        ratio = 370 / 154
        for i in range(upscaled_mask.shape[0]):
            for j in range(upscaled_mask.shape[1]):
                upscaled_mask[i, j] = 255 if mask[int(i / ratio), int(j / ratio)] else 0
        return upscaled_mask