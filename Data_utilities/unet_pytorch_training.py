import os
import random
import albumentations as A
import cv2
import numpy as np
from config import IMAGE_DIR, MASK_DIR, ALLOWED_IMG_EXTS, IMG_HEIGHT, IMG_WIDTH
from torch.utils.data import Dataset


class LoadProcessTrainingSet:
    def __init__(self):
        super().__init__()
        self.training_set = []

    def list_dicts_of_training_data(self):
        """
        Create a dictionary training_set of image/mask dictionaries.
        """
        accepted_exts = tuple(ext.lower() for ext in ALLOWED_IMG_EXTS)
        accepted_mask_suffix = tuple("_mask" + ext for ext in accepted_exts)

        masks_by_stem = {}
        for dirpath, _, filenames in os.walk(MASK_DIR):
            for m_name in filenames:
                if m_name.lower().endswith(accepted_mask_suffix):
                    stem = os.path.splitext(m_name)[0].lower()[:-len("_mask")]
                    masks_by_stem[stem] = os.path.join(dirpath, m_name)

        images_by_stem = {}
        for dirpath, _, filenames in os.walk(IMAGE_DIR):
            for i_name in filenames:
                lowercase_img = i_name.lower()
                if lowercase_img.endswith(accepted_exts) and not lowercase_img.endswith(accepted_mask_suffix):
                    images_by_stem[os.path.splitext(lowercase_img)[0]] = os.path.join(dirpath, i_name)

        common = sorted(images_by_stem.keys() & masks_by_stem.keys())
        unpaired = (images_by_stem.keys() | masks_by_stem.keys()) - set(common)

        # Any masks and images that do not have matching stems will raise a ValueError.
        # The total number of unpaired as well as printing the first 10 mismatches.
        if unpaired:
            raise ValueError(f"Unpaired training set images/masks ({len(unpaired)} total): {sorted(unpaired)[:10]}")

        # By using the matched pairs, create a list of dictionaries for each training image, corresponding mask +
        # matching stem. This list "training_set" can be used to gather the training set easily w/o mismatch issues.
        for c in common:
            self.training_set.append({
                "image":images_by_stem[c],
                "mask":masks_by_stem[c],
                "stem":c}
            )

        return self.training_set

class TrainingSetGenerator(Dataset):
    def __init__(self):
        super().__init__()
        self.training_set = []


    def albumentations(self):
        train_transform = A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),

                A.Rotate(angle_range=(-90, 90), p=0.5),

                A.RandomBrightnessContrast(
                    brightness_range=(-0.2,0.2),
                    contrast_range=(-0.2,0.2),
                    p=0.5
                ),
            ],
            seed=42,
        )

    # def augment_image_intensity(self, train_image):

    #     img = train_image.astype(np.float32)
    #
    #     if random.random() < 0.5:
    #         alpha = random.uniform(0.9, 1.1)
    #         img *= alpha
    #
    #     if random.random() < 0.4:
    #         beta = random.uniform(-10, 10)
    #         img += beta
    #
    #     if random.random() < 0.7:
    #         gamma = random.uniform(0.8, 1.4)
    #         img = 255.0 * ((np.clip(img, 0, 255) / 255.0) ** gamma)
    #
    #     if random.random() < 0.3:
    #         if img.ndim == 3 and img.shape[-1] == 1:
    #             img2d = img[:, :, 0]
    #             img2d = cv2.GaussianBlur(img2d, (3, 3), 0)
    #             img = img2d[:, :, np.newaxis]
    #         else:
    #             img = cv2.GaussianBlur(img, (3, 3), 0)
    #
    #     if random.random() < 0.5:
    #         noise = np.random.normal(0, 6, img.shape)
    #         img += noise
    #
    #     img = np.clip(img, 0, 255).astype(np.uint8)
    #     if img.ndim == 2:
    #         img = img[:, :, np.newaxis]
    #
    #     return img
