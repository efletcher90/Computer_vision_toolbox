import os
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from unet_config import IMAGE_DIR, MASK_DIR, ALLOWED_IMG_EXTS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, SEED
from torch.utils.data import Dataset, DataLoader

class TrainingSetGenerator(Dataset):
    def __init__(self, augmentation=None, seed=None if SEED is None else SEED):
        super().__init__()
        self.augmentation = augmentation
        self.training_set = []
        self.list_dicts_of_training_data()
        self.seed = seed

    def list_dicts_of_training_data(self):
        """
        Find all training images and masks, match them by filename stem and create a list of dictionaries
        containing their file paths.
        """
        accepted_exts = tuple(ext.lower() for ext in ALLOWED_IMG_EXTS)
        accepted_mask_suffix = tuple("_mask" + ext for ext in accepted_exts)

        # use os.walk to go through subdirectories of MASK_DIR and identify mask files that end with "_mask" and have a
        # lowercase accepted file type, before identifying the stem of the mask (i.e. file name w/o _mask and extension)
        # and getting the path of the image

        masks_by_stem = {}
        for dirpath, _, filenames in os.walk(MASK_DIR):
            for m_name in filenames:
                if m_name.lower().endswith(accepted_mask_suffix):
                    stem = os.path.splitext(m_name)[0].lower()[:-len("_mask")]
                    masks_by_stem[stem] = os.path.join(dirpath, m_name)

        # same with os.walk for the training images but also checks that these do not have the _mask suffix
        images_by_stem = {}
        for dirpath, _, filenames in os.walk(IMAGE_DIR):
            for i_name in filenames:
                lowercase_img = i_name.lower()
                if lowercase_img.endswith(accepted_exts) and not lowercase_img.endswith(accepted_mask_suffix):
                    images_by_stem[os.path.splitext(lowercase_img)[0]] = os.path.join(dirpath, i_name)

        # creates a list of images and masks by their common keys, unpaired images/masks are then identified by
        # removing the common key matches
        common = sorted(images_by_stem.keys() & masks_by_stem.keys())
        unpaired = (images_by_stem.keys() | masks_by_stem.keys()) - set(common)

        # Any masks and images that do not match will raise a ValueError.
        # Prints - the total number of unpaired as well as the first 10 mismatches.
        if unpaired:
            raise ValueError(f"Unpaired training set images/masks ({len(unpaired)} total): {sorted(unpaired)[:10]}")

        # By using the matched pairs, append a dictionary for each training image, corresponding mask +
        # matching stem. This list "training_set" can be used to gather the training set easily w/o mismatch issues.
        for stem in common:
            self.training_set.append(
                {
                "image":images_by_stem[stem],
                "mask":masks_by_stem[stem],
                "stem":stem
                }
            )

        return self.training_set

    def __len__(self):
        """
        Returns the total number of training images in training_set.
        """
        return len(self.training_set)

    def __getitem__(self, idx):
        """
        From the training_set list of dictionaries, load one image and its corresponding mask,
        apply Albumentations (i.e. augmentations) and return them as tensors.
        """
        sample = self.training_set[idx]

        image_path = sample["image"]
        mask_path = sample["mask"]

        # read image file path as colour image, then convert to RGB
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # depending on the input channels set in the config, convert to GS if required
        if IMG_CHANNELS == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # read all label masks as GS and then normalise to a binary '0 or 1' mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)

        # run the albumentation augmentations if selected
        if self.augmentation is not None:
            augmentation = self.augmentation(image=image, mask=mask)
            image = augmentation["image"]
            mask = augmentation["mask"]

        # With ToTensorV2 used, the mask will not have a channel dimension whilst the RGB/GS image will (e.g. 3,h,w or
        # 1,h,w) therefore the mask requires an addition of a dimension in the 0 index - going from _,h,w to 1,h,w
        mask = mask.unsqueeze(dim=0)

        # masks are changed from int to float32 to match the image tensors following the ToTensorV2
        mask = mask.float()

        return image, mask

    def albumentations_augmentation(self):
        train_transform = A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-90, 90)),
                A.RandomBrightnessContrast(
                    p=0.5,
                    brightness_limit=(-0.2,0.2),
                    contrast_limit=(-0.2,0.2),
                )
            ],
            seed=self.seed,
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

 # def k_fold_cross_validation(self):
