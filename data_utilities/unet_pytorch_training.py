import os
import cv2
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from unet_config import (IMAGE_DIR, MASK_DIR, ALLOWED_IMG_EXTS, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, SEED,
                         AUG_PRESET, AUG_STRENGTH)
from torch.utils.data import Dataset, DataLoader

AUG_PRESETS = ("oct_bscan", "en_face", "none")


def resize_and_pad():
    """
    Scale the longest side to fit IMG_HEIGHT/IMG_WIDTH (keeping the aspect ratio so layer thicknesses are not
    distorted) and pad the remainder with 0 (background) for both image and mask.
    """
    return [
        A.LongestMaxSize(max_size_hw=(IMG_HEIGHT, IMG_WIDTH)),
        A.PadIfNeeded(min_height=IMG_HEIGHT, min_width=IMG_WIDTH, position="center",
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    ]


def normalise_and_tensor():
    """
    Scale pixel values from 0-255 to 0-1 and convert image/mask to torch tensors (image -> C,H,W; mask -> H,W).
    """
    return [
        A.Normalize(mean=(0.0,) * IMG_CHANNELS, std=(1.0,) * IMG_CHANNELS, max_pixel_value=255.0),
        ToTensorV2(),
    ]


def build_eval_transform():
    """
    Deterministic transform for validation/prediction: no random changes, only resize/pad, normalise and tensor.
    """
    return A.Compose(resize_and_pad() + normalise_and_tensor())


def build_train_transform(preset=AUG_PRESET, strength=AUG_STRENGTH, seed=SEED):
    """
    Build the training augmentation pipeline for the chosen preset (see AUG_PRESET in unet_config).
    Geometric transforms are applied to the image and mask together (masks use nearest-neighbour, so stay binary),
    intensity transforms are applied to the image only. `strength` scales the size of every random change.
    """
    if preset not in AUG_PRESETS:
        raise ValueError(f"Unknown AUG_PRESET '{preset}', choose from {AUG_PRESETS}")
    if strength < 0:
        raise ValueError(f"AUG_STRENGTH must be >= 0, got {strength}")

    s = strength

    if preset == "none":
        return A.Compose(resize_and_pad() + normalise_and_tensor(), seed=seed)

    # ---------- geometry ---------- #
    if preset == "oct_bscan":
        # B-scans have a fixed orientation (vitreous at the top, choroid at the bottom), so no vertical flips or
        # 90 degree rotations. Horizontal flip = left/right eye. Extra vertical shift as the retina moves up/down.
        # Constant 0 fill as the space above/below the retina is dark, and reflecting would create fake layers.
        geometric = [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                rotate=(-10 * s, 10 * s),
                scale=(1 - 0.1 * s, 1 + 0.1 * s),
                translate_percent={"x": (-0.05 * s, 0.05 * s), "y": (-0.10 * s, 0.10 * s)},
                border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0,
                p=0.5,
            ),
            # light warp for retinal curvature - kept small so thin layers keep their order
            A.ElasticTransform(alpha=15 * s, sigma=8, p=0.15),
        ]
    else:  # en_face
        # no fixed orientation, so every flip and 90 degree rotation is a valid new sample
        geometric = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                rotate=(-20 * s, 20 * s),
                scale=(1 - 0.1 * s, 1 + 0.1 * s),
                translate_percent=(-0.05 * s, 0.05 * s),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.ElasticTransform(alpha=30 * s, sigma=6, p=0.2),
        ]

    # ---------- intensity (image only) ---------- #
    # brightness/contrast/gamma cover differences between devices and exposure; noise/blur/downscale cover
    # scan quality. MultiplicativeNoise approximates OCT speckle.
    intensity = [
        A.RandomBrightnessContrast(brightness_limit=0.15 * s, contrast_limit=0.2 * s, p=0.5),
        A.RandomGamma(gamma_limit=(100 - 30 * s, 100 + 30 * s), p=0.4),
        A.OneOf(
            [
                A.MultiplicativeNoise(multiplier=(1 - 0.15 * s, 1 + 0.15 * s), elementwise=True),
                A.GaussNoise(std_range=(0.01 * s, 0.05 * s)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.Downscale(scale_range=(max(1 - 0.5 * s, 0.25), max(1 - 0.1 * s, 0.25))),
            ],
            p=0.4 if s > 0 else 0.0,
        ),
    ]

    return A.Compose(resize_and_pad() + geometric + intensity + normalise_and_tensor(), seed=seed)


class TrainingSetGenerator(Dataset):
    def __init__(self, augmentation=None, seed=None if SEED is None else SEED):
        super().__init__()
        self.augmentation = augmentation
        self.eval_transform = build_eval_transform()
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
        if mask is None:
            raise ValueError(f"Could not read mask from {mask_path}")
        mask = (mask > 0).astype(np.uint8)

        # image and mask must cover the same pixels for the geometric augmentations to line up
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image {image_path} is {image.shape[:2]} but mask {mask_path} is {mask.shape[:2]}")

        # run the albumentation augmentations if selected, otherwise just resize/pad, normalise and convert to
        # tensors so the output is always the same shape and type
        transform = self.augmentation if self.augmentation is not None else self.eval_transform
        augmentation = transform(image=image, mask=mask)
        image = augmentation["image"]
        mask = augmentation["mask"]

        # With ToTensorV2 used, the mask will not have a channel dimension whilst the RGB/GS image will (e.g. 3,h,w or
        # 1,h,w) therefore the mask requires an addition of a dimension in the 0 index - going from _,h,w to 1,h,w
        mask = mask.unsqueeze(dim=0)

        # masks are changed from int to float32 to match the image tensors following the ToTensorV2
        mask = mask.float()

        return image, mask

    def albumentations_augmentation(self, preset=AUG_PRESET, strength=AUG_STRENGTH):
        """
        Return the training augmentation pipeline for the chosen preset (defaults to AUG_PRESET/AUG_STRENGTH in
        unet_config), e.g. dataset.augmentation = dataset.albumentations_augmentation()
        """
        return build_train_transform(preset=preset, strength=strength, seed=self.seed)


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
