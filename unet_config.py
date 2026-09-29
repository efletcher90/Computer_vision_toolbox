from pathlib import Path
from UNet_architectures.unet_pytorch import BCEDiceLoss

# ---------- Root Directory ---------- #
PATH_START = Path().resolve()   # Address of project folder

# ---------- Model Directories ---------- #
PYTORCH_UNET_FILE = str(PATH_START/Path(r'src/UNet_Model/pytorch_unet_model.pth'))
KERAS_UNET_FILE = str(PATH_START/Path(r'src/UNet_Model/keras_unet_model.keras'))

# ---------- Training Set Directories ---------- #
IMAGE_DIR = str(PATH_START/Path(r"src/training_set/Images"))
MASK_DIR = str(PATH_START/Path(r"src/training_set/Masks"))

# ---------- Test Image Directory ---------- #
TEST_IMG_DIR= str(PATH_START/Path(r"src/test_images/"))

# ---------- Training Set Parameters ---------- #
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 512, 512, 1

# ---------- Augmentation Settings ---------- #
""" Select the albumentations preset used for training:
oct_bscan ---> OCT B-scans (cross-sections). Keeps the retina upright: horizontal flips, small rotations
               (+/-10 deg), vertical shifts, speckle noise and device-style contrast/resolution changes
en_face   ---> images with no fixed orientation (en face OCT/OCT-A, fundus, flatmounts, microscopy):
               adds vertical flips and 90 degree rotations
none      ---> no random augmentation, only resize/pad, normalise and convert to tensor
"""
AUG_PRESET = "oct_bscan"

# Scales the size of every random change (rotation angle, shift, noise, contrast etc.) in the preset.
# 1.0 = default ranges, 0.5 = half as strong, 0.0 = only flips/rotate90 remain (keep roughly within 0-2)
AUG_STRENGTH = 1.0

LOSS_F = BCEDiceLoss()
L_RATE = 0.0001

# ---------- Model Mode Selection ---------- #
""" Select run mode for model running:
train_and_predict ---> (re)trains the CNN on IMAGE_DIR/MASK_DIR then predicts on user-selected images
predict           ---> loads existing CNN and predicts on user-selected images only
sanity_check      ---> runs dataset_checks on IMAGE_DIR/MASK_DIR
"""
RUN_MODE = "predict"

# ---------- Accepted file extensions ---------- #
ALLOWED_IMG_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

SEED = 42

