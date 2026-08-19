from pathlib import Path
from CNN_model_architectures.unet_pytorch import BCEDiceLoss

# ---------- Root Directory ---------- #
PATH_START = Path().resolve()   # Address of project folder

# ---------- Model Directories ---------- #
PYTORCH_UNET_FILE = str(PATH_START/Path(r'src/UNet_Model/pytorch_unet_model.pth'))
KERAS_UNET_FILE = str(PATH_START/Path(r'src/UNet_Model/keras_unet_model.keras'))

# ---------- Training Set Directories ---------- #
IMAGE_DIR = str(PATH_START/Path(r"src/Training_set/Images"))
MASK_DIR = str(PATH_START/Path(r"src/Training_set/Masks"))
TEST
# ---------- Training Set Parameters ---------- #
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 512, 512, 1

LOSS_F = BCEDiceLoss()
L_RATE = 0.0001

# ---------- Model Mode Selection ---------- #
""" Select run mode for full pipeline:
train_and_predict ---> (re)trains the CNN on IMAGE_DIR/MASK_DIR then predicts on user-selected images
predict           ---> loads existing CNN and predicts on user-selected images only
sanity_check      ---> runs dataset_checks on IMAGE_DIR/MASK_DIR
"""
RUN_MODE = "predict"

# ---------- Accepted file extensions ---------- #
ALLOWED_IMG_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']



