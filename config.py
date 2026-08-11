from pathlib import Path
from CNN_model_architectures.unet_pytorch import BCEDiceLoss

# ---------- Model and Training Set Directories ---------- #
PATH_START = Path(__file__).parent

PYTORCH_UNET_FILE = str(PATH_START/Path(r'src/Pytorch_UNet_Model/pytorch_unet_model.pth'))
KERAS_UNET_FILE = str(PATH_START/Path(r'src/Keras_UNet_Model/keras_unet_model.keras'))

IMAGE_DIR = str(PATH_START/Path(r"src/Training_set/Images"))
MASK_DIR = str(PATH_START/Path(r"src/Training_set/Masks"))

# ---------- Model Mode Selection ---------- #
""" Select run mode for full pipeline:
 ---> "train_and_predict" ---> (re)trains the CNN on IMAGE_DIR/MASK_DIR then predicts on user-selected images
 ---> "predict"           ---> loads existing CNN and predicts on user-selected images only
 ---> "sanity_check"      ---> runs dataset_checks on IMAGE_DIR/MASK_DIR
"""
RUN_MODE = "predict"

# ---------- UNet config enums ---------- #
ALLOWED_IMG_EXTS = ['png', 'jpg', 'jpeg', 'tif', 'tiff']

if
LOSS_F = BCEDiceLoss()
L_RATE = 0.0001


