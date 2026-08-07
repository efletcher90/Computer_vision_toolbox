import pathlib

from CNN_model_architectures.unet_pytorch import BCEDiceLoss

PATH_START = pathlib.Path(__file__).parent



ALLOWED_IMG_EXTS = set(['png', 'jpg', 'jpeg', 'tif', 'tiff'])

# ---------- PyTorch UNet config enums ---------- #

LOSS_F = BCEDiceLoss()
L_RATE = 0.0001

