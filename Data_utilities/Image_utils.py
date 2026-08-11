import config
from pathlib import Path
import cv2

class LoadProcessTrainingSet:
    def __init__(self):
        self.file_list = []
        self.mask_list = []

    def load_and_filter_files(self, file_list, mask_list):
        self.file_list = [file for file in file_list if file.lower().endswith(config.ALLOWED_IMG_EXTS)]
        self.mask_list = [mask for mask in mask_list if mask.lower().endswith(config.ALLOWED_IMG_EXTS)]


