import config

class LoadProcessImagesAndMasks:
    def __init__(self):
        self.file_list = []
        self.mask_list = []

    def filter_accepted_ext(self, file_list):
        """
        :param file_list: list of file paths
        :return:
        """
        return [file for file in file_list if file.lower().endswith(config.ALLOWED_IMG_EXTS)]
