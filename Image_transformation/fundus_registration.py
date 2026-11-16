import numpy as np
import cv2
from matplotlib import pyplot as plt
from rtnls_registration import Registration, open_image
import os
from Image_transformation.mp_oct_enums_helpers import OCT_DIR

class FundusRegistrator:
    def __init__(self, oct_dir):
        self.oct_dir = oct_dir

    # Create list of the SLO tifs during OCT imaging
    def get_slo_list(self):
        oct_list = []

        # find all the scan folders within the root directory then os.walk
        folders = sorted([os.path.join(self.oct_dir, name) for name in os.listdir(self.oct_dir) if os.path.isdir(os.path.join(self.oct_dir, name))])

        for folder in folders:
            for dirpath, _, filenames in os.walk(folder):
                # gather lower case filenames in folder
                filenames = {files.lower(): files for files in filenames}

                for file in filenames:
                    if file.endswith('slo.tif'):
                        fundus_path = os.path.join(dirpath, file)

                        p_id, exam_time, eye = file.split('_')[:3]

                        oct_list.append({'fundus_file': fundus_path, 'patient_id': p_id, 'exam_time': exam_time,'eye': eye})

        return oct_list

if __name__ == "__main__":
    registrator = FundusRegistrator(OCT_DIR)
    fundus_list = registrator.get_slo_list()

    for fundus in fundus_list:
        print(fundus)










# def to_uint8(img):
#     """16-bit -> 8-bit; leaves 8-bit images alone."""
#     return (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else img
#
#
# def to_gray(img):
#     """Single channel; uses green for colour images, as the registration preprocessing does."""
#     return img if img.ndim == 2 else img[..., 1]
#
#
# def red_cyan(fixed, moving):
#     """Fixed image in red, moving image in cyan. Where vessels align they look grey/white."""
#     moving = cv2.resize(moving, fixed.shape[::-1])
#     return np.dstack([fixed, moving, moving])
#
#
# # --- load ---
# mp_img = to_uint8(open_image(r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\sandbox\3678_MP.png'))
# slo_img = to_uint8(open_image(r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\sandbox\3678_SLO.tif'))
#
# # MAIA stills are stored rotated 90° from the report view. np.rot90(k=1) = PIL ROTATE_90 (anticlockwise).
# # Remove this line if 3678_MP.png has already been rotated.
# # mp_img = np.ascontiguousarray(np.rot90(mp_img, k=1))
#
# # --- before registration ---
# plt.imshow(red_cyan(to_gray(slo_img), to_gray(mp_img)))
# plt.title('Before registration')
# plt.axis('off')
# plt.show()
#
# # --- register MP -> SLO ---
# registrator = Registration()
# transform = registrator(mp_img, slo_img)
#
# # warp the MP image into the SLO's pixel frame (same size as the SLO)
# mp_warped = transform.warp(mp_img, slo_img.shape[:2])
#
# # --- after registration ---
# plt.imshow(red_cyan(to_gray(slo_img), to_gray(mp_warped)))
# plt.title('After registration')
# plt.axis('off')
# plt.show()
#
# # --- map MP stimulus coordinates onto the SLO ([[x0, y0], [x1, y1], ...] in MP image pixels) ---
# coords_mp = np.array([[500, 500], [900, 500]])
# coords_slo = transform.apply(coords_mp)
#
# plt.imshow(to_gray(slo_img), cmap='gray')
# plt.scatter(coords_slo[:, 0], coords_slo[:, 1], c='c', s=20)
# plt.title('MP points on SLO')
# plt.axis('off')
# plt.show()
