import os
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import math
import cv2

# --- Enums -------- #
# dir for mp and oct data
OUTPUT_DIR = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\Outputs'
MP_DIR = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\MP_for_overlay'
OCT_DIR = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\OCT_masks_for_overlay'
AAVB_FONT = r'C:\Users\EdFletcher\PycharmProjects\Computer_vision_toolbox\PPNeueMachina-PlainRegular.otf'

# wanted keys to patient metadata stored in 'projection.xml' dict
METADATA_KEYS = ['ExamID', 'ExamBaselineID', 'Age', 'Eye', 'DateTime', 'fieldDegreesX', 'fieldDegreesY', 'examPRLX_pix',
                 'examPRLY_pix',
                 'baselinePRLX_pix', 'baselinePRLY_pix', 'baselineShiftX_pix', 'baselineShiftY_pix', 'fixationCenterX',
                 'fixationCenterY'
                 ]
# microperimetry db colour values, as shown in pdf files
DB_COLOUR_VALUES = {'<0': '#060608', '0': '#672B8D', '1': '#842577', '2': '#A31E5C', '3': '#C41E40', '4': '#E11E27',
                    '5': '#EE2025', '6': '#EE2125', '7': '#EE2524', '8': '#EF2924', '9': '#EF3024', '10': '#EF3723',
                    '11': '#F04022', '12': '#F04922', '13': '#F05122', '14': '#F15C23', '15': '#F16623',
                    '16': '#F46F23', '17': '#F47722', '18': '#F68D1F', '19': '#FAA21C', '20': '#FEB714',
                    '21': '#FDCC10', '22': '#F3EC0C', '23': '#E4E518', '24': '#D3DE25', '25': '#BCD531',
                    '26': '#A3CD3A', '27': '#8AC53F', '28': '#73BF43', '29': '#59BA46', '30': '#43B648',
                    '31': '#31B349', '32': '#27B24A', '33': '#1DAE4A', '34': '#16A649', '35': '#119C48',
                    '36': '#0E9245'}

# ------------------ #
# helper functions. These have been separated for Class tidiness

#to QC check the overlay of mp stim points and the png still image provided. Compare to the pdfs in the respective folders
def _test_grid_centres(patient, out_dir="grid_centre_tests"):
    os.makedirs(out_dir, exist_ok=True)
    g = lambda k: float(patient.get(k, 0))
    candidates = {
        "image_centre":   (0, 0),
        "bl_prl":         (g("baselinePRLX_pix"), g("baselinePRLY_pix")),
        "bl_prl_minus_shift":(g("baselinePRLX_pix") - g("baselineShiftX_pix"),
                           g("baselinePRLY_pix") - g("baselineShiftY_pix")),
        "exam_prl":       (g("examPRLX_pix"), g("examPRLY_pix")),
        "bl_prl_xflip":   (-g("baselinePRLX_pix"), g("baselinePRLY_pix")),
        "exam_prl_xflip": (-g("examPRLX_pix"), g("examPRLY_pix")),
        "bl_prl_swap": (-g("baselinePRLY_pix"), g("baselinePRLX_pix")),
        "bl_prl_swap_neg": (g("baselinePRLY_pix"), -g("baselinePRLX_pix")),

    }
    arr = np.array(Image.open(patient["scan_path"]), dtype=np.uint16)
    base = Image.fromarray((arr >> 8).astype(np.uint8)).convert("RGB")
    width, height = base.size
    scale = width / g("fieldDegreesX")

    for name, (dx, dy) in candidates.items():
        for ysign, ylabel in ((-1, "yup"), (1, "ydown")):
            cx, cy = width / 2 + dx, height / 2 + ysign * dy
            cx, cy = cy, width - cx                      # rotate centre to match ROTATE_90
            img = base.transpose(Image.Transpose.ROTATE_90)
            draw = ImageDraw.Draw(img)
            for k, v in patient.items():
                if not k.startswith("stimulus_") or k == "stimulus_00":
                    continue
                fi, ray, ang = v
                db = -1 if fi > 0 else -fi
                a = math.radians(ang)
                x = cx + ray * math.cos(a) * scale
                y = cy - ray * math.sin(a) * scale
                col = DB_COLOUR_VALUES["<0" if db < 0 else str(db)]
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col)

            img.save(os.path.join(out_dir, f"{patient.get('ExamID')}_{name}_{ylabel}.png"))

def _to_uint8(img):
    # converts any 16-bit images to 8-bit. 8-bit images are left out
    return (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else img

def _to_gray(img):
    # unless the image has only two dimensions, it creates the image GS
    return img if img.ndim == 2 else img[..., 1]

def _red_cyan(oct_img, mp_img):
    # The fixed slo scan in red and the moving/transforming MP image in cyan
    mp_img = cv2.resize(mp_img, oct_img.shape[::-1])
    return np.dstack([oct_img, mp_img, mp_img])

def _normalise_eye(eye):
    # Standardises either OS/OD or left/right across the datasets/images
    e = str(eye).strip().lower()
    if e in ('left', 'l', 'os'):
        return 'OS'
    if e in ('right', 'r', 'od'):
        return 'OD'
    return e

def _number_multiple_recordings_in_visit(p_data):
    # mp data can have 2 recordings per eye at a visit (e.g. 2 baselines).
    # Store a sequential number (e.g. 1, 2), in date order, as under 'repeat' key,
    counts = {}
    for patient in sorted(p_data, key=lambda d: d['DateTime']):
        key = (patient['patient_id'], patient['Eye'], patient['timepoint'])
        counts[key] = counts.get(key, 0) + 1
        patient['repeat'] = counts[key]

    return p_data

# def _transform_scale_rotation(transform, centre_point):
#     # measure the transform's scale and rotation around a point by mapping a short line through it
#     p0, p1 = transform.apply(np.array([centre_point, centre_point + [100, 0]]))
#     d = p1 - p0
#
#     return np.linalg.norm(d) / 100, np.degrees(np.arctan2(d[1], d[0]))