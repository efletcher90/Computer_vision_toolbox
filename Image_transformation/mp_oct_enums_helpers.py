import os
import numpy as np
import csv
import PIL.Image as Image
from PIL import ImageDraw
import math
from matplotlib.colors import BoundaryNorm
from pathlib import Path
import cv2
from scipy.ndimage import binary_opening, distance_transform_edt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# --- Enums -------- #
# dir for mp and oct data
PATH_START = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data'
# OUTPUT_DIR = str(PATH_START/Path(r'Outputs'))
OUTPUT_DIR = r"C:\Users\EdFletcher\Downloads\Outputs"

MP_DIR = str(PATH_START/Path(r'MP_for_overlay'))
OCT_DIR = str(PATH_START/Path(r'OCT_masks_for_overlay'))
EZ_LUT = str(PATH_START/Path(r'EZ_RPE_scalebar_LUT.csv'))
RPE_LUT = str(PATH_START/Path(r'RPE_BM_scalebar_LUT.csv'))

AAVB_FONT = r"C:\Users\EdFletcher\PycharmProjects\Computer_vision_toolbox\Image_transformation\Supporting_data\PPNeueMachina-PlainRegular.otf"

# wanted keys to patient metadata stored in 'projection.xml' dict
METADATA_KEYS = ['ExamID', 'ExamBaselineID', 'Age', 'Eye', 'DateTime', 'fieldDegreesX', 'fieldDegreesY', 'examPRLX_pix',
                 'examPRLY_pix', 'baselinePRLX_pix', 'baselinePRLY_pix', 'baselineShiftX_pix', 'baselineShiftY_pix',
                  'fixationCenterX', 'fixationCenterY']

# microperimetry db colour values, as shown in PDF files
DB_COLOUR_VALUES = {'<0': '#060608', '0': '#672B8D', '1': '#842577', '2': '#A31E5C', '3': '#C41E40', '4': '#E11E27',
                    '5': '#EE2025', '6': '#EE2125', '7': '#EE2524', '8': '#EF2924', '9': '#EF3024', '10': '#EF3723',
                    '11': '#F04022', '12': '#F04922', '13': '#F05122', '14': '#F15C23', '15': '#F16623',
                    '16': '#F46F23', '17': '#F47722', '18': '#F68D1F', '19': '#FAA21C', '20': '#FEB714',
                    '21': '#FDCC10', '22': '#F3EC0C', '23': '#E4E518', '24': '#D3DE25', '25': '#BCD531',
                    '26': '#A3CD3A', '27': '#8AC53F', '28': '#73BF43', '29': '#59BA46', '30': '#43B648',
                    '31': '#31B349', '32': '#27B24A', '33': '#1DAE4A', '34': '#16A649', '35': '#119C48',
                    '36': '#0E9245'}

# for the OCT still image, get the number of pixels per degree to 1 dp
SLO_PX_PER_DEG = round(float(1536 / 30), 1)

# ------------------ #
# helper functions. These have been separated for Class tidiness
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
        return 'Left'
    if e in ('right', 'r', 'od'):
        return 'Right'
    return e

def _number_multiple_recordings_at_timepoint(p_data):
    # mp data can have 2 recordings per eye at a timepoint (e.g. 2 baselines).
    # Store a sequential number (e.g. 1, 2), in date order, as under 'repeat' key,
    counts = {}
    for patient in sorted(p_data, key=lambda d: d['DateTime']):
        key = (patient['patient_id'], patient['Eye'], patient['timepoint'])
        counts[key] = counts.get(key, 0) + 1
        patient['repeat'] = counts[key]

    return p_data

def _stim_text_hex(d):
    # white text on the dark ellipses is fine, but not so good on the light ones -- this switches the number colour
    stim_hex_value = "#000000" if d > 2 else "#FFFFFF"
    return stim_hex_value

def _rgb_to_thickness(map_img, lut_csv, max_rgb_thickness=30):
   # use csv to open the lut_csv corresponding to the thickness:colour mapping from Cleveland
    with open(lut_csv, newline='', encoding='utf-8') as lut_csv:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(lut_csv)]

        # create a dict of key=RGB pixel data as a tuple and value=thickness in microns
        um_rgb_dict = {(int(r['R']), int(r['G']), int(r['B'])): float(r['mean_um']) for r in rows}

        # arrays of the table colours and thicknesses, in the same order, for the nearest colour search
        lut_rgb = np.array(list(um_rgb_dict.keys()), dtype=float)
        lut_um = np.array(list(um_rgb_dict.values()))

        rgb_img = np.array(Image.open(map_img).convert('RGB'))

        # reshape image as a series of 1D pixels. Then align to the closest RGB value in the LUT for that pixel
        # then get the thickness for that pixel
        colours, inverse = np.unique(rgb_img.reshape(-1, 3), axis=0, return_inverse=True)
        distances = np.linalg.norm(colours[:, None, :].astype(float) - lut_rgb[None, :, :], axis=2)
        values = lut_um[distances.argmin(axis=1)]

        # Register any pixel that has a corresponding thickness greater than the max_rgb_thickness variable to NaN
        # This way these will not be overlaid onto the final SLO image. We're not interested in any thicknesses that are
        # greater than ~30um... this is the current parameter for the helper fn
        values[distances.min(axis=1) > max_rgb_thickness] = np.nan

        # return the map png as well as the thickness map, both as ndarrays --- stored together as a tuple
        return rgb_img, values[inverse.ravel()].reshape(rgb_img.shape[:2])

def _mp_matplotlib_colourmap():
    # slice the dictionary values from the second entry onwards, the black for <0 dB is ignored as required
    colours = list(DB_COLOUR_VALUES.values())[1:]
    mp_cmap = ListedColormap(colours, name='mp_cmap')

    return mp_cmap

def _clean_mask(thickness, max_um):
    # pixels at or below max_um are removed. i.e. NaN (outside the scan). These can be flecks, and atm, aren't being looked at
    return binary_opening(np.nan_to_num(thickness, nan=999) <= max_um, iterations=3)

# ------- debugging functions ------------------
#to QC check the overlay of mp stim points and the png still image provided. Compare to the PDFs in the respective folders
def _test_grid_centres(patient, out_dir="grid_centre_tests"):
    os.makedirs(out_dir, exist_ok=True)
    grid = lambda k: float(patient.get(k, 0))
    candidates = {
        "image_centre":   (0, 0),
        "bl_prl":         (grid("baselinePRLX_pix"), grid("baselinePRLY_pix")),
        "bl_prl_minus_shift":(grid("baselinePRLX_pix") - grid("baselineShiftX_pix"),
                           grid("baselinePRLY_pix") - grid("baselineShiftY_pix")),
        "exam_prl":       (grid("examPRLX_pix"), grid("examPRLY_pix")),
        "bl_prl_xflip":   (-grid("baselinePRLX_pix"), grid("baselinePRLY_pix")),
        "exam_prl_xflip": (-grid("examPRLX_pix"), grid("examPRLY_pix")),
        "bl_prl_swap": (-grid("baselinePRLY_pix"), grid("baselinePRLX_pix")),
        "bl_prl_swap_neg": (grid("baselinePRLY_pix"), -grid("baselinePRLX_pix")),

    }
    arr = np.array(Image.open(patient["scan_path"]), dtype=np.uint16)
    base = Image.fromarray((arr >> 8).astype(np.uint8)).convert("RGB")
    width, height = base.size
    scale = width / grid("fieldDegreesX")

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

# debugging output for alignment of the ez and rpe structure hulls on the gs slo still at visit timepoints
def _ez_and_rpe_maps_on_slo_img(registration_data):
    font = FontProperties(fname=AAVB_FONT, size=10)
    for scan in registration_data:
        plt.figure(figsize=(6, 6))
        plt.imshow(scan['slo_img'], cmap='gray')
        plt.contour(np.nan_to_num(scan['ez_px_thicknesses'][1], nan=99), levels=[0.5, 1.5], colors='red', alpha=0.3)
        plt.contour(np.nan_to_num(scan['rpe_px_thicknesses'][1], nan=99), levels=[0.5, 1.5], colors='blue', alpha=0.3)
        plt.text(0.03, 0.07, 'EZ loss (EZ–RPE @ 0 µm)', color='red', transform=plt.gca().transAxes)
        plt.text(0.03, 0.03, 'RPE loss (RPE–BM @ 0 µm)', color='blue', transform=plt.gca().transAxes)
        plt.grid(alpha=0.3, color='white')
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        plt.title(f"{scan['patient_id']}: EZ_and_RPE_boundaries_{scan['exam_id']}_{scan['timepoint']}", font=font)
        plt.show()

# align timepoints with ez disruption and loss (i.e. 0-20um and 0um, as determined by Cleveland)
# ontop of stim_id locations (yellow circles highlight problematic stims.
# this is to pinpoint any stim sites that have high sensitivity and yet low ez structure
def _plot_ez_integrity_and_sensitivity_per_timepoint(reg_data, suspect_ids, edgecolours=None):
    font = FontProperties(fname=AAVB_FONT, size=10)
    for scan in reg_data:
        ez = scan['ez_px_thicknesses'][1]
        stimuli = scan['stimuli']
        xs = [stim['transf_px_x'] for stim in stimuli]
        ys = [stim['transf_px_y'] for stim in stimuli]
        sens_range = [max(stim['intensity'], 0) for stim in stimuli]

        plt.figure(figsize=(6, 6))
        plt.imshow(scan['slo_img'], cmap='gray')

        mp_cmap = _mp_matplotlib_colourmap()
        mp_norm = BoundaryNorm(np.arange(-0.5, 37.5, 1), mp_cmap.N)

        # EZ loss (0 um)
        plt.contour(_clean_mask(ez, 0.5), levels=[0.5], colors='red', linewidths=1)
        # disrupted EZ border (0-20 um)
        plt.contour(_clean_mask(ez, 20), levels=[0.5], colors='orange', linewidths=1)
        plt.grid(alpha=0.3, color='white', linestyle='-')
        sc = plt.scatter(xs, ys, c=sens_range, cmap=mp_cmap, norm=mp_norm, s=30, linewidths=1, edgecolors='white')

        # circle the suspect stimuli
        for stim in stimuli:
            if stim['id'] in suspect_ids:
                plt.scatter(stim['transf_px_x'], stim['transf_px_y'], s=250, facecolors='none', edgecolors=edgecolours, linewidths=2)

        cbar = plt.colorbar(sc, ticks=range(0, 37, 6), label='Sensitivity (dB)', fraction=0.046, pad=0.04)
        plt.text(0.03, 0.07, 'EZ loss (0 µm)', color='red', transform=plt.gca().transAxes)
        plt.text(0.03, 0.03, 'EZ disrupted (≤ 20 µm)', color='orange', transform=plt.gca().transAxes)
        cbar.minorticks_off()
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        plt.tight_layout()
        plt.title(f"{scan['patient_id']}: EZ_integrity_{scan['exam_id']}_{scan['timepoint']}", font=font)
        plt.show()

# Adds each stimulus' distance from the EZ-loss border(negative inside the lesion)
def _ez_loss_border_distance_to_stim(registration_data, ez_loss_um=0.5):
    for scan in registration_data:
        lesion = _clean_mask(scan['ez_px_thicknesses'][1], ez_loss_um)
        signed = np.where(lesion, -distance_transform_edt(lesion), distance_transform_edt(~lesion))
        for s in scan['stimuli']:
            s['border_dist_px'] = float(signed[int(round(s['transf_px_y'])), int(round(s['transf_px_x']))])

def _offset_test(registration_data, max_shift=100, step=5):
    # correlation of EZ thickness with sensitivity for the stimulus grid shifted by (dx, dy) px; best should be ~(0, 0)
    for scan in registration_data:
        ez = scan['ez_px_thicknesses'][1]
        scores = {}
        for dx in range(-max_shift, max_shift + 1, step):
            for dy in range(-max_shift, max_shift + 1, step):
                pairs = np.array([(s['intensity'], ez[int(round(s['transf_px_y'] + dy)), int(round(s['transf_px_x'] + dx))])
                                  for s in scan['stimuli']])
                pairs = pairs[~np.isnan(pairs[:, 1])]
                scores[(dx, dy)] = np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]
        best = max(scores, key=scores.get)
        print(f"{scan['timepoint']}: best shift {best} r={scores[best]:.3f}, no shift r={scores[(0, 0)]:.3f}")