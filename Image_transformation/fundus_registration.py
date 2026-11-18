import os
from PIL import Image
from mp_oct_enums_helpers import _normalise_eye, _to_gray, _red_cyan, _to_uint8, AAVB_FONT, _transform_scale_rotation
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe
import numpy as np
from rtnls_registration import Registration, open_image

class FundusRegistrator:
    def __init__(self, p_data_coords_dicts, oct_dir):
        self.oct_dir = oct_dir
        self.p_data_coords_dicts = p_data_coords_dicts

    # Create list of the SLO tifs during OCT imaging
    def get_slo_list(self):
        oct_list = []

        # find all the scan folders within the root directory then os.walk
        folders = sorted([os.path.join(self.oct_dir, name) for name in os.listdir(self.oct_dir) if os.path.isdir(os.path.join(self.oct_dir, name))])

        for folder in folders:
            for dirpath, _, filenames in os.walk(folder):
                # gather lower case filenames in folder
                filenames = {files.lower(): files for files in filenames}

                fundus_path = ez_rpe_path = rpe_bm_path = slo_name = None

                for file in filenames:
                    if file.endswith('slo.tif'):
                        slo_name = filenames[file]
                        fundus_path = os.path.join(dirpath, slo_name)
                    elif file.endswith('ez_rpe_minus_fluid_map.png'):
                        ez_rpe_path = os.path.join(dirpath, filenames[file])
                    elif file.endswith('rpe_bm_map.png'):
                        rpe_bm_path = os.path.join(dirpath, filenames[file])

                # once every file in the folder has been checked, add an entry if it has an SLO image
                if fundus_path:
                    p_id, exam_time, eye = slo_name.split('_')[:3]

                    oct_list.append({'slo_fundus_file': fundus_path,
                                     'ez_rpe_map': ez_rpe_path,
                                     'rpe_bm_map': rpe_bm_path,
                                     'patient_id': p_id,
                                     'exam_time': exam_time,
                                     'eye': eye}
                                    )
        return oct_list

    def match_oct_with_mp(self, oct_scan_imgs):
        oct_mp_pairs = []

        for patient in self.p_data_coords_dicts:
            # only use the second baseline recording for each eye in the folder--- will need to confirm that this is fine later
            if patient['timepoint'].lower() == 'baseline' and patient['repeat'] == 1:
                continue

            eye = _normalise_eye(patient['Eye'])
            timepoint = patient['timepoint'].lower()

            matches = [img for img in oct_scan_imgs if img['patient_id'] == patient['patient_id']
                       and img['exam_time'].lower() == timepoint and _normalise_eye(img['eye']) == eye]

            if not matches:
                print(
                    f"{patient['patient_id']}_{patient['ExamID']}: no OCT SLO found for {eye} eye at {patient['timepoint']}")
                continue

            slo = matches[0]
            oct_mp_pairs.append({'patient_id': patient['patient_id'],
                                 'exam_id': patient['ExamID'],
                                 'eye': patient['Eye'],
                                 'timepoint': patient['timepoint'],
                                 'mp_still': patient['scan_path'],
                                 'stimuli': patient['computed_stim_values'],
                                 'slo_file': slo['slo_fundus_file'],
                                 'ez_rpe_map': slo['ez_rpe_map'],
                                 'rpe_bm_map': slo['rpe_bm_map']})

        return oct_mp_pairs

    def mp_oct_fundus_registration(self, mp_oct_pairs):
        font = FontProperties(fname=AAVB_FONT, size=14)
        font_stroke = [pe.withStroke(linewidth=1, foreground='black')]
        registered_pairs = []

        for pair in mp_oct_pairs:

            mp_img = Image.open(pair['mp_still']).transpose(Image.Transpose.ROTATE_90)
            mp_img = _to_uint8(np.array(mp_img))
            slo_img = _to_uint8(np.array(Image.open(pair['slo_file'])))

            # register MP to OCT SLO
            registrator = Registration(max_rotation_deg=10)
            transform = registrator(mp_img, slo_img)

            # number of matched points the registration was based on (0 means it failed)
            n_matches = len(getattr(registrator, 'final_matches', []))
            transform_info = f'Registration based on {n_matches} matched points' if n_matches else 'Registration FAILED'
            print(f"{pair['exam_id']} {pair['timepoint']}: {transform_info}")

            # warp the MP image into the SLO's pixel frame (same size as the SLO)
            mp_warped = transform.warp(mp_img, slo_img.shape[:2])

            # create a 2 plot figure, showing the before and after registration
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6.4))

            ax1.imshow(_red_cyan(_to_gray(slo_img), _to_gray(mp_img)))
            ax1.set_title(f'Before registration', fontproperties=font)
            ax1.axis('off')

            ax2.imshow(_red_cyan(_to_gray(slo_img), _to_gray(mp_warped)))
            ax2.set_title(f'After registration', fontproperties=font)
            ax2.axis('off')

            # add AAVBio font and add a stroke to the cyan and red label text
            fig.text(0.99, 0.94, 'MP scan (transformed)', color='#00FFFF', fontproperties=font, ha='right', va='bottom', path_effects=font_stroke)
            fig.text(0.99, 0.91, 'OCT scan (fixed)', color='#FF0000', fontproperties=font, ha='right', va='bottom', path_effects=font_stroke)

            # print the figure title as well as the scale and rotation values of the mp scan about the centre point of the image
            fig.suptitle(f"Patient: {pair['patient_id']} - MP_Exam_ID: {pair['exam_id']} - {pair['eye']} Eye - {pair['timepoint']}", font=font, fontsize=14)
            fig.text(0.5, 0.02, transform_info, ha='center', va='bottom', font=font, fontsize=14)
            fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.07, wspace=0.02)
            plt.show()
#
            # Keep the transform data and the prepared images (i.e. transformed 8-bit MP scan and single channel SLO) for the final overlay
            if n_matches:
                registered_pairs.append({**pair,'transform': transform,'mp_img': mp_img,'slo_img': _to_gray(slo_img),'transform_info': transform_info})

        return registered_pairs
