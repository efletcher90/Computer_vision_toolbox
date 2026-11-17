import os
from mp_oct_enums_helpers import OCT_DIR, _normalise_eye

class FundusRegistrator:
    def __init__(self, patient_data_dicts, oct_dir):
        self.oct_dir = oct_dir
        self.patient_data = patient_data_dicts

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

    def match_oct_with_mp(self, patient_data, scan_imgs):
        oct_mp_pairs = []

        for patient in patient_data:
            # only use the second baseline recording for each eye in the folder--- will need to confirm that this is fine later
            if patient['timepoint'].lower() == 'baseline' and patient['repeat'] == 1:
                continue

            eye = _normalise_eye(patient['Eye'])
            timepoint = patient['timepoint'].lower()

            matches = [img for img in scan_imgs if img['patient_id'] == patient['patient_id']
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

    # def run(self, mp_data):



if __name__ == "__main__":

    # QC check for SLO images and fundus list, entry point only accessible here
    oct_parser = FundusRegistrator(OCT_DIR)
    fundus_list = oct_parser.get_slo_list()

    for fundus in fundus_list:
        print(fundus)

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
