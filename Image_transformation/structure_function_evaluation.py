import numpy as np
import matplotlib.pyplot as plt

class StructureFunctionAnalysis:
    def __init__(self, registration_data, stim_radius=11):
        self.registration_data = registration_data
        self.stim_radius = stim_radius

        # The MP stillImage.png has 1024 px over the "fieldDegreesX = 36.5°"  so:  1024 / 36.5 = 28.05 px/°
        # For the OCT SLO, it has 1536 px over a 30° field (presented in protocol), so:  1536 / 30 = 51.2 px/°
        # Transformation scale from registration, MP -> SLO px/°: 51.2 / 28.05 = 1.825
        # MP stimulus is Goldmann III, 0.43° diameter == 0.215° radius. On the MP still:  0.43° x 28.05 px/° = 12.1 px diameter or radius = 6 px
        # Therefore, on the SLO with correction: = 12.1 px diameter * 1.825 = 22.08 px or 11.04 px for stim radius

    def direct_structure_function_measurement(self):
        sf_dict = []

        for scan in self.registration_data:
            ez_um = scan['ez_px_thicknesses'][1]
            rpe_um = scan['rpe_px_thicknesses'][1]

            for stim in scan['stimuli']:
                x, y = stim['trans_px_x'], stim['trans_px_y']
                x_int, y_int = int(round(x)), int(round(y))

                # square window around the stimulus, and the pixels within its disk. Indexing is [row, column] = [y, x] --- got some AI help here!
                window = np.s_[y_int - self.stim_radius:y_int + self.stim_radius + 1, x_int - self.stim_radius:x_int + self.stim_radius + 1]
                yy, xx = np.ogrid[window]
                disk = (xx - x) ** 2 + (yy - y) ** 2 <= self.stim_radius ** 2

                # thickness pixels under the stimulus, ignoring NaN (outside the OCT scan)
                ez_pixels = ez_um[window][disk]
                rpe_pixels = rpe_um[window][disk]
                ez_pixels = ez_pixels[~np.isnan(ez_pixels)]
                rpe_pixels = rpe_pixels[~np.isnan(rpe_pixels)]

                sf_dict.append({'patient_id': scan['patient_id'],
                                'exam_id': scan['exam_id'],
                                'eye': scan['eye'],
                                'timepoint': scan['timepoint'],
                                'stimulus_id': stim['id'],
                                'sensitivity_db': stim['intensity'],
                                'transf_px_x': round(float(x), 1),
                                'transf_px_y': round(float(y), 1),
                                'mean_ez_rpe_um': ez_pixels.mean() if ez_pixels.size else np.nan,
                                'mean_rpe_bm_um': rpe_pixels.mean() if rpe_pixels.size else np.nan}
                               )

        return sf_dict

    def linear_regression(self, sf_data):




    # def interpolated_structure_function_dicts(self):

    # def perilesion_sf_analysis(self):