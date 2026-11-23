import numpy as np
from PIL import Image
from pykrige.ok import OrdinaryKriging
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from mp_oct_enums_helpers import _mp_matplotlib_colourmap

class StructureFunctionAnalysis:
    def __init__(self, registration_data, px_per_deg, stim_radius=11):
        self.registration_data = registration_data
        self.px_per_deg = px_per_deg
        self.stim_radius = stim_radius
        self.mp_cmap = _mp_matplotlib_colourmap()

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
                x, y = stim['transf_px_x'], stim['transf_px_y']
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

    # interpolate a function map around the stimulation sites and align with structure maps. Spatial interpolation
    # investigated using geospatial kriging methods.
    def interpolated_structure_function_measurement(self, shape=(1536, 1536), plot = True):
        # Kriging for every SLO pixel from the transformed stimulus positions. This makes one map per scan
        # SLO image shape, thrown to variable instead of 1536x1536 hardcoded
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w]

        for scan in self.registration_data:
            stimuli = scan['stimuli']
            x_values = np.array([stim['transf_px_x'] for stim in stimuli])
            y_values = np.array([stim['transf_px_y'] for stim in stimuli])

            # -1 is set to 0 dB so the map doesn't bridge across the scotoma
            db_values = np.array([stim['intensity'] if stim['intensity'] >= 0 else 0 for stim in stimuli], dtype=float)

            # fit the variogram in degrees, so its range reads as a retinal distance (1536 px / 30 degrees = 51.2 px/deg)
            ok = OrdinaryKriging(x_values / self.px_per_deg, y_values / self.px_per_deg, db_values, variogram_model='spherical', nlags=6, enable_plotting=False)

            sill, vrange, nugget = ok.variogram_model_parameters
            print(
                f"{scan['exam_id']} {scan['timepoint']}: sill {sill:.1f}, range {vrange:.1f} deg, nugget {nugget:.1f}")

            # predict on a coarse grid (every `step` px), then resize to full SLO resolution
            grid_x = np.arange(0, w, step=8) / self.px_per_deg
            grid_y = np.arange(0, h, step=8) / self.px_per_deg
            z, variance = ok.execute('grid', grid_x, grid_y)

            sens_map = np.array(Image.fromarray(np.asarray(z, dtype=np.float32)).resize((w, h), Image.Resampling.BILINEAR))
            var_map = np.array(Image.fromarray(np.asarray(variance, dtype=np.float32)).resize((w, h), Image.Resampling.BILINEAR))

            # blank everything outside the outermost stimuli (I don't want kriging to extrapolate outside the mp area)
            hull = Delaunay(np.column_stack([x_values, y_values]))
            outside = hull.find_simplex(np.column_stack([xx.ravel(), yy.ravel()])).reshape(h, w) < 0
            sens_map[outside] = np.nan
            var_map[outside] = np.nan

            # store on the scan for plotting and pixel pairing later. np.clip() the sensitivity info to MAIA's 0-36 dB stim range
            scan['sens_map'] = np.clip(sens_map, 0, 36)
            scan['sens_var_map'] = var_map
            scan['kriging_model'] = ok

        # visual for the readout fo reach map, turn off via the kwarg in function

        # one bin per whole dB (0-36), so each value gets its own colour from the list, in order
        mp_norm = BoundaryNorm(np.arange(-0.5, 37.5, 1), self.mp_cmap.N)

        if plot:
            scan = self.registration_data[0]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(scan['slo_img'], cmap='gray')
            im1 = ax1.imshow(scan['sens_map'], cmap=self.mp_cmap, norm=mp_norm, alpha=0.7)
            ax1.set_title('Kriged sensitivity (dB)')
            cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, ticks=range(0, 37, 6), label='Sensitivity (dB)')
            cbar.minorticks_off()
            im2 = ax2.imshow(scan['sens_var_map'], cmap='magma_r')
            ax2.set_title('Kriging variance')
            fig.colorbar(im2, ax=ax2, fraction=0.046)
            for ax in (ax1, ax2):
                ax.axis('off')
            plt.show()

        return self.registration_data


    # linear regression models are too crude for the dataset - use Bayesian LMM
    # @staticmethod
    # def linear_regression(sf_data, x_key='mean_rpe_bm_um'):
    #     x_values = []
    #     y_values = []
    #
    #     for v in sf_data:
    #         x = v[x_key]
    #         y = v['sensitivity_db']
    #
    #         # skip stimuli outside the scan (i.e. NaN thickness) and not-seen points
    #         if np.isnan(x) or y < 0:
    #             continue
    #
    #         x_values.append(x)
    #         y_values.append(y)
    #
    #     # run linear regression
    #     slope, intercept, r, p, std_err = stats.linregress(x_values, y_values)
    #
    #     # fitted line from the smallest to the largest x
    #     x_line = np.array([min(x_values), max(x_values)])
    #     y_line = slope * x_line + intercept
    #
    #     plt.scatter(x_values, y_values, s=15)
    #     plt.plot(x_line, y_line, color='red')
    #     plt.xlabel(f'{x_key} (µm)')
    #     plt.ylabel(f'{'Retinal sensitivity (db)'}')
    #     plt.title(f'slope {slope:.3f} dB/µm, r={r:.2f}, p={p:.3g}, n={len(x_values)}')
    #     plt.show()
    #
    #     return slope, intercept, r, p, std_err

# Work on a mixed-model stats plot and the interpolated map and run with plot, then look at the perilesion area


    # def interpolated_structure_function_dicts(self):

    # def perilesion_sf_analysis(self):