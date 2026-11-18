from mp_oct_enums_helpers import _rgb_to_thickness
import numpy as np
from PIL import Image

class OctThicknessMapping:
    def __init__(self, registered_pairs, ez_lut_csv, rpe_lut_csv):
        self.registered_pairs = registered_pairs
        self.ez_lut_csv = ez_lut_csv
        self.rpe_lut_csv = rpe_lut_csv

    def get_pixel_thickness_values(self):
        rgb_values_ez = []
        rgb_values_rpe = []

        for t_map in self.registered_pairs:
            ez_img = t_map['ez_rpe_map']
            rpe_img = t_map['rpe_bm_map']

            # each entry is (rgb, thickness), or None if the map is missing
            rgb_values_ez.append(_rgb_to_thickness(ez_img, self.ez_lut_csv) if ez_img else None)
            rgb_values_rpe.append(_rgb_to_thickness(rpe_img, self.rpe_lut_csv) if rpe_img else None)

        return rgb_values_ez, rgb_values_rpe

    def overlay_rgb_maps(self, rgb_values_ez, rgb_values_rpe, min_um=0, max_um=30):
        for pair, ez_values, rpe_values in zip(self.registered_pairs, rgb_values_ez, rgb_values_rpe):
            slo = Image.fromarray(pair['slo_img']).convert("RGB")

            for key, values in (('ez_rpe_map', ez_values), ('rpe_bm_map', rpe_values)):
                if values is None:
                    pair[f'{key}_thin_overlay'] = None
                    continue
                rgb, thickness = values
                map_img = Image.fromarray(rgb)

                # mask of pixels within the thickness range. NaN (outside scan) compares as False, so it's excluded
                in_range = (thickness >= min_um) & (thickness <= max_um)
                mask = Image.fromarray(in_range.astype(np.uint8) * 255)

                # match the map and mask to the SLO size (nearest keeps colours and the mask exact)
                if map_img.size != slo.size:
                    map_img = map_img.resize(slo.size, Image.Resampling.NEAREST)
                    mask = mask.resize(slo.size, Image.Resampling.NEAREST)

                # map colour where the mask is white, SLO everywhere else
                pair[f'{key}_thickness_overlay'] = Image.composite(map_img, slo, mask)

