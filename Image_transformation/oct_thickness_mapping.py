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

            # each entry is a tuple of the map and thickness data (ndarrays; or None if the map is missing)
            rgb_values_ez.append(_rgb_to_thickness(ez_img, self.ez_lut_csv) if ez_img else None)
            rgb_values_rpe.append(_rgb_to_thickness(rpe_img, self.rpe_lut_csv) if rpe_img else None)

        return rgb_values_ez, rgb_values_rpe

    def overlay_rgb_maps(self, rgb_values_ez, rgb_values_rpe, min_um=0, max_um=30):
        for pair, ez_values, rpe_values in zip(self.registered_pairs, rgb_values_ez, rgb_values_rpe):
            slo = Image.fromarray(pair['slo_img']).convert("RGB")

            for key, values in (('ez_rpe_map', ez_values), ('rpe_bm_map', rpe_values)):
                rgb, thickness = values
                map_img = Image.fromarray(rgb)

                # Keeps mask of pixels within the thickness range, any others == False and therefore is excluded
                # from the overlay. We're only interested in thicknesses of these tissues at lesion or perilesion
                in_range = (thickness >= min_um) & (thickness <= max_um)
                mask = Image.fromarray(in_range.astype(np.uint8) * 255)

                # maps and SLO are expected to be the same size (1536x1536px) -- give debug error if not
                if map_img.size != slo.size:
                    raise ValueError(f"{pair['exam_id']} {key}: map is {map_img.size}, SLO is {slo.size}")

                # map colour where the mask is white, SLO everywhere else
                pair[f'{key}_thickness_overlay'] = Image.composite(map_img, slo, mask)

