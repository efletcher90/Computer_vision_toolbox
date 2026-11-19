from Image_transformation.fundus_mapping import CreateMicroperimetryDataDicts, CalculateMicroperimetryCoordinates, FundusOverlays
from Image_transformation.oct_thickness_mapping import OctThicknessMapping
from Image_transformation.fundus_registration import FundusRegistrator
from Image_transformation.mp_oct_enums_helpers import (DB_COLOUR_VALUES, MP_DIR, OUTPUT_DIR, OCT_DIR, METADATA_KEYS,
                                                       _test_grid_centres, _number_multiple_recordings_in_visit,
                                                       AAVB_FONT, EZ_LUT, RPE_LUT)
import os
import time

def main(test_grid_centres_flag=False, timestamp=time.strftime("%Y-%m-%d_%H%M")):
    timestamp=time.strftime("%Y-%m-%d_%H%M")
    xml_parser = CreateMicroperimetryDataDicts(MP_DIR, METADATA_KEYS)

    mp_files, patient_id = xml_parser.get_mp_list()
    patient_data_dicts = xml_parser.parse_mp_xml_data(mp_files, patient_id)
    patient_data_dicts = _number_multiple_recordings_in_visit(patient_data_dicts)

    calculator = CalculateMicroperimetryCoordinates(patient_data_dicts)
    data_with_coords = calculator.calculate_pixel_coordinates()

    # This call is used to check for the centre grid points on
    if test_grid_centres_flag:
        for patient in patient_data_dicts:
            _test_grid_centres(patient)

    oct_parser = FundusRegistrator(data_with_coords, OCT_DIR, OUTPUT_DIR, AAVB_FONT, timestamp)
    oct_file_list = oct_parser.get_slo_list()

    matched_mp_oct = oct_parser.match_oct_with_mp(oct_file_list)

    # identify matches for debugging
    for match in matched_mp_oct:
        print(match['patient_id'], match['exam_id'], match['eye'], match['timepoint'], '->', os.path.basename(match['slo_file']))

    overlay = FundusOverlays(data_with_coords, DB_COLOUR_VALUES, MP_DIR, OUTPUT_DIR, timestamp, AAVB_FONT)

    overlay.draw_mp_data_to_maia_scan()
    registration_data = oct_parser.mp_oct_fundus_registration(matched_mp_oct)

    mapping = OctThicknessMapping(registration_data, EZ_LUT, RPE_LUT)
    rgb_values_ez, rgb_values_rpe = mapping.get_pixel_thickness_values()
    mapping.overlay_rgb_maps(rgb_values_ez, rgb_values_rpe)

    overlay.overlay_thickness_maps_to_oct(registration_data)
    overlay.overlay_transformed_stim_ellipses_to_oct(registration_data)

if __name__ == "__main__":
    main(test_grid_centres_flag=False)

