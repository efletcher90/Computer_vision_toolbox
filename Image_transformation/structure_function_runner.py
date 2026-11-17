from microperimetry_mapping import CreateMPDataDicts, CalculateStimCoordinates, OverlayEllipsesToFundus
from fundus_registration import FundusRegistrator
from Image_transformation.mp_oct_enums_helpers import DB_COLOUR_VALUES, MP_DIR, OCT_DIR, METADATA_KEYS, _test_grid_centres, _number_multiple_recordings_in_visit
import os

def main(test_grid_centres_flag=False):
    xml_parser = CreateMPDataDicts(MP_DIR, METADATA_KEYS)

    mp_files, patient_id = xml_parser.get_mp_list()
    patient_data_dicts = xml_parser.parse_mp_xml_data(mp_files, patient_id)
    patient_data_dicts = _number_multiple_recordings_in_visit(patient_data_dicts)

    calculator = CalculateStimCoordinates(patient_data_dicts)
    data_with_coords = calculator.calculate_pixel_coordinates()

    # This call is used to check for the centre grid points on
    if test_grid_centres_flag:
        for patient in patient_data_dicts:
            _test_grid_centres(patient)


    oct_parser = FundusRegistrator(data_with_coords, OCT_DIR)
    oct_files = oct_parser.get_slo_list()

    matched_mp_oct = oct_parser.match_oct_with_mp(data_with_coords, oct_files)

    for match in matched_mp_oct:
        print(match['patient_id'], match['exam_id'], match['eye'], match['timepoint'], '->',
              os.path.basename(match['slo_file']))


    # Need to transform the stimulation sites based on the registration -- so the method in the call below needs changing
    overlay = OverlayEllipsesToFundus(data_with_coords, DB_COLOUR_VALUES, MP_DIR)
    mp_saved_folder = overlay.draw_overlays()





    print(f"MP overlays saved to: {mp_saved_folder}")
    # print(f"OCT overlays saved to: {oct_saved_folder}")

if __name__ == "__main__":
    main(test_grid_centres_flag=False)
