from microperimetry_mapping import CreateMPDataDicts, CalculateStimCoordinates, OverlayEllipsesToFundus
from fundus_registration import FundusRegistrator
from Image_transformation.mp_oct_enums_helpers import DB_COLOUR_VALUES, MP_DIR, OCT_DIR, METADATA_KEYS, _test_grid_centres

def main():
    xml_parser = CreateMPDataDicts(MP_DIR, METADATA_KEYS)

    mp_files, patient_id = xml_parser.get_mp_list()
    patient_data_dicts = xml_parser.parse_mp_xml_data(mp_files, patient_id)

    calculator = CalculateStimCoordinates(patient_data_dicts)
    data_with_coords = calculator.calculate_pixel_coordinates()

    # This call is used to check for the centre grid points on
    # for patient in patient_data_dicts:
    #     _test_grid_centres(patient)

    overlay = OverlayEllipsesToFundus(data_with_coords, DB_COLOUR_VALUES, MP_DIR)
    mp_saved_folder = overlay.draw_overlays()



    oct_registrator = FundusRegistrator(OCT_DIR)
    oct_saved_folder = oct_registrator.run(data_with_coords)

    print(f"MP overlays saved to: {mp_saved_folder}")
    print(f"OCT overlays saved to: {oct_saved_folder}")

if __name__ == "__main__":
    main()
