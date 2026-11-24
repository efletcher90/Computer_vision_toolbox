from Image_transformation.fundus_mapping import CreateMicroperimetryDataDicts, CalculateMicroperimetryCoordinates, FundusOverlays
from Image_transformation.oct_thickness_mapping import OctThicknessMapping
from Image_transformation.fundus_registration import FundusRegistrator
from Image_transformation.structure_function_evaluation import StructureFunctionAnalysis
from Image_transformation.mp_oct_enums_helpers import (DB_COLOUR_VALUES, MP_DIR, OUTPUT_DIR, OCT_DIR, METADATA_KEYS,
                                                       _test_grid_centres, _number_multiple_recordings_at_timepoint,
                                                       AAVB_FONT, EZ_LUT, RPE_LUT, SLO_PX_PER_DEG, _ez_and_rpe_maps_on_slo_img,
                                                       _plot_ez_integrity_and_sensitivity_per_timepoint, _offset_test, _ez_loss_border_distance_to_stim)
import os
import time

def main(run_checks=True):
    timestamp=time.strftime("%Y-%m-%d_%H%M")
    xml_parser = CreateMicroperimetryDataDicts(MP_DIR, METADATA_KEYS)

    mp_files, patient_id = xml_parser.get_mp_list()
    patient_data_dicts = xml_parser.parse_mp_xml_data(mp_files, patient_id)
    patient_data_dicts = _number_multiple_recordings_at_timepoint(patient_data_dicts)

    calculator = CalculateMicroperimetryCoordinates(patient_data_dicts)
    data_with_coords = calculator.calculate_pixel_coordinates()

    # This call is used to check for the centre grid points on
    if run_checks:
        for patient in patient_data_dicts:
            _test_grid_centres(patient)

    oct_parser = FundusRegistrator(data_with_coords, OCT_DIR, OUTPUT_DIR, AAVB_FONT, timestamp)
    oct_file_list = oct_parser.get_slo_list()

    mp_oct_matches = oct_parser.match_oct_with_mp(oct_file_list)

    # identify matches for debugging
    for match in mp_oct_matches:
        print(match['patient_id'], match['exam_id'], match['eye'], match['timepoint'], '->', os.path.basename(match['slo_file']))

    overlay = FundusOverlays(data_with_coords, DB_COLOUR_VALUES, MP_DIR, OUTPUT_DIR, timestamp, AAVB_FONT)

    overlay.draw_mp_data_to_maia_scan()
    registration_data = oct_parser.mp_oct_fundus_registration(mp_oct_matches)

    mapping = OctThicknessMapping(registration_data, EZ_LUT, RPE_LUT)
    rgb_values_ez, rgb_values_rpe = mapping.get_pixel_thickness_values()

    # check: EZ-loss outline on each SLO (thickness maps must line up with the SLO)
    if run_checks:
        _ez_and_rpe_maps_on_slo_img(registration_data)

    mapping.overlay_rgb_maps(rgb_values_ez, rgb_values_rpe)

    overlay.overlay_thickness_maps_to_oct(registration_data)
    overlay.overlay_transformed_stim_ellipses_to_oct(registration_data)

    _ez_loss_border_distance_to_stim(registration_data)

    analysis = StructureFunctionAnalysis(registration_data, timestamp, SLO_PX_PER_DEG, OUTPUT_DIR)
    analysis.direct_structure_function_measurement()

    # add other stim_ids if you want to highglight on gs slo image.
    # The fn kwarg for 'edgecolors' needs to be 'yellow' or not None
    if run_checks:
        _offset_test(registration_data)
        _plot_ez_integrity_and_sensitivity_per_timepoint(registration_data, {'stimulus_07'})

    analysis.sf_linear_mixed_model_fitting(plot_name=f"Combined model: time+stim_id adj")
    analysis.sf_linear_mixed_model_fitting(thickness_cols=['direct_ez_mean_thickness'], plot_name=f"EZ loss-only: time+stim_id adj")
    analysis.sf_linear_mixed_model_fitting(thickness_cols=['direct_rpe_mean_thickness'], plot_name=f"RPE loss-only: time+stim_id adj")

    analysis.sf_linear_mixed_model_fitting(not_seen='floor')
    analysis.sf_linear_mixed_model_fitting(exclude_border_px=25)


    analysis.interpolated_kriged_sensitivity_map()

    # sf_data.scotoma_and_layer_loss_correlation()
    # sf_data.perilesion_sf_analysis()

if __name__ == "__main__":
    main()
