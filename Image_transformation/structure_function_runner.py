from Image_transformation.fundus_mapping import MicroperimetryParser, MicroperimetryCoordinateCalculator, FundusOverlayRenderer
from Image_transformation.oct_thickness_mapping import OctThicknessMapper
from Image_transformation.fundus_registration import FundusRegistrator
from Image_transformation.structure_function_evaluation import StructureFunctionAnalysis
from Image_transformation.mp_oct_enums_helpers import (DB_COLOUR_VALUES, MP_DIR, OUTPUT_DIR, OCT_DIR, METADATA_KEYS,
                                                       _test_grid_centres, _number_multiple_recordings_at_timepoint,
                                                       AAVB_FONT, EZ_LUT, RPE_LUT, SLO_PX_PER_DEG, _ez_and_rpe_maps_on_slo_img,
                                                       _plot_ez_integrity_and_sensitivity_per_timepoint, _ez_loss_border_distance_to_stim)
import os
import time

def main(run_debug_checks=True):
    timestamp=time.strftime("%Y-%m-%d_%H%M")

    # Create mp_parser instance to gather microperimetry (mp) data from the xml files
    mp_parser = MicroperimetryParser(MP_DIR, METADATA_KEYS)
    mp_files, patient_id = mp_parser.get_mp_list()
    exams = mp_parser.parse_mp_xml_data(mp_files, patient_id)
    # add a counter to the baseline mp recordings to distinguish the duplicates. Only use one for registration
    exams = _number_multiple_recordings_at_timepoint(exams)

    # Instantiate the 'coord_calculator' object and pass it the 'exams' mp data to find the mp_stim loci coordinates
    coord_calculator = MicroperimetryCoordinateCalculator(exams)
    exams_with_coords = coord_calculator.calculate_pixel_coordinates()

    # This call is used to check for the centre grid points during debugging
    if run_debug_checks:
        for patient in exams:
            _test_grid_centres(patient)

    registrator = FundusRegistrator(exams_with_coords, OCT_DIR, OUTPUT_DIR, AAVB_FONT, timestamp)
    oct_files = registrator.get_slo_list()

    mp_oct_pairs = registrator.match_oct_with_mp(oct_files)

    # identify matches for debugging
    for pair in mp_oct_pairs:
        print(pair['patient_id'], pair['exam_id'], pair['eye'], pair['timepoint'], '->', os.path.basename(pair['slo_file']))

    overlay_renderer = FundusOverlayRenderer(exams_with_coords, DB_COLOUR_VALUES, MP_DIR, OUTPUT_DIR, timestamp, AAVB_FONT)

    overlay_renderer.draw_mp_data_to_maia_scan()
    registered_pairs = registrator.mp_oct_fundus_registration(mp_oct_pairs)

    thickness_mapper = OctThicknessMapper(registered_pairs, EZ_LUT, RPE_LUT)
    rgb_values_ez, rgb_values_rpe = thickness_mapper.get_pixel_thickness_values()

    # check: EZ-loss outline on each SLO (thickness maps must line up with the SLO)
    if run_debug_checks:
        _ez_and_rpe_maps_on_slo_img(registered_pairs)

    thickness_mapper.overlay_rgb_maps(rgb_values_ez, rgb_values_rpe)

    overlay_renderer.overlay_thickness_maps_to_oct(registered_pairs)
    overlay_renderer.overlay_transformed_stim_ellipses_to_oct(registered_pairs)

    _ez_loss_border_distance_to_stim(registered_pairs)

    sf_analysis = StructureFunctionAnalysis(registered_pairs, timestamp, SLO_PX_PER_DEG, OUTPUT_DIR)
    sf_analysis.direct_structure_function_measurement()

    if run_debug_checks:
        _plot_ez_integrity_and_sensitivity_per_timepoint(registered_pairs)

    sf_analysis.sf_linear_mixed_model_fitting(thickness_cols=['ez_thickness_at_loci'], plot_name=f"EZ loss-only", zero_neg_mp_responses=True)
    sf_analysis.sf_linear_mixed_model_fitting(thickness_cols=['rpe_thickness_at_loci'], plot_name=f"RPE loss-only", zero_neg_mp_responses=True)
    sf_analysis.sf_linear_mixed_model_fitting(plot_name=f"EZ and RPE combined model", zero_neg_mp_responses=True)

    sf_analysis.interpolated_kriged_sensitivity_map()

    # sf_analysis.scotoma_and_layer_loss_correlation()
    # sf_analysis.perilesion_sf_analysis()

if __name__ == "__main__":
    main()
