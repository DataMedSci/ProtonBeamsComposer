import time
from os.path import join

import logging
import numpy as np
import shutil

from pbc.bragg_peak import BraggPeak
from pbc.helpers import calculate_number_of_peaks_gottschalk_80_rule, diff_max_from_left_99, diff_max_from_range_90, \
    make_precise_end_calculations, load_data_from_dump, create_output_dir
from pbc.plotting import plot_plateau, plot_sobp
from pbc.sobp import SOBP

logger = logging.getLogger(__name__)


def optimization_wrapper(input_peaks, target_modulation, target_range, output_dir=None, disable_plots=False,
                         preview_start_plot=False, options_for_optimizer=None):
    """
    Optimization wrapper.

    :param input_peaks: peaks used in optimize process
    :param target_modulation: desired modulation aimed for
    :param target_range: desired range (distal) aimed for
    :param output_dir: path for plots etc.
    :param disable_plots: disables all the plots in this function
    :param preview_start_plot: shows preview plot before optimization
    :param options_for_optimizer: dict with options for scipy optimize, given options override default
    """
    start, stop, step = 0, target_range + 2.5, 0.01
    test_sobp = SOBP(input_peaks, def_domain=[start, stop, step])
    logger.info(test_sobp)
    logger.debug(test_sobp.positions())

    if preview_start_plot:
        plot_sobp(start=start,
                  stop=stop,
                  step=step,
                  sobp_object=test_sobp,
                  helper_lines=False,
                  plot_path=join(output_dir, 'preview_sobp.png'),
                  display_plot=True,
                  datafile_path=join(output_dir, 'preview_sobp.dat'))

    time_st = time.time()
    res = test_sobp.optimize_sobp(target_modulation=target_modulation,
                                  target_range=target_range,
                                  optimization_options=options_for_optimizer)
    logger.info("Optimization function took {0:.2f} seconds".format(time.time() - time_st))

    logger.info("Optimization output:\n{0}".format(res))

    # apply calculated weights to peaks
    optimization_results = res['x']
    for peak_idx, peak_object in enumerate(test_sobp.component_peaks):
        peak_object.weight = optimization_results[peak_idx]

    if not disable_plots:
        plot_sobp(start=start,
                  stop=stop,
                  sobp_object=test_sobp,
                  target_modulation=target_modulation,
                  target_range=target_range,
                  helper_lines=True,
                  plot_path=join(output_dir, 'sobp.png'),
                  datafile_path=join(output_dir, 'sobp.dat'))

        plot_plateau(sobp_object=test_sobp,
                     target_modulation=target_modulation,
                     target_range=target_range,
                     higher=False,
                     plot_path=join(output_dir, 'plateau_zoom.png'),
                     datafile_path=join(output_dir, 'plateau_zoom.dat'))

    logger.info("Optimization wrapper finished.")

    return test_sobp


def basic_optimization(input_args):
    """Test overall optimization capabilities for given spread and range"""

    # create output dir
    output_dir = create_output_dir(input_args.name)

    # log to file in output dir
    file_log = logging.FileHandler(filename=join(output_dir, 'optimization.log'), mode='w')
    logging.getLogger().addHandler(file_log)

    if input_args.spread > input_args.range:
        logger.critical("Spread cannot be greater than range!")
        return -1
    elif not input_args.full:
        desired_range = input_args.range
        desired_modulation = input_args.spread

    # this is some measured data generated using DataMedSci/pymchelper --plotdata
    # option and SHIELD-HIT12A simulation results
    if not input_args.input_bp_file:
        x_peak, y_peak = load_data_from_dump(file_name=join('data', 'cydos_new.csv'), delimiter=';')
        shutil.copy(join('data', 'cydos_new.csv'), join(output_dir, 'bp.dat'))
        logging.debug("Copying peak database ({0}) to output dir as bp.dat.".format('cydos_new.csv'))
    else:
        x_peak, y_peak = load_data_from_dump(file_name=input_args.input_bp_file, delimiter=input_args.delimeter)
        shutil.copy(input_args.input_bp_file, join(output_dir, 'bp.dat'))
        logging.debug("Copying peak database specified by user ({0}) to output dir as bp.dat.".format(
                      input_args.input_bp_file))

    # if it is in centimeters convert to millimeters
    if x_peak.max() < 10:
        x_peak *= 10
        logger.warning("Multiplying initial peak values by 10!")

    # we want values to be in range <0; 1>
    y_peak /= y_peak.max()

    if input_args.smooth:
        from scipy.signal import savgol_filter
        logger.info("Applying filter to input data.")
        if input_args.window:
            y_peak = savgol_filter(y_peak, window_length=input_args.window, polyorder=3)
            logger.info("Filter window = {0} used.".format(input_args.window))
        else:
            y_peak = savgol_filter(y_peak, window_length=5, polyorder=3)
            logger.info("Filter window = {0} used.".format(5))

    testing_peak = BraggPeak(x_peak, y_peak)
    if input_args.range > testing_peak.range():
        raise ValueError("Impossible range specified: {0}, max range of peak is {1}."
                         "\nUse --full range to generate full-range SOBP."
                         .format(input_args.range, testing_peak.range()))

    if input_args.full == 'both':
        desired_range = testing_peak.range(val=0.90)
        desired_modulation = desired_range
        logger.info("Using full-range ({0}), full-modulation option ({1}).".format(desired_range, desired_modulation))
    elif input_args.full == 'range':
        desired_range = testing_peak.range(val=0.90)
        desired_modulation = input_args.spread
        logger.info("Using full-range ({0}) option. Desired spread = {1}".format(desired_range, desired_modulation))
    elif input_args.full == 'spread':
        desired_range = input_args.range
        desired_modulation = desired_range
        logger.info("Using full-modulation ({0}) option. Desired range = {1}".format(desired_modulation, desired_range))

    if input_args.halfmod and input_args.full != 'spread':
        desired_modulation = desired_range / 2
        logger.info("Using half-modulation ({0}) option.".format(desired_modulation))

    if input_args.peaks:
        number_of_peaks = input_args.peaks
        logger.info("Using {0} as number of peaks in optimization.".format(input_args.peaks))
    else:
        number_of_peaks = calculate_number_of_peaks_gottschalk_80_rule(peak_to_measure=testing_peak,
                                                                       spread=desired_modulation)

        logger.info("Got {0} peaks from Gottschalk rule calculation.".format(number_of_peaks))

        if input_args.add_to_gott:
            number_of_peaks += input_args.add_to_gott
            logger.info("Added {0} peak(s) to Gottschalk's rule calculation result. Now it is {1} peaks total.".format(
                        input_args.add_to_gott, number_of_peaks))

    # generate list of input peaks
    inp_peaks = [BraggPeak(x_peak, y_peak) for _ in range(number_of_peaks)]

    # base positions of peaks, range and desired modulation
    base_position = desired_range - desired_modulation

    # pull back last peak, especially when calculating max range SOBP,
    # because position_of_max == distal_range is impossible to achieve in lab
    pull_back_last_peak = diff_max_from_range_90(inp_peaks[-1])

    # todo: allow position of first peak to equal 0.0?
    if base_position == 0:
        begin = base_position + 0.0001
    else:
        begin = base_position
    end = desired_range - pull_back_last_peak

    starting_positions = np.linspace(start=begin, stop=end, num=number_of_peaks)
    logger.info("First setup for peaks is start = {0:.3f}; end= {1:.3f}".format(begin, end))

    for idx, peak in enumerate(inp_peaks):
        peak.position = starting_positions[idx]
        peak.weight = 0.1
    inp_peaks[-1].weight = 0.9

    # just make quick calculation without going too deep
    first_opt_dict = {'disp': False, 'eps': 1e-4, 'ftol': 1e-4, 'gtol': 1e-4}

    logger.info("Running initial optimization...")
    res_sobp_object = optimization_wrapper(input_peaks=inp_peaks,
                                           target_modulation=desired_modulation,
                                           target_range=desired_range,
                                           output_dir=output_dir,
                                           disable_plots=True,  # input_args.no_plot,
                                           options_for_optimizer=first_opt_dict)

    left_res, right_res = make_precise_end_calculations(res_sobp_object)

    logger.info("Position of 0.99 from left is {0}\n\tTarget val was: {1}\n\tDiff of left vals: {2}".format(
                left_res, base_position, abs(base_position - left_res)))
    logger.info("Position of 0.9 from right {0}\n\tTarget val was: {1}\n\tDiff of right vals: {2}".format(
                right_res, desired_range, abs(desired_range - right_res)))

    # calculate difference between desired proximal/distal range and what we got from optimization
    # for proximal - do not shift if generating full modulation or proximal >= 0.99 is already satisfied
    if not input_args.full == 'both' and desired_range != desired_modulation and \
       res_sobp_object.y_at_x(base_position) <= 0.99:
        left_error = base_position - left_res
        logger.info("Left (99) error after first optimization is: {0}".format(left_error))
    else:
        left_error = 0
        logger.info("Left (99) is OK! Current val: {0}".format(res_sobp_object.y_at_x(base_position)))

    right_error = desired_range - right_res
    logger.info("Right (90) error after first optimization is: {0}".format(right_error))

    if end + right_error > testing_peak.range():
        logger.critical("Shifting position exceeds range of base peak!")
        raise ValueError("Shifting failed!")

    corrected_starting_positions = np.linspace(start=begin + left_error, stop=end + right_error, num=number_of_peaks)

    plot_plateau(sobp_object=res_sobp_object,
                 target_modulation=desired_modulation,
                 target_range=desired_range,
                 datafile_path=join(output_dir, 'preview_plateau.dat'),
                 plot_path=join(output_dir, 'preview_plateau.png'))

    for idx, peak in enumerate(inp_peaks):
        peak.position = corrected_starting_positions[idx]
        peak.weight = 0.1
    inp_peaks[-1].weight = 0.9

    res_sobp_object = optimization_wrapper(input_peaks=inp_peaks,
                                           target_modulation=desired_modulation,
                                           target_range=desired_range,
                                           output_dir=output_dir,
                                           disable_plots=input_args.no_plot)

    left_res, right_res = make_precise_end_calculations(res_sobp_object)

    logger.info("Position of 0.99 from left is {0}\n\tTarget val was: {1}\n\tDiff of left vals: {2}".format(
                left_res, base_position, abs(base_position - left_res)))
    logger.info("Position of 0.9 from right {0}\n\tTarget val was: {1}\n\tDiff of right vals: {2}".format(
                right_res, desired_range, abs(desired_range - right_res)))

    new_right_error = abs(desired_range - right_res)
    logger.log(25, "Corrected right end at 0.90:\n\tfrom: {0:.16f}\n\tto: {1:.16f}\n\tbetter by: {2:.16f}".format(
               right_error, new_right_error, right_error - new_right_error))

    plot_plateau(sobp_object=res_sobp_object,
                 target_modulation=desired_modulation,
                 target_range=desired_range,
                 datafile_path=join(output_dir, 'corrected_plateau.dat'),
                 plot_path=join(output_dir, 'corrected_plateau.png'))

    logger.info("Optimization process finished")
