import time
from os.path import join

import logging
import pandas as pd
import numpy as np

from pbc.bragg_peak import BraggPeak
from pbc.helpers import calculate_number_of_peaks_gottschalk_80_rule, diff_max_from_left_99, diff_max_from_range_90, \
    make_precise_end_calculations
from pbc.plotting import plot_plateau, plot_sobp
from pbc.sobp import SOBP

logger = logging.getLogger(__name__)


def optimization_wrapper(input_peaks, target_modulation, target_range, disable_plots=False, preview_start_plot=False):
    """Just test some optimization options..."""
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
                  save_plot=False,
                  display_plot=True)

    time_st = time.time()
    res = test_sobp.optimize_modulation(target_modulation=target_modulation, target_range=target_range)
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
                  save_plot=False)

        plot_plateau(sobp_object=test_sobp,
                     target_modulation=target_modulation,
                     target_range=target_range)

    return test_sobp


def basic_optimization(input_args):
    """Test overall optimization capabilities for given spread and range"""
    if input_args.spread > input_args.range:
        logger.critical("Spread cannot be greater than range!")
        return -1
    elif not input_args.full:
        desired_range = input_args.range
        desired_modulation = input_args.spread

    with open(join('data', 'bp.csv'), 'r') as bp_file:
        data = pd.read_csv(bp_file, sep=';')

    x_peak = data[data.columns[0]].as_matrix()
    y_peak = data[data.columns[1]].as_matrix()

    testing_peak = BraggPeak(x_peak, y_peak)
    testing_domain = np.arange(0, 30, 0.001)

    if input_args.full == 'both':
        desired_range = testing_peak.range(testing_domain)
        desired_modulation = desired_range
    elif input_args.full == 'range':
        desired_range = testing_peak.range(testing_domain)
        desired_modulation = input_args.spread
    elif input_args.full == 'spread':
        desired_range = input_args.range
        desired_modulation = desired_range

    if input_args.halfmod:
        desired_modulation = desired_range / 2

    number_of_peaks = calculate_number_of_peaks_gottschalk_80_rule(peak_to_measure=testing_peak,
                                                                   domain=testing_domain,
                                                                   spread=desired_modulation)

    # in most cases this gives better results
    # todo: make this a parameter
    number_of_peaks += 1
    logger.info("Got %s peaks from Gottschalk rule calculation." % number_of_peaks)

    # use Gottschalk Rule result to generate list of input peaks
    inp_peaks = [BraggPeak(x_peak, y_peak) for _ in range(number_of_peaks)]

    # base positions of peaks on GR result, range and desired modulation
    base_position = desired_range - desired_modulation

    push_first_peak = diff_max_from_left_99(inp_peaks[-1])
    pull_back_last_peak = diff_max_from_range_90(inp_peaks[-1])

    begin = base_position + push_first_peak
    end = desired_range - pull_back_last_peak

    starting_positions = np.linspace(start=begin, stop=end, num=number_of_peaks)

    for idx, peak in enumerate(inp_peaks):
        peak.position = starting_positions[idx]
        peak.weight = 0.1
    inp_peaks[-1].weight = 0.9

    res_sobp_object = optimization_wrapper(input_peaks=inp_peaks,
                                           target_modulation=desired_modulation,
                                           target_range=desired_range,
                                           disable_plots=input_args.no_plot)

    left_res, right_res = make_precise_end_calculations(res_sobp_object)

    logger.info("\n\tPosition of 0.99 from left is {0}\n\tTarget val was: {1}\n\tDiff of left vals: {2}".format(
                left_res, base_position, abs(base_position - left_res)))
    logger.info("\n\tPosition of 0.9 from right {0}\n\tTarget val was: {1}\n\tDiff of right vals: {2}".format(
                right_res, desired_range, abs(desired_range - right_res)))

    plot_plateau(sobp_object=res_sobp_object,
                 target_modulation=desired_modulation,
                 target_range=desired_range - pull_back_last_peak,
                 dump_data=True,
                 dump_path='plateau.csv')

    # calculate difference between desired range and actual SOBP range we got from optimization
    right_error = desired_range - right_res

    # todo: analyze Gottschalk rule calculation (probably right_error * 1.2 factor comes from there...)
    corrected_starting_positions = np.linspace(start=begin, stop=end + right_error * 1.2, num=number_of_peaks)

    for idx, peak in enumerate(inp_peaks):
        peak.position = corrected_starting_positions[idx]
        peak.weight = 0.1
    inp_peaks[-1].weight = 0.9

    res_sobp_object = optimization_wrapper(input_peaks=inp_peaks,
                                           target_modulation=desired_modulation,
                                           target_range=desired_range,
                                           disable_plots=input_args.no_plot)

    left_res, right_res = make_precise_end_calculations(res_sobp_object)

    logger.info("\n\tPosition of 0.99 from left is {0}\n\tTarget val was: {1}\n\tDiff of left vals: {2}".format(
                left_res, base_position, abs(base_position - left_res)))
    logger.info("\n\tPosition of 0.9 from right {0}\n\tTarget val was: {1}\n\tDiff of right vals: {2}".format(
                right_res, desired_range, abs(desired_range - right_res)))

    new_right_error = abs(desired_range - right_res)
    logger.log(25, "Corrected right end at 0.90:\n\tfrom: {0:.16f}\n\tto: {1:.16f}\n\tbetter by: {2:.16f}".format(
               right_error, new_right_error, right_error - new_right_error))

    plot_plateau(sobp_object=res_sobp_object,
                 target_modulation=desired_modulation,
                 target_range=desired_range - pull_back_last_peak,
                 dump_data=True,
                 dump_path='corrected_plateau.csv')
