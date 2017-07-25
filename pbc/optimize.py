import time
from os.path import join

import logging
import pandas as pd
import numpy as np

from pbc.bragg_peak import BraggPeak
from pbc.helpers import calculate_number_of_peaks_gottshalk_80_rule
from pbc.plotting import plot_plateau, plot_sobp
from pbc.sobp import SOBP

logger = logging.getLogger(__name__)


def test_optimize(input_peaks, target_range, target_modulation):
    """Just test some optimization options..."""
    start, stop, step = 0, target_range + 2.5, 0.01
    test_sobp = SOBP(input_peaks, def_domain=[start, stop, step])
    print(test_sobp)
    print(test_sobp.positions())

    plot_sobp(start=start,
              stop=stop,
              step=step,
              sobp_object=test_sobp,
              helper_lines=False)

    time_st = time.time()
    res = test_sobp.optimize_modulation(target_modulation=target_modulation, target_range=target_range)
    print("---------------------------------------------------")
    print("Time: %.2f (s)" % (time.time() - time_st))
    print(res)

    # apply calculated weights to peaks
    optimization_results = res['x']
    for peak_idx, peak_object in enumerate(test_sobp.component_peaks):
        peak_object.weight = optimization_results[peak_idx]

    plot_sobp(start=start,
              stop=stop,
              sobp_object=test_sobp,
              target_modulation=target_modulation,
              target_range=target_range,
              helper_lines=True)

    plot_plateau(sobp_object=test_sobp,
                 target_modulation=target_modulation,
                 target_range=target_range)


def main(input_args):
    if input_args.spread > input_args.range:
        logger.critical("Spread cannot be greater than range!")
        return

    with open(join('data', 'bp.csv'), 'r') as bp_file:
        data = pd.read_csv(bp_file, sep=';')

    x_peak = data[data.columns[0]].as_matrix()
    y_peak = data[data.columns[1]].as_matrix()

    # load file with positions and weights
    with open(join('data', 'pos.txt'), 'r') as pos_file:
        pos_we_data = pd.read_csv(pos_file, sep=';')

    positions = pos_we_data['position'].as_matrix()
    weights = pos_we_data['weight'].as_matrix()

    print("Positions: %s" % positions)
    print("Weights: %s " % weights)

    testing_peak = BraggPeak(x_peak, y_peak)
    testing_domain = np.arange(0, 30, 0.001)

    desired_range = 15.0  # testing_peak.range(testing_domain)
    desired_modulation = 15.0

    number_of_peaks = calculate_number_of_peaks_gottshalk_80_rule(peak_to_measure=testing_peak,
                                                                  domain=testing_domain,
                                                                  spread=desired_modulation)

    logger.info("Got %s peaks from Gottshalck rule calculation." % number_of_peaks)

    # use Gottshalk Rule result to generate list of input peaks
    inp_peaks = [BraggPeak(x_peak, y_peak) for _ in range(number_of_peaks)]

    # base positions of peaks on GR result, range and desired modulation
    base_position = desired_range - desired_modulation
    starting_positions = np.linspace(base_position, desired_range, number_of_peaks)
    for idx, peak in enumerate(inp_peaks):
        peak.position = starting_positions[idx]  # positions[idx]
        peak.weight = 0.1  # weights[idx]
    inp_peaks[-1].weight = 0.9

    test_optimize(inp_peaks, desired_range, desired_modulation)
