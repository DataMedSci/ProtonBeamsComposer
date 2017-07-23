import time
from copy import copy
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pbc.bragg_peak import BraggPeak
from pbc.sobp import SOBP


def calculate_number_of_peaks_gottshalk_80_rule(peak_to_measure, domain, spread):
    """
    Calculate number of peaks optimal for SOBP optimization
    using Gotshalck 80% rule.
    """
    temp_peak = copy(peak_to_measure)
    temp_peak.weight = 1.0
    width = temp_peak.width_at(domain=domain, val=0.8)
    n_of_optimal_peaks = int(np.ceil(spread / width))
    return n_of_optimal_peaks + 1


def test_optimize():
    start, stop, step = 0, 17.5, 0.01
    test_sobp = SOBP(inp_peaks, def_domain=[start, stop, step])
    print(test_sobp)
    print(test_sobp.positions())

    test_domain = np.arange(start, stop, step)
    sobp_vals = test_sobp.overall_sum()
    plt.plot(test_domain, sobp_vals, label='sum', color='red')
    plt.show()

    target = 15.0
    time_st = time.time()
    res = test_sobp.optimize_modulation(target_modulation=15, target_range=15)
    print("---------------------------------------------------")
    print("Time: %.2f (s)" % (time.time() - time_st))
    print(res)
    re = res['x']
    for idx, peak in enumerate(test_sobp.component_peaks):
        peak.weight = re[idx]
    sobp_vals = test_sobp.overall_sum()
    mod = test_sobp.modulation()
    ran = test_sobp.range()
    plateau_domain = np.arange(0, target, 0.1)
    plateau = test_sobp.overall_sum(plateau_domain)
    plateau_factor = sum([abs(pp - 1.0) for pp in plateau])

    print(mod, ran)
    plt.plot([start, stop], [1, 1], color='yellow')
    plt.plot(test_domain, sobp_vals, label='sum', color='red')
    plt.title("Modulation: {0:.3f}, Range: {1:.3f}, Md-diff: {2:.2f}, Plateau-diff: {3:.3f}"
              .format(mod, ran, abs(mod-target), plateau_factor))
    plt.show()

    # plot plateau only
    plt.plot([start, stop], [0.98, 0.98], color='orange')
    plt.plot([start, stop], [0.99, 0.99], color='green')
    plt.plot([start, stop], [1, 1], color='blue')
    plt.plot([start, stop], [1.02, 1.02], color='orange')
    plt.plot([start, stop], [1.01, 1.01], color='green')
    plt.plot(test_domain, sobp_vals, label='sum', color='red')
    plt.title("Modulation: {0:.3f}, Range: {1:.3f}, Md-diff: {2:.2f}, Plateau-diff: {3:.3f}"
              .format(mod, ran, abs(mod - target), plateau_factor))
    axes = plt.gca()
    axes.set_xlim([0, 15])
    axes.set_ylim([0.95, 1.05])
    plt.show()

if __name__ == '__main__':
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

    number_of_peaks = calculate_number_of_peaks_gottshalk_80_rule(peak_to_measure=testing_peak,
                                                                  domain=testing_domain, spread=15)
    print(number_of_peaks)

    # use 17 (as calculated by Gottshalk rule above) prepared peaks from file
    # todo: make this auto-adjustable, not file-hardcoded like now
    inp_peaks = [BraggPeak(x_peak, y_peak) for i in range(17)]

    for idx, peak in enumerate(inp_peaks):
        peak.position = positions[idx]
        peak.weight = weights[idx]

    test_optimize()
