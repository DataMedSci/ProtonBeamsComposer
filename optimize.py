import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from copy import copy

from pbc.bragg_peak import BraggPeak
from pbc.sobp import SOBP


def calculate_number_of_peaks_gottshalk_80_rule(peak, domain, spread):
    """
    Calculate number of peaks optimal for SOBP optimization
    using Gotshalck 80% rule.
    """
    temp_peak = copy(peak)
    temp_peak.weight = 1.0
    width = temp_peak.width_at(domain=domain, val=0.8)
    print(width)
    n_of_optimal_peaks = int(np.ceil(spread / width))
    return n_of_optimal_peaks + 1


def test_optimize():
    start, stop, step = 0, 20, 0.01
    test_sobp = SOBP(inp_peaks, def_domain=[start, stop, step])
    print(test_sobp)
    print(test_sobp.positions())

    test_domain = np.arange(start, stop, step)
    sobp_vals = test_sobp.overall_sum()
    plt.plot(test_domain, sobp_vals, label="sum", color="red")
    plt.show()

    target = 15.0
    import time
    time_st = time.time()
    res = test_sobp.optimize_modulation(target_modulation=target)
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
    plateau_factor = sum([abs(pp - 0.9) for pp in plateau])

    print(mod, ran)
    plt.plot([start, stop], [0.9, 0.9], color='yellow')
    plt.plot(test_domain, sobp_vals, label="sum", color="red")
    plt.title("Modulation: {0:.3f}, Range: {1:.3f}, Md-diff: {2:.2f}, Plateau-diff: {3:.3f}"
              .format(mod, ran, abs(mod-target), plateau_factor))
    plt.show()

if __name__ == '__main__':
    with open("bp.csv", 'r') as bp_file:
        data = pd.read_csv(bp_file, sep=';')

    x_peak = data[data.columns[0]].as_matrix()
    y_peak = data[data.columns[1]].as_matrix()

    # load file with positions and weights
    with open("pos.txt", "r") as pos_file:
        pos_we_data = pd.read_csv(pos_file, sep=';')

    positions = pos_we_data['position'].as_matrix()
    weights = pos_we_data['weight'].as_matrix()

    print("Positions: %s" % positions)
    print("Weights: %s " % weights)

    testing_peak = BraggPeak(x_peak, y_peak)
    testing_domain = np.arange(0, 30, 0.001)

    number_of_peaks = calculate_number_of_peaks_gottshalk_80_rule(peak=testing_peak, domain=testing_domain, spread=15)
    print(number_of_peaks)

    # use 17 (as calculated by Gottshalk rule above) prepared peaks from file
    # todo: make this auto-adjustable, not file-hardcoded like now
    inp_peaks = [BraggPeak(x_peak, y_peak) for i in range(17)]

    lng = len(inp_peaks)
    for idx, peak in enumerate(inp_peaks):
        peak.position = positions[idx]
        peak.weight = weights[idx]

    test_optimize()
