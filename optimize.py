import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pbc.bragg_peak import BraggPeak
from pbc.sobp import SOBP


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

    p1 = BraggPeak(x_peak, y_peak)
    p2 = BraggPeak(x_peak, y_peak)
    p3 = BraggPeak(x_peak, y_peak)
    p4 = BraggPeak(x_peak, y_peak)
    p5 = BraggPeak(x_peak, y_peak)
    p6 = BraggPeak(x_peak, y_peak)
    p7 = BraggPeak(x_peak, y_peak)
    p8 = BraggPeak(x_peak, y_peak)
    p9 = BraggPeak(x_peak, y_peak)
    p10 = BraggPeak(x_peak, y_peak)
    p11 = BraggPeak(x_peak, y_peak)
    p12 = BraggPeak(x_peak, y_peak)
    p13 = BraggPeak(x_peak, y_peak)
    p14 = BraggPeak(x_peak, y_peak)
    p15 = BraggPeak(x_peak, y_peak)

    inp_peaks = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]

    lng = len(inp_peaks)
    for idx, peak in enumerate(inp_peaks):
        peak.position = positions[idx]
        peak.weight = weights[idx]

    start, stop, step = 0, 25, 0.001
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
    print(mod, ran)
    plt.plot([start, stop], [0.9, 0.9], color='yellow')
    plt.plot(test_domain, sobp_vals, label="sum", color="red")
    plt.title("Modulation: {0:.3f}, Range: {1:.3f}, Diff: {2:.2f}".format(mod, ran, abs(mod-target)))
    plt.show()
