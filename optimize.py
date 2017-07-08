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

    a = BraggPeak(x_peak, y_peak)
    a.position = 14.5
    a.weight = .8

    b = BraggPeak(x_peak, y_peak)
    b.position = 13.
    b.weight = .3

    c = BraggPeak(x_peak, y_peak)
    c.position = 11.5
    c.weight = .3

    d = BraggPeak(x_peak, y_peak)
    d.position = 9.
    d.weight = .3

    e = BraggPeak(x_peak, y_peak)
    e.position = 6.5
    e.weight = .3

    f = BraggPeak(x_peak, y_peak)
    f.position = 4.
    f.weight = .25

    g = BraggPeak(x_peak, y_peak)
    g.position = 1.
    g.weight = .2

    inp_peaks = [b, c, d, e, a, f, g]

    start, stop, step = 0, 25, 0.001
    test_sobp = SOBP(inp_peaks, def_domain=[start, stop, step])
    print(test_sobp)
    print(test_sobp.positions())

    test_domain = np.arange(start, stop, step)
    # sobp_vals = test_sobp.overall_sum()
    # plt.plot(test_domain, sobp_vals, label="sum", color="red")
    # plt.show()

    res = test_sobp.optimize_modulation(target_modulation=10.0)
    print("---------------------------------------------------")
    print(res)
    print(res['x'])
    # re = [0.29422737, 0.29419299, 0.30788737, 0.30591321, 0.79163851, 0.25610167, 0.196808]
    re = res['x']
    for idx, peak in enumerate(test_sobp.component_peaks):
        peak.weight = re[idx]
    sobp_vals = test_sobp.overall_sum()
    mod = test_sobp.modulation()
    ran = test_sobp.range()
    print(mod, ran)
    plt.plot([start, stop], [0.9, 0.9], color='yellow')
    plt.plot(test_domain, sobp_vals, label="sum", color="red")
    plt.title("Modulation: {0}, Range: {1}".format(mod, ran))
    plt.show()
