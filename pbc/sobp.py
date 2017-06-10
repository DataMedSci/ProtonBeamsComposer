import logging
from copy import copy

import numpy as np
import matplotlib.pyplot as plt

from pbc.bragg_peak import BraggPeak

logging.basicConfig(level=0)
logger = logging.getLogger(__name__)


class SOBP:
    def __init__(self, bragg_peaks, param_list=None):
        """
        If param_list=None assume bp is a list of BraggPeak instances.
        If param_list is given assume bp is a single BraggPeak and param_list
            contains positions of it with weights.
        """
        if isinstance(bragg_peaks, list) and len(bragg_peaks) > 0 and not param_list:
            try:
                for peak in bragg_peaks:
                    if not isinstance(peak, BraggPeak):
                        raise TypeError("Peak list should consist of BraggPeak objects!")
                self.component_peaks = bragg_peaks
                logger.debug("Creating sobp from peaks with positions:\n%s" % [bp.position for bp in bragg_peaks])
            except TypeError:
                logger.error("List should contain only BraggPeak instances!")
                raise
        elif isinstance(bragg_peaks, BraggPeak) and param_list:
            self.component_peaks = []
            for pos, wei in param_list:
                tmp_peak = copy(bragg_peaks)
                tmp_peak.position = pos
                tmp_peak.weight = wei
                self.component_peaks.append(tmp_peak)
        else:
            raise ValueError('Unsupported init data.')

    def __repr__(self):
        return repr([p.position for p in self.component_peaks])

    def overall_sum(self, domain):
        tmp_sobp = []
        for peak in self.component_peaks:
            tmp_peak = peak.evaluate(domain)
            tmp_sobp.append(tmp_peak)
        return sum(tmp_sobp)


if __name__ == '__main__':
    from os.path import join
    import pandas as pd

    with open(join("..", "bp.csv"), 'r') as bp_file:
        data = pd.read_csv(bp_file, sep=';')

    x_peak = data[data.columns[0]].as_matrix()
    y_peak = data[data.columns[1]].as_matrix()

    # load file with positions and weights
    with open(join("..", "pos.txt"), "r") as pos_file:
        pos_we_data = pd.read_csv(pos_file, sep=';')

    positions = pos_we_data['position'].as_matrix()
    weights = pos_we_data['weight'].as_matrix()

    print("Positions: %s" % positions)
    print("Weights: %s " % weights)

    b = BraggPeak(x_peak, y_peak)
    b.position = 18.
    b.weight = .1

    c = BraggPeak(x_peak, y_peak)
    c.position = 19.5
    c.weight = .15

    d = BraggPeak(x_peak, y_peak)
    d.position = 21.
    d.weight = .20

    e = BraggPeak(x_peak, y_peak)
    e.position = 22.5
    e.weight = .55

    inp_peaks = [b, c, d, e]
    test_sobp = SOBP(inp_peaks)

    test_domain = np.arange(15, 25, 0.1)
    plt.plot(test_domain, test_sobp.overall_sum(test_domain), label="sum")
    for p in inp_peaks:
        plt.plot(test_domain, p.evaluate(test_domain), label=p.position)
    plt.legend()
    plt.show()
