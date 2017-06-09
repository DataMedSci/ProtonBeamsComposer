import logging

import numpy as np
import matplotlib.pyplot as plt

import beprof

from pbc.bragg_peak import BraggPeak

logging.basicConfig(level=0)
logger = logging.getLogger(__name__)


class SOBPOld:
    def __init__(self, peak_list):
        if isinstance(peak_list, list) and len(peak_list) > 0:
            logger.debug("Creating SOBP...")
            self.sobp = self.calc_sobp(peak_list)
        else:
            raise TypeError("List object required!")

    @staticmethod
    def calc_sobp(plist):
        recalculated_peaks = []
        for p in plist:
            primary = p["properties"]["primary_position"]
            shifted = p["properties"]["shifted_position"]
            height = p["properties"]["height"]
            # if primary != shifted ?? some a-b<delta
            if abs(primary - shifted) > 0.0001:
                temp_x = p["bp"].x + (shifted - primary)
                logger.debug("Shifting by %.4f" % (shifted - primary))
            else:
                temp_x = p["bp"].x
                logger.debug("No shifting required.")
            temp_y = p["bp"].y * p["properties"]["height"]
            temp_peak = beprof.profile.Profile(np.array([temp_x, temp_y]).T)
            # normalize?
            temp_peak.rescale(temp_peak.y.max())
            # todo: is it all mandatory?... improve - sum to one Profile
            recalculated_peaks.append(temp_peak)
            logger.debug(
                "\t(calc. SOBP) Got BraggPeak with primary position %.2f, shifted position %.2f, height %.2f" %
                (primary, shifted, height)
            )
        return recalculated_peaks

    def val(self, x):
        temp_val = 0
        for p in self.sobp:
            temp_val += p.y_at_x(x)
        return temp_val

    def calc_on_mesh(self, mesh_array):
        """Calc chi^2 on given mesh"""
        result = 0
        logger.debug("Calc chi^2 on mesh...")
        for m in mesh_array:
            temp = (self.val(m) - 1) ** 2
            logger.debug("\t(Calc chi) Got %.4f on position %.2f" % (self.val(m), m))
            if str(temp) != 'nan':
                result += temp
            else:
                raise ValueError("Got 'nan' instead of number!")
        return result


class SOBP:
    def __init__(self, peak_list):
        if not isinstance(peak_list, list) or not len(peak_list) > 0:
            raise TypeError("List object required!")
        for peak in peak_list:
            if not isinstance(peak, BraggPeak):
                raise TypeError("Peak list should consist of BraggPeak objects!")
        self.component_peaks = peak_list

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

    a = BraggPeak(x_peak, y_peak)

    a.weight = .75
    a.position = 12.5

    s = SOBP([a, BraggPeak([1, 2, 3, 4, 5], [.1, .2, .5, 1, .2])])

    dom = np.arange(0, 30, .1)
    so = s.overall_sum(dom)

    plt.plot(dom, so)
    plt.show()

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
