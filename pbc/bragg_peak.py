import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

logging.basicConfig(level=0)
logger = logging.getLogger(__name__)


class BraggPeak:
    def __init__(self, bp_domain, bp_vals):
        if len(bp_domain) != len(bp_vals):
            raise ValueError("Domain and values have different lengths!")
        self.spline = interpolate.InterpolatedUnivariateSpline(bp_domain, bp_vals, ext=3)
        self.initial_position = bp_domain[np.array(bp_vals).argmax()]
        self.current_position = self.initial_position
        self.weight = 1.0
        logger.debug("Creating BraggPeak...\nPrimary max position: %f\n"
                     "Calculated spline:\n%s"
                     % (self.initial_position, self.spline))

    # todo: add more class methods like __len__, __setitem__
    # https://docs.python.org/3/reference/datamodel.html
    def __str__(self):
        return str("{0} with position: {1} and weight: {2}".format(
                   self.__class__.__name__, self.position, self.weight))

    def __repr__(self):
        return repr(self.spline)

    def __getitem__(self, point):
        """Returns value for given x axis point"""
        return self.spline(point)

    @property
    def position(self):
        return self.current_position

    @position.setter
    def position(self, new_position):
        self.current_position = new_position

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, new_weight):
        if 0.0 <= new_weight and new_weight <= 1.0:
            self._weight = new_weight
        else:
            raise ValueError("Weight should be from 0.0 to 1.0")

    def evaluate(self, x_arr):
        """Evaluate for given domain"""
        return self.weight * self.spline(x_arr + self.initial_position - self.current_position)

    def get_spread_idx(self, x_arr, val):
        """
        For single peak this should find closest value to given
        on the left and right side of BraggPeak.

        This functions splits search domain in two parts to ensure
        two different points from left and right side of the peak.

        :param x_arr - search domain
        :param val - desired value
        """
        val_arr = self.evaluate(x_arr)
        if val > val_arr.max():
            raise ValueError('Desired values cannot be greater than max in BraggPeak!')
        merge_idx = val_arr.argmax()
        left = val_arr[:merge_idx]
        right = val_arr[merge_idx:]
        idx_left = (np.abs(left - val)).argmin()
        idx_right = (np.abs(right - val)).argmin()
        return idx_left, merge_idx + idx_right

    def spread_at_90(self, x_arr):
        ll, rr = self.get_spread_idx(x_arr, 0.9)
        return x_arr[ll], x_arr[rr]

    def range(self, x_arr):
        return self.spread_at_90(x_arr)[1]

    def fwhm(self, x_arr):
        """Full width af half-maximum"""
        ll, rr = self.get_spread_idx(x_arr, 0.5)
        return x_arr[ll], x_arr[rr]

    def modulation(self, x_arr, val):
        """Distance from left to right for given threshold val"""
        pass


if __name__ == '__main__':
    from os.path import join
    import pandas as pd

    with open(join("..", "bp.csv"), 'r') as bp_file:
        data = pd.read_csv(bp_file, sep=';')

    x_peak = data[data.columns[0]].as_matrix()
    y_peak = data[data.columns[1]].as_matrix()

    a = BraggPeak(x_peak, y_peak)

    a.weight = .95
    a.position = 12.5

    test_domain = np.arange(0, 30, .1)
    kle = a.evaluate(test_domain)
    plt.plot(test_domain, kle)
    plt.show()

    print(a.get_spread_idx(x_peak, 0.9))
