import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

logger = logging.getLogger(__name__)


class BraggPeak(object):
    def __init__(self, bp_domain, bp_vals):
        """
        BraggPeak is a function created from bp_domain and bp_vals
        using scipy.interpolate module. Whenever a function is called,
        this calculated spline function along with domain specified by user
        are used. bp_domain and bp_vals are lost after initialization.
        """
        if len(bp_domain) != len(bp_vals):
            raise ValueError("Domain and values have different lengths!")
        self.spline = interpolate.InterpolatedUnivariateSpline(bp_domain, bp_vals, ext=3)
        self.initial_position = bp_domain[np.array(bp_vals).argmax()]
        self.current_position = self.initial_position
        self._weight = 1.0
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
        if new_weight < 0.0 or new_weight > 1.0:
            raise ValueError("Weight should be from 0.0 to 1.0")
        else:
            self._weight = new_weight

    def evaluate(self, domain):
        """Calculate BP values for given domain"""
        return self._weight * self.spline(domain + self.initial_position - self.current_position)

    def _calculate_idx_for_given_height_value(self, domain, val=0.8):
        """
        This is a helper function and it returns indices based on given domain.

        For single peak this should find closest value to given
        on the left and right side of BraggPeak.
        This functions splits search domain in two parts to ensure
        two different points from left and right side of the peak.

        :param domain - search domain
        :param val - percentage value where the width is calculated at
        """
        val_arr = self.evaluate(domain=domain)
        if val > val_arr.max():
            raise ValueError('Desired values cannot be greater than max in BraggPeak!')
        merge_idx = val_arr.argmax()
        left = val_arr[:merge_idx]
        right = val_arr[merge_idx:]
        idx_left = (np.abs(left - val)).argmin()
        idx_right = (np.abs(right - val)).argmin()
        return idx_left, merge_idx + idx_right

    def range(self, domain, val=0.9):
        """Return range at given value on the dropping/further side of BP"""
        _, idx = self._calculate_idx_for_given_height_value(domain=domain, val=val)
        return domain[idx]

    def width_at(self, domain, val=0.8):
        left_idx, right_idx = self._calculate_idx_for_given_height_value(domain=domain, val=val)
        return domain[right_idx] - domain[left_idx]


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
    print(a.range(test_domain))
    plt.plot(test_domain, kle)
    plt.show()
