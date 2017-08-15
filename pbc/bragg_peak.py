import logging
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

logger = logging.getLogger(__name__)


class BraggPeak(object):
    def __init__(self, bp_domain, bp_vals):
        """
        BraggPeak object is a function created from bp_domain and bp_vals
        using scipy.interpolate module. Whenever a class function is called,
        this calculated spline function along with domain specified by user
        is used.

        :param bp_domain: used as a domain in interpolation, lost after class initialization
        :param bp_vals: used as values in interpolation, lost after class initialization
        """
        if len(bp_domain) != len(bp_vals):
            raise ValueError("Domain and values have different lengths!")
        self.spline = interpolate.InterpolatedUnivariateSpline(bp_domain, bp_vals, ext=3)
        self.initial_position = bp_domain[np.array(bp_vals).argmax()]
        self.current_position = self.initial_position
        self._weight = 1.0
        logger.debug("Creating BraggPeak...\n\tPrimary max position: {0}"
                     "\n\tPeak range: {1}".format(self.initial_position, self.range()))

    def __repr__(self):
        return str("{0} with position: {1} and weight: {2}".format(
                   self.__class__.__name__, self.position, self.weight))

    def __str__(self):
        return str(self.spline)

    def __getitem__(self, point):
        """Returns value for given point on X-axis"""
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
        try:
            idx_left = (np.abs(left - val)).argmin()
        except ValueError:
            idx_left = None
        idx_right = (np.abs(right - val)).argmin()
        return idx_left, merge_idx + idx_right

    def range(self, val=0.90, precision=0.001):
        """Return range at given value on the dropping/further side of BP"""
        pos = self.position
        tmp_dom = np.arange(pos, pos + 2, precision)
        peak_cp = copy(self)
        peak_cp.weight = 1.0
        ran = np.interp([val], peak_cp.evaluate(tmp_dom)[::-1], tmp_dom[::-1])
        return ran[0]

    def proximal_range(self, val=0.990, precision=0.001):
        pos = self.position
        tmp_dom = np.arange(pos - 2, pos, precision)
        peak_cp = copy(self)
        peak_cp.weight = 1.0
        proximal = np.interp([val], peak_cp.evaluate(tmp_dom), tmp_dom)
        return proximal[0]

    def width_at(self, val=0.80, precision=0.001):
        distal = self.range(val=val, precision=precision)
        proximal = self.proximal_range(val=val, precision=precision)
        return distal - proximal


if __name__ == '__main__':
    from os.path import join
    from beprof import profile
    from pbc.helpers import load_data_from_dump

    x_peak, y_peak = load_data_from_dump(file_name=join('..', 'data', 'cydos1.dat'), delimiter=' ')
    # x_peak, y_peak = load_data_from_dump(file_name=join('..', 'data', '3500.dat'), delimiter=' ')

    y_peak /= y_peak.max()

    a = BraggPeak(x_peak, y_peak)

    yy = np.vstack((x_peak, y_peak)).T
    prof = profile.Profile(yy)

    print("left 99% bef", prof.x_at_y(0.99, reverse=False))
    print("left 99% pbc", a.proximal_range(val=0.99))
    print("right 90% bef", prof.x_at_y(0.90, reverse=True))
    print("right 90% pbc", a.range(0.90))

    print("wid new", a.width_at(val=0.80))

    # they should cover each other
    # plt.plot(prof.x, prof.y, 'r')
    plt.plot(x_peak, y_peak, 'b')
    plt.show()
