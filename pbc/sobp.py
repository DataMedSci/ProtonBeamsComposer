import logging
from copy import copy

import numpy as np

from pbc.bragg_peak import BraggPeak

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SOBP(object):
    def __init__(self, bragg_peaks, param_list=None, def_domain=None):
        """
        If param_list=None assume bp is a list of BraggPeak instances.

        If param_list is given assume bp is a single BraggPeak and param_list
            contains positions of it with weights. This list should be formatted
            like this:
                param_list = [[pos1, wei1], [pos2, wei2], ...]
            e.g.
                param_list = [[10, 0.8], [12, 0.75], [15.5, 0.3]]
            The above will generate a SOBP with 3 peaks with positions and weights
            from param_list.

        def_domain is optional and should be a list: [start, stop, step]
            if such list is given - it is passed to numpy.arange(start, stop, step)
            to create default domain used in functions when no other is specified
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
        if def_domain:
            try:
                self._def_domain = np.arange(def_domain[0], def_domain[1], def_domain[2])
                logger.info("Using defined default domain: \n\tStart: {0} Stop: {1} Step: {2}"
                            .format(def_domain[0], def_domain[1], def_domain[2]))
            except TypeError:
                self._def_domain = None
                logger.warning("Given default domain params are invalid! No default domain specified.")
        else:
            self._def_domain = None

    def __repr__(self):
        return "SOBP({0})".format(self.positions())

    def __str__(self):
        return "SOBP Object consisting of peaks with positions:\n\t{0}\nDefault domain:\n\t{1}"\
            .format(self.positions(), self.def_domain)

    def _has_defined_domain(self, dom):
        """
        Helper function for specifying domain used in calculations.

        - return *dom* if not None else
        - return default domain if specified else
        - raise ValueError
        """
        if dom is not None:
            return dom
        elif dom is None and self.def_domain is not None:
            return self.def_domain
        else:
            raise ValueError("No domain specified (argument/default)!")

    def _section_bounds_idx(self, domain=None, val=0.9):
        """
        Helper function.

        Finds closest value to given on the far left and far right side of SOBP.
        Returns two indexes of found values.

        In others words, it searches for the longest section satisfying both:
            val_arr[start_idx] > :val: and val_arr[end_idx] > :val:
        where val_arr = self.overall_sum(domain) and domain is specified by user
        or the default domain is used (if specified).

        This functions splits search domain in parts to ensure
        two different points from left and right side of the peak.

        :param domain - search domain, if none given use default domain
        :param val - threshold value on both left and right side
        """
        domain = self._has_defined_domain(domain)
        val_arr = self.overall_sum(domain)
        if val > val_arr.max():
            raise ValueError('Desired values cannot be greater than max in SOBP, which is %s!' % val_arr.max())
        tmp_idx = []
        # iterate over known peak max positions and check if values are satisfying our val criteria
        for peak in self.positions():
            pos = (np.abs(domain - peak)).argmin()
            if val_arr[pos] > val:
                tmp_idx.append(pos)
        # the domain will probably be divided into left, middle, right
        # middle is irrelevant but its length is required to calculate
        # right index as "left domain len + gap len + right domain position"
        gap_between = 0
        if len(tmp_idx) > 1:
            left_merge_idx = min(tmp_idx)
            right_merge_idx = max(tmp_idx)
            gap_between = right_merge_idx - left_merge_idx
            left = val_arr[:left_merge_idx]
            right = val_arr[right_merge_idx:]
        else:
            # default split based on position of max in SOBP
            # to ensure getting 2 different points
            merge_idx = val_arr.argmax()
            left = val_arr[:merge_idx]
            right = val_arr[merge_idx:]
        # find idx of desired val in calculated partitions
        idx_left = self._argmin(left, val)
        idx_right = self._argmin(right, val)
        return idx_left, len(left) + gap_between + idx_right

    @staticmethod
    def _argmin(array, val):
        """
        Find index of closest element in array preserving condition: array[idx] >= val
        """
        dist = 99999
        position = None
        for i, elem in enumerate(array):
            if np.abs(elem - val) < dist and elem >= val:
                dist = np.abs(elem - val)
                position = i
        if position:
            return position
        else:
            raise ValueError("Nothing found...")

    @property
    def def_domain(self):
        return self._def_domain

    @def_domain.setter
    def def_domain(self, domain_array):
        self._def_domain = domain_array

    def overall_sum(self, domain=None):
        """
        Calculate sum of peaks included in SOBP using given or default domain.
        If none of the above is specified - raise ValueError.
        Also, divide calculated sum by its max value and return as result.
        """
        domain = self._has_defined_domain(domain)
        tmp_sobp = []
        for peak in self.component_peaks:
            tmp_peak = peak.evaluate(domain)
            tmp_sobp.append(tmp_peak)
        tmp_sum = sum(tmp_sobp)
        tmp_sum /= tmp_sum.max()
        return tmp_sum

    def positions(self):
        """Return list of positions of BraggPeaks contained in SOBP"""
        return [peak.position for peak in self.component_peaks]

    def spread(self, domain=None, val=0.9):
        """Distance from left to right for given threshold val"""
        domain = self._has_defined_domain(domain)
        left_idx, right_idx = self._section_bounds_idx(domain, val)
        return domain[right_idx] - domain[left_idx]

    def range(self, domain=None, val=0.9):
        domain = self._has_defined_domain(domain)
        _, right_idx = self._section_bounds_idx(domain, val)
        return domain[right_idx]


if __name__ == '__main__':
    from os.path import join
    import matplotlib.pyplot as plt
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
    a.position = 24.
    a.weight = .7

    b = BraggPeak(x_peak, y_peak)
    b.position = 20.5
    b.weight = .7

    c = BraggPeak(x_peak, y_peak)
    c.position = 19.5
    c.weight = .15

    d = BraggPeak(x_peak, y_peak)
    d.position = 16.
    d.weight = .20

    e = BraggPeak(x_peak, y_peak)
    e.position = 22.5
    e.weight = .55

    inp_peaks = [b, c, d, e, a]

    start, stop, step = 12, 26, 0.1
    test_sobp = SOBP(inp_peaks)
    print(test_sobp)
    print(test_sobp.positions())

    test_domain = np.arange(start, stop, step)
    test_sobp.def_domain = test_domain

    sobp_vals = test_sobp.overall_sum()
    mx = sobp_vals.max()
    mn = sobp_vals.min()
    print("Max val in sobp: {}".format(mx))
    print("Min val in sobp: {}".format(mn))

    t = 0.75
    ll, rr = test_sobp._section_bounds_idx(val=t)
    print(ll, rr)
    print(test_domain[ll])
    print(test_domain[rr])
    print(sobp_vals[ll], sobp_vals[rr])
    ll = test_domain[ll]
    rr = test_domain[rr]
    plt.plot([ll, ll], [0, mx])
    plt.plot([rr, rr], [0, mx])
    plt.plot([start, stop], [t, t])

    print("Spread: %s" % test_sobp.spread(val=t))
    print("Range: %s" % test_sobp.range(val=t))

    plt.plot(test_domain, sobp_vals, 'o-', label="sum")
    for p in inp_peaks:
        plt.plot(test_domain, p.evaluate(test_domain), label=p.position)
    plt.legend()
    plt.title("Spread: {0:.3f} Range: {1:.3f}".format(test_sobp.spread(val=t), test_sobp.range(val=t)))
    plt.show()
