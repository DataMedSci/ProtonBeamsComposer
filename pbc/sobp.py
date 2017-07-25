import logging
from copy import copy

import numpy as np
import scipy.optimize

from pbc.bragg_peak import BraggPeak
from pbc.helpers import argmin_with_condition, dump_data_to_file, load_data_from_dump

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
            raise ValueError("Unsupported init data.")
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

    def _section_bounds_idx(self, domain=None, threshold=0.9, threshold_right=0.9):
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

        :param domain: search domain, if none given use default domain
        :param threshold: threshold value on the left side
        :param threshold_right: additional parameter for threshold value on the right side
        """
        domain = self._has_defined_domain(domain)
        val_arr = self.overall_sum(domain=domain, rescale_to_one=False)
        if threshold > val_arr.max() or threshold_right > val_arr.max():
            val_arr = self.overall_sum(domain=domain, rescale_to_one=True)
            logger.warning("Division by max value in SOBP occurred!")
            if threshold > val_arr.max() or threshold_right > val_arr.max():
                raise ValueError("Desired values cannot be greater than max in SOBP, which is %s!" % val_arr.max())
        tmp_idx_left = []
        tmp_idx_right = []
        # iterate over known peak max positions and check if values are satisfying our val criteria
        for peak in self.positions():
            pos = (np.abs(domain - peak)).argmin()
            if val_arr[pos] >= threshold:
                tmp_idx_left.append(pos)
            if val_arr[pos] >= threshold_right:
                tmp_idx_right.append(pos)
        # the domain will probably be divided into left, middle, right
        # middle is irrelevant but its length is required to calculate
        # right index as "left domain len + gap len + right domain position"
        gap_between = 0
        if len(tmp_idx_left) > 1 and len(tmp_idx_right) > 1:
            left_merge_idx = min(tmp_idx_left)
            right_merge_idx = max(tmp_idx_right)
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
        idx_left = argmin_with_condition(left, threshold)
        idx_right = argmin_with_condition(right, threshold_right)
        return idx_left, len(left) + gap_between + idx_right

    @property
    def def_domain(self):
        return self._def_domain

    @def_domain.setter
    def def_domain(self, domain_array):
        self._def_domain = domain_array

    def overall_sum(self, domain=None, rescale_to_one=False):
        """
        Calculate sum of peaks included in SOBP using given or default domain.
        If none of the above is specified - raise ValueError.
        """
        domain = self._has_defined_domain(domain)
        tmp_sobp = []
        for peak in self.component_peaks:
            tmp_peak = peak.evaluate(domain)
            tmp_sobp.append(tmp_peak)
        tmp_sum = sum(tmp_sobp)
        if rescale_to_one:
            tmp_sum /= tmp_sum.max()
        return tmp_sum

    def positions(self):
        """Return list of positions of BraggPeaks contained in SOBP"""
        return [peak.position for peak in self.component_peaks]

    def fwhm(self, domain=None):
        """Full width at half maximum"""
        return self.modulation(domain, left_threshold=0.5)

    def range(self, val=0.9, domain=None):
        """Return range of SOBP using given domain at the furthest end"""
        domain = self._has_defined_domain(domain)
        _, right_idx = self._section_bounds_idx(domain, val)
        return domain[right_idx]

    def modulation(self, domain=None, left_threshold=0.99, right_threshold=0.9):
        """Calculate modulation using given thresholds"""
        domain = self._has_defined_domain(domain)
        left_idx, right_idx = self._section_bounds_idx(domain, left_threshold, right_threshold)
        return domain[right_idx] - domain[left_idx]

    def _flat_plateau_factor_helper(self, spread, end, target_val=1.0):
        """Helper function for optimization - calculate how flat is SOBP plateau"""
        plateau_domain = np.arange(end - spread, end, 0.1)
        plateau = self.overall_sum(plateau_domain)
        # iterate over plateau points and calculate diff from target
        return sum([abs(pp - target_val) for pp in plateau])

    def _optimization_helper(self, data_to_unpack, target_modulation, target_range):
        """
        Helper function for scipy.optimize - should calculate modulation
        and plateau factor (how flat is SOBP plateau between two points:
            - target_range
            - target_range - target_modulation

        :param data_to_unpack: data from scipy.optimize
        :param target_modulation:
        :param target_range: if not given, use the furthest peak position in SOBP
        :return:
        """
        if len(data_to_unpack) != len(self.component_peaks):
            raise ValueError("Length check failed for data to unpack...")
        for idx, peak in enumerate(self.component_peaks):
            peak.weight = data_to_unpack[idx]
        spread_factor = (self.modulation() - target_modulation)**2
        plateau_factor = self._flat_plateau_factor_helper(spread=target_modulation, end=target_range)
        return spread_factor + plateau_factor**3

    def optimize_modulation(self, target_modulation, target_range=None, optimization_options=None):
        """
        Optimise weights of peaks in SOBP object. This function does not
        move peaks from original position, it only manipulates their weights.

        :param target_modulation:
        :param target_range:
        :param optimization_options: option dict passed to scipy.optimization function
        :return:
        """
        if optimization_options:
            options = optimization_options
        else:
            options = {'disp': True, 'eps': 1e-8, 'ftol': 1e-20, 'gtol': 1e-20,  'maxls': 40}

        initial_weights = []
        bound_list = []
        furthest_range = 0.0
        lower_bound = 0.0 + options['eps']
        upper_bound = 1.0 - options['eps']
        for peak in self.component_peaks:
            initial_weights.append(peak.weight)
            bound_list.append((lower_bound, upper_bound))
            if peak.position > furthest_range:
                furthest_range = peak.position
        if target_range is None:
            target_range = furthest_range
        initial_weights = np.array(initial_weights)
        res = scipy.optimize.minimize(self._optimization_helper,
                                      x0=initial_weights,
                                      args=(target_modulation, target_range),
                                      bounds=bound_list,
                                      method='L-BFGS-B',
                                      options=options)
        return res

if __name__ == '__main__':
    from os.path import join
    import matplotlib.pyplot as plt
    import pandas as pd

    with open(join("..", "data", "bp.csv"), 'r') as bp_file:
        data = pd.read_csv(bp_file, sep=';')

    x_peak = data[data.columns[0]].as_matrix()
    y_peak = data[data.columns[1]].as_matrix()

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

    start, stop, step = 0, 26, 0.1
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
    t2 = 0.6
    ll, rr = test_sobp._section_bounds_idx(threshold=t, threshold_right=t2)
    print(ll, rr)
    print(test_domain[ll])
    print(test_domain[rr])
    print(sobp_vals[ll], sobp_vals[rr])
    ll = test_domain[ll]
    rr = test_domain[rr]
    plt.plot([ll, ll], [0, mx], color='yellow')
    plt.plot([rr, rr], [0, mx], color='orange')
    plt.plot([start, stop], [t, t], color='yellow', label=str(t) + "; left (val=%s)" % ll)
    plt.plot([start, stop], [t2, t2], color='orange', label=str(t2) + "; right (val=%s)" % rr)

    tmp_fwhm = test_sobp.fwhm()
    tmp_range = test_sobp.range(val=t)
    tmp_modulation = test_sobp.modulation(left_threshold=t, right_threshold=t2)
    print("FWHM: %s [mm]" % tmp_fwhm)
    print("Range: %s [mm]" % tmp_range)
    print("Modulation: %s [mm]" % tmp_modulation)

    plt.plot(test_domain, sobp_vals, 'o-', label="sum", color="red")
    for p in inp_peaks:
        plt.plot(test_domain, p.evaluate(test_domain), label=p.position)
    plt.legend()
    plt.title("FWHM: {0:.2f}, range: {1:.2f}, modulation: {2:.2f} (l_threshold:{3:.2f},r_threshold:{4:.2f})"
              .format(tmp_fwhm, tmp_range, tmp_modulation, t, t2))
    # plt.show()

    dump_data_to_file(test_domain, sobp_vals, 'test.csv')
