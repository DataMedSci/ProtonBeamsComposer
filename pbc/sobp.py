import logging
from copy import copy

import numpy as np
import scipy.optimize

from pbc.bragg_peak import BraggPeak
from pbc.helpers import diff_max_from_range_90

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

        self._plateau_domain_for_optimization = None

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

    def section_bounds(self, domain=None, threshold=0.9, threshold_right=0.9):
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
            left_merge_idx = val_arr.argmax()
            right_merge_idx = left_merge_idx
            left = val_arr[:left_merge_idx]
            right = val_arr[right_merge_idx:]
        # find desired val in calculated partitions using interpolation
        proximal = np.interp([threshold], left[::], domain[:left_merge_idx:])
        distal = np.interp([threshold_right], right[::-1], np.flip(domain[right_merge_idx::], 0))
        return proximal[0], distal[0]

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
        return self.modulation(domain, left_threshold=0.5, right_threshold=0.5)

    def range(self, val=0.9, domain=None):
        """Return distal-range of SOBP (at the furthest end)"""
        domain = self._has_defined_domain(domain)
        _, distal = self.section_bounds(domain=domain, threshold=val)
        return distal

    def proximal_range(self, val=0.99, domain=None):
        """Return proximal-range of SOBP"""
        domain = self._has_defined_domain(domain)
        proximal, _ = self.section_bounds(domain=domain, threshold=val)
        return proximal

    def modulation(self, domain=None, left_threshold=0.99, right_threshold=0.9):
        """Calculate modulation using given thresholds"""
        domain = self._has_defined_domain(domain)
        proximal, distal = self.section_bounds(domain=domain, threshold=left_threshold, threshold_right=right_threshold)
        return distal - proximal

    def _flat_plateau_factor_helper(self, target_val=1.0):
        """
        Helper function for optimization - calculate how flat is SOBP plateau
        using square of differences between plateau points and target
        horizontal line (function y = target_val)
        """
        plateau = self.overall_sum(self._plateau_domain_for_optimization)
        # iterate over plateau points and calculate diff from target
        return sum([abs(target_val - pp) for pp in plateau])

    def _optimization_helper(self, data_to_unpack):  # , target_modulation, target_range):
        """
        Helper function for scipy.optimize - should calculate modulation
        and plateau factor (how flat is SOBP plateau between two points:
            - target_range
            - target_range - target_modulation

        :param data_to_unpack: data from scipy.optimize (list of peak weights)
        :return:
        """
        if len(data_to_unpack) != len(self.component_peaks):
            raise ValueError("Length check failed for data to unpack...")
        for idx, peak in enumerate(self.component_peaks):
            peak.weight = data_to_unpack[idx]
        plateau_factor = self._flat_plateau_factor_helper()
        return plateau_factor ** 2

    def optimize_sobp(self, target_modulation, target_range, optimization_options=None):
        """
        Optimise weights of peaks in SOBP object. This function does not
        move peaks from original position, it only manipulates their weights.

        :param target_modulation:
        :param target_range:
        :param optimization_options: option dict passed to scipy.optimization function
        :return:
        """
        options = {
            # 'disp': True,
            'eps': 1e-8,  # If jac is approximated, use this value for the step size.
            'ftol': 1e-12,
            'gtol': 1e-12,  # Gradient norm must be less than gtol before successful termination.
            'maxls': 40,
            'maxfun': 30000,
            'maxiter': 15000,
            'maxcor': 40,
        }

        # if external options are specified - overwrite default
        if optimization_options:
            for option_name, option_value in optimization_options.iteritems():
                options[option_name] = option_value

        initial_weights = []
        bound_list = []
        # weight for BP has to be 0 >= weight >=1 and optimization
        # shifts weight by eps, so to prevent ValueErrors from BraggPeak.weight
        # make the bounds smaller by eps
        lower_bound = 0.0 + options['eps']
        upper_bound = 1.0 - options['eps']

        for peak in self.component_peaks:
            initial_weights.append(peak.weight)
            bound_list.append((lower_bound, upper_bound))

        initial_weights = np.array(initial_weights)

        # cut plateau domain a bit from right to exclude the drop to 0.9 'range'
        cut_from_right = diff_max_from_range_90(self.component_peaks[-1])
        self._plateau_domain_for_optimization = np.arange(start=target_range - target_modulation,
                                                          stop=target_range - cut_from_right,
                                                          step=0.01)

        res = scipy.optimize.minimize(self._optimization_helper,
                                      x0=initial_weights,
                                      # args=(target_modulation, target_range),
                                      bounds=bound_list,
                                      method='L-BFGS-B',
                                      options=options)
        return res

if __name__ == '__main__':
    from os.path import join
    import matplotlib.pyplot as plt
    from pbc.helpers import load_data_from_dump

    x_peak, y_peak = load_data_from_dump(file_name=join("..", "data", "cydos1.dat"), delimiter=' ')

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
    ll, rr = test_sobp.section_bounds(threshold=t, threshold_right=t2)
    print(ll, rr)

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
    plt.title("FWHM: {0:.2f}, range: {1:.2f}, modulation: {2:.2f}".format(tmp_fwhm, tmp_range, tmp_modulation))
    plt.show()
