from copy import copy

import numpy as np


def calculate_number_of_peaks_gottshalk_80_rule(peak_to_measure, domain, spread):
    """
    Calculate number of peaks optimal for SOBP optimization
    using Gotshalck 80% rule.
    """
    temp_peak = copy(peak_to_measure)
    temp_peak.weight = 1.0
    width = temp_peak.width_at(domain=domain, val=0.8)
    n_of_optimal_peaks = int(np.ceil(spread // width))
    return n_of_optimal_peaks + 1


def argmin_with_condition(array, val):
    """
    Helper function find index of closest element in array
    preserving condition: array[idx] >= val
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
        # Nothing found, return zero idx as numpy does in such situations
        return 0
