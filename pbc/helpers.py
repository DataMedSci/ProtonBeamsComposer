from copy import copy

import numpy as np


def calculate_number_of_peaks_gottschalk_80_rule(peak_to_measure, domain, spread):
    """
    Calculate number of peaks optimal for SOBP optimization
    on given spread using Gottschalk 80% rule.
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


def dump_data_to_file(domain, values, file_name):
    temp = np.column_stack((domain, values))
    with open(file=file_name, mode='wb') as dump_file:
        np.savetxt(dump_file, temp, delimiter=";", fmt='%.18f', newline='\n')


def load_data_from_dump(file_name):
    with open(file=file_name, mode='r') as dump_file:
        x, y = np.loadtxt(dump_file, delimiter=';', usecols=(0, 1), unpack=True)
    return x, y


def diff_max_from_range_90(peak):
    pos = peak.position
    tmp_dom = np.arange(pos, pos + 2, 0.0001)
    ran = peak.range(tmp_dom, val=0.900)
    return ran - pos


def diff_max_from_left_99(peak):
    pos = peak.position
    tmp_dom = np.arange(pos - 1, pos + 1, 0.0001)
    left_99_idx = peak._calculate_idx_for_given_height_value(tmp_dom, 0.990)[0]
    left_99_val = tmp_dom[left_99_idx]
    return abs(pos - left_99_val)


def make_precise_end_calculations(sobp_object):
    prec_dom = np.arange(-2, 30, 0.0001)
    ll, rr = sobp_object._section_bounds_idx(domain=prec_dom, threshold=0.99, threshold_right=0.90)
    return prec_dom[ll], prec_dom[rr]
