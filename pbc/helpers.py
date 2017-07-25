from copy import copy
from os.path import join

import numpy as np


def calculate_number_of_peaks_gottshalk_80_rule(peak_to_measure, domain, spread):
    """
    Calculate number of peaks optimal for SOBP optimization
    on given spread using Gottshalck 80% rule.
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
