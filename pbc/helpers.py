from os import mkdir, listdir, getcwd
from os.path import join
import datetime

import numpy as np


def calculate_number_of_peaks_gottschalk_80_rule(peak_to_measure, spread):
    """
    Calculate number of peaks optimal for SOBP optimization
    on given spread using Gottschalk 80% rule.
    """
    width = peak_to_measure.width_at(val=0.80)
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


def load_data_from_dump(file_name, delimiter=';'):
    with open(file=file_name, mode='r') as dump_file:
        x, y = np.loadtxt(dump_file, delimiter=delimiter, usecols=(0, 1), unpack=True)
    return x, y


def diff_max_from_range_90(peak):
    pos = peak.position
    ran = peak.range(val=0.900)
    return ran - pos


def diff_max_from_left_99(peak):
    pos = peak.position
    left_99 = peak.proximal_range(val=0.990)
    return abs(pos - left_99)


def make_precise_end_calculations(sobp_object):
    precise_dom = np.arange(-2, 30, 0.0001)
    ll, rr = sobp_object.section_bounds(domain=precise_dom, threshold=0.990, threshold_right=0.900)
    return ll, rr


def create_output_dir(dir_name):
    now = datetime.datetime.now()
    time_marker = "{0}_{1}_{2}".format(now.day, now.month, now.year)
    folder_name = str(dir_name + '_' + time_marker)

    if 'output' not in listdir(getcwd()):
        mkdir('output')

    if folder_name in listdir(join(getcwd(), 'output')):
        additional_number = 1
        while folder_name + '_' + str(additional_number) in listdir(join(getcwd(), 'output')):
            additional_number += 1
        folder_name += '_' + str(additional_number)

    path = join('output', folder_name)
    mkdir(path)

    return path
