import unittest
from os.path import join

import pandas as pd

from pbc.bragg_peak import BraggPeak
from pbc.sobp import SOBP


class TestSOBPInit(unittest.TestCase):
    with open(join("..", "bp.csv"), 'r') as bp_file:
        data = pd.read_csv(bp_file, sep=';')

    x_peak = data[data.columns[0]].as_matrix()
    y_peak = data[data.columns[1]].as_matrix()

    def setUp(self):
        self.a = BraggPeak(self.x_peak, self.y_peak)
        self.a.position = 14.
        self.b = BraggPeak(self.x_peak, self.y_peak)
        self.b.position = 18.

    def test_empty(self):
        with self.assertRaises(TypeError) as e:
            SOBP()
        assert "__init__() missing 1 required positional argument: 'bragg_peaks'" in str(e.exception)

    def test_empty_list(self):
        with self.assertRaises(ValueError) as e:
            SOBP([])
        assert "Unsupported init data." in str(e.exception)

    def test_single_peak_no_params(self):
        s = SOBP([self.a])
        assert isinstance(s, SOBP)
        assert len(s.component_peaks) == 1
        assert s.component_peaks[0].position == 14.

    def test_peak_list_no_params(self):
        s = SOBP([self.a, self.b])
        assert isinstance(s, SOBP)
        assert len(s.component_peaks) == 2
        assert s.component_peaks[0].position == 14.
        assert s.component_peaks[1].position == 18.

    def test_invalid_peak_list(self):
        with self.assertRaises(TypeError) as e:
            s = SOBP([1, 2, 3])
        assert "Peak list should consist of BraggPeak objects!" in str(e.exception)
