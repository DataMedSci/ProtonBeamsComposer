import unittest

from pbc.bragg_peak import BraggPeak
from pbc.sobp import SOBP


class TestSOBPInit(unittest.TestCase):
    def setUp(self):
        self.a = BraggPeak([1, 2, 3, 4, 5], [.1, .3, .7, 1., .2])
        self.a.position = 14.
        self.b = BraggPeak([10, 12, 14, 16], [.2, .4, 1., .2])
        self.b.position = 18.

    def test_empty(self):
        with self.assertRaises(TypeError) as e:
            SOBP()
        assert "__init__()" in str(e.exception)

    def test_empty_list(self):
        with self.assertRaises(ValueError) as e:
            SOBP([])
        assert "Unsupported init data." in str(e.exception)

    def test_single_peak_list_no_params(self):
        s = SOBP([self.a])
        assert isinstance(s, SOBP)
        assert len(s.component_peaks) == 1
        assert s.component_peaks[0].position == 14.

    def test_peak_list_no_params(self):
        s = SOBP([self.a, self.b])
        assert isinstance(s, SOBP)
        assert len(s.component_peaks) == 2
        assert repr(s) == "SOBP([14.0, 18.0])"
        self.assertAlmostEqual(s.component_peaks[0].position, 14.)
        self.assertAlmostEqual(s.component_peaks[1].position, 18.)

    def test_invalid_peak_list(self):
        with self.assertRaises(TypeError) as e:
            SOBP([1, 2, 3])
        assert "Peak list should consist of BraggPeak objects!" in str(e.exception)

    def test_invalid_list_and_empty_params(self):
        with self.assertRaises(TypeError) as e:
            SOBP([1, 2, 3], [])
        assert "Peak list should consist of BraggPeak objects!" in str(e.exception)

    def test_single_peak_and_empty_params(self):
        with self.assertRaises(ValueError) as e:
            SOBP(BraggPeak([1, 2, 3, 4], [1, 1, 1, 1]), [])
        assert "Unsupported init data" in str(e.exception)

    def test_single_peak(self):
        with self.assertRaises(ValueError) as e:
            SOBP(BraggPeak([1, 2, 3, 4], [1, 1, 1, 1]))
        assert "Unsupported init data" in str(e.exception)

    def test_invalid_types(self):
        """Some tests for data that should not pass"""
        with self.assertRaises(ValueError) as e:
            SOBP((1, 2, 3, 4), (1, 2, 3, 4))
        assert "Unsupported init data" in str(e.exception)

        with self.assertRaises(ValueError) as e:
            SOBP({})
        assert "Unsupported init data" in str(e.exception)

        with self.assertRaises(ValueError) as e:
            SOBP(1, 1)
        assert "Unsupported init data" in str(e.exception)

    def test_single_peak_with_params(self):
        params = [[10, 0.8], [12, 0.75], [15.5, 0.3]]
        s = SOBP(self.a, params)

        assert isinstance(s, SOBP)
        assert len(s.component_peaks) == 3
        assert repr(s) == "SOBP([10, 12, 15.5])"

        self.assertAlmostEqual(s.component_peaks[0].position, 10.)
        self.assertAlmostEqual(s.component_peaks[1].position, 12.)
        self.assertAlmostEqual(s.component_peaks[2].position, 15.5)

        self.assertAlmostEqual(s.component_peaks[0].weight, 0.8)
        self.assertAlmostEqual(s.component_peaks[1].weight, 0.75)
        self.assertAlmostEqual(s.component_peaks[2].weight, 0.3)
