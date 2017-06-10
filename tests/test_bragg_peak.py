import unittest

from pbc.bragg_peak import BraggPeak


class TestBraggPeakInit(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(TypeError) as e:
            BraggPeak()
        assert "__init__()" in str(e.exception)

    def test_single_empty_list(self):
        with self.assertRaises(TypeError) as e:
            BraggPeak([])
        assert "__init__()" in str(e.exception)

    def test_double_empty_list(self):
        with self.assertRaises(Exception) as e:
            BraggPeak([], [])
        assert "failed" in str(e.exception)

    def test_unequal_lists(self):
        with self.assertRaises(ValueError) as e:
            BraggPeak([1], [])
        assert "Domain and values have different lengths!"in str(e.exception)

        with self.assertRaises(ValueError) as e:
            BraggPeak([], [1])
        assert "Domain and values have different lengths!"in str(e.exception)

    def test_too_short_lists(self):
        with self.assertRaises(Exception) as e:
            BraggPeak([1], [1])
        assert "failed" in str(e.exception)

        with self.assertRaises(Exception) as e:
            BraggPeak([1, 2], [1, 1])
        assert "failed" in str(e.exception)

        with self.assertRaises(Exception) as e:
            BraggPeak([1, 2, 3], [1, 1, 1])
        assert "failed" in str(e.exception)

    def test_min_len_lists(self):
        a = BraggPeak([1, 2, 3, 4], [1, 1, 1, 1])
        assert isinstance(a, BraggPeak)
        self.assertAlmostEqual(a.position, 1.)
        self.assertAlmostEqual(a.weight, 1.)
        self.assertAlmostEqual(a[2], 1.)  # domain
        self.assertAlmostEqual(a[2.3], 1.)  # between given points
        self.assertAlmostEqual(a[-1.2], 1.)  # before
        self.assertAlmostEqual(a[5.2], 1.)  # after
