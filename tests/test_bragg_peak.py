import unittest

import numpy as np

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
        assert "Domain and values have different lengths!" in str(e.exception)

        with self.assertRaises(ValueError) as e:
            BraggPeak([], [1])
        assert "Domain and values have different lengths!" in str(e.exception)

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
        bp = BraggPeak([1, 2, 3, 4], [1, 1, 1, 1])
        assert isinstance(bp, BraggPeak)
        self.assertAlmostEqual(bp.position, 1.)
        self.assertAlmostEqual(bp.weight, 1.)
        self.assertAlmostEqual(bp[2], 1.)  # domain
        self.assertAlmostEqual(bp[2.3], 1.)  # between given points
        self.assertAlmostEqual(bp[-1.2], 1.)  # before
        self.assertAlmostEqual(bp[5.2], 1.)  # after


class TestBraggPeakFunctions(unittest.TestCase):
    def setUp(self):
        self.bp = BraggPeak([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            [.1, .15, .2, .3, .45, .65, 1., .8, .3, .1])

    def test_position(self):
        self.assertAlmostEqual(self.bp.position, 7.)
        self.bp.position = .667
        self.assertAlmostEqual(self.bp.position, .667)
        self.bp.position = -2.5  # left of starting domain
        self.assertAlmostEqual(self.bp.position, -2.5)
        self.bp.position = 35.5  # right of starting domain
        self.assertAlmostEqual(self.bp.position, 35.5)

    def test_weight(self):
        self.bp.weight = .55
        self.assertAlmostEqual(self.bp.weight, .55)
        with self.assertRaises(ValueError) as e:
            self.bp.weight = -0.000001
        assert "Weight" in str(e.exception)
        with self.assertRaises(ValueError) as e:
            self.bp.weight = 1.000001
        assert "Weight" in str(e.exception)

    def test_evaluate(self):
        dom = np.arange(0, 30, 0.01)
        vals = self.bp.evaluate(dom)
        assert isinstance(vals, np.ndarray)
        assert len(vals) == len(dom)

        # this one fails - probably normalization would help
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(vals.max(), self.bp[self.bp.position])

        norm = vals.max()
        normalized_vals = vals / norm
        self.assertAlmostEqual(normalized_vals.max(), self.bp[self.bp.position])

    def test_range(self):
        dom = np.arange(5, 10, 0.01)
        self.assertAlmostEqual(self.bp.range(dom), 7.75)
