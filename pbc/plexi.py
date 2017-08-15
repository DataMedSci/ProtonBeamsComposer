"""
This will contain all plexi related operations like generating modulator, range shifter etc.
"""

from scipy.interpolate import InterpolatedUnivariateSpline


class PlexiInterpolator(object):
    def __init__(self, thickness_table, range_table):
        """
        Alternatively max_position table can be given,
        if so - it will work as max_position_to_thickness mapper etc.
        """
        self.range_to_thickness = InterpolatedUnivariateSpline(range_table[::-1], thickness_table[::-1], ext=0)
        self.thickness_to_range = InterpolatedUnivariateSpline(plexi_thickness, plexi_range, ext=0)

    def range_to_thickness(self, val):
        # neither range nor max_position below zero is acceptable
        if val < 0:
            raise ValueError("Impossible (less 0) value given!")
        return self.range_to_thickness(val)

    def thickness_to_range(self, val):
        return self.thickness_to_range(val)


if __name__ == '__main__':
    from os.path import join
    from pbc.helpers import load_data_from_dump

    plexi_thickness, plexi_range = load_data_from_dump(file_name=join('..', 'data', 'plexi_max.dat'), delimiter=';')
    pos, wei = load_data_from_dump(file_name=join('..', 'data', 'example_result.dat'), delimiter=';')

    interp = PlexiInterpolator(plexi_thickness, plexi_range)

    print(interp.range_to_thickness(pos))
    print(interp.thickness_to_range(0.00000))
