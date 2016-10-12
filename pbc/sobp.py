import logging

import numpy as np

import beprof

logger = logging.getLogger(__name__)


class SOBP:
    def __init__(self, peak_list):
        if isinstance(peak_list, list) and len(peak_list) > 0:
            logger.debug("Creating SOBP...")
            self.sobp = self.calc_sobp(peak_list)
        else:
            raise TypeError("List object required!")

    @staticmethod
    def calc_sobp(plist):
        recalculated_peaks = []
        for p in plist:
            primary = p["properties"]["primary_position"]
            shifted = p["properties"]["shifted_position"]
            height = p["properties"]["height"]
            # if primary != shifted ?? some a-b<delta
            if abs(primary - shifted) > 0.0001:
                temp_x = p["bp"].x + (shifted - primary)
                logger.debug("Shifting by %.4f" % (shifted - primary))
            else:
                temp_x = p["bp"].x
                logger.debug("No shifting required.")
            temp_y = p["bp"].y * p["properties"]["height"]
            temp_peak = beprof.profile.Profile(np.array([temp_x, temp_y]).T)
            # normalize?
            temp_peak.rescale(temp_peak.y.max())
            # todo: is it all mandatory?... improve - sum to one Profile
            recalculated_peaks.append(temp_peak)
            logger.debug(
                "\t(calc. SOBP) Got BraggPeak with primary position %.2f, shifted position %.2f, height %.2f" %
                (primary, shifted, height)
            )
        return recalculated_peaks

    def val(self, x):
        temp_val = 0
        for p in self.sobp:
            temp_val += p.y_at_x(x)
        return temp_val

    def calc_on_mesh(self, mesh_array):
        """Calc chi^2 on given mesh"""
        result = 0
        logger.debug("Calc chi^2 on mesh...")
        for m in mesh_array:
            temp = (self.val(m) - 1) ** 2
            logger.debug("\t(Calc chi) Got %.4f on position %.2f" % (self.val(m), m))
            if str(temp) != 'nan':
                result += temp
            else:
                raise ValueError("Got 'nan' instead of number!")
        return result
