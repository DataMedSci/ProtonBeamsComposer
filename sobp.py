import logging
import numpy as np

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
        temp_peak = None
        for p in plist:
            # todo: multiplying by coefficients and shifting(?)
            if temp_peak is not None:
                if np.array_equal(temp_peak.x, p["bp"].x):
                    temp_peak.y += p["bp"].y * p["properties"]["height"]
                    logger.debug(
                        "\t(calc. SOBP) Got BraggPeak with primary position %.2f, shifted position %.2f, height %.2f" % (
                            p["properties"]["primary_position"],
                            p["properties"]["shifted_position"],
                            p["properties"]["height"])
                    )
                else:
                    raise ValueError("Inconsistent domains!")
            else:
                temp_peak = p["bp"]
                temp_peak.y *= p["properties"]["height"]
                logger.debug(
                    "\t(calc. SOBP) Got BraggPeak with primary position %.2f, shifted position %.2f, height %.2f" % (
                        p["properties"]["primary_position"],
                        p["properties"]["shifted_position"],
                        p["properties"]["height"])
                )
        temp_peak.rescale(temp_peak.y.max())
        return temp_peak

    def val(self, x):
        return self.sobp.y_at_x(x)

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
