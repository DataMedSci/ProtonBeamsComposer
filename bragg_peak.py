import logging

logger = logging.getLogger(__name__)


class BraggPeak:
    def __init__(self, bragg_peak, primary_position, shifted_position, height):
        # todo: merge those?
        self.peak = bragg_peak
        self.primary_pos = primary_position
        self.shifted_pos = shifted_position
        self.height = height
        self.bp_data = {"bp": self.peak,
                        "properties":
                            {"primary_position": self.primary_pos,
                             "shifted_position": self.shifted_pos,
                             "height": self.height
                             }
                        }
        logger.debug("Creating BraggPeak - primary position %.2f, shifted position %.2f height %.2f\nfrom\n%s" % (
            primary_position, shifted_position, height, bragg_peak))

    # todo: add more class methods like __len__, __setitem__
    def __repr__(self):
        return repr(self.bp_data)

    def __getitem__(self, item):
        return self.bp_data[item]


if __name__ == '__main__':
    # small testing area
    kle = BraggPeak([], 10, 20, 30)
    print(kle)
    print(kle['properties']['height'])
