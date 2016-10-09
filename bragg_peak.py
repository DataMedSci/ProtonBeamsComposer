import logging

logger = logging.getLogger(__name__)


class BraggPeak:
    def __init__(self, bragg_peak, position, height):
        self.peak = bragg_peak
        self.position = position
        self.height = height
        logger.debug("Creating BraggPeak - position %.2f, height %.2f\nfrom\n%s" % (position, height, bragg_peak))

    # todo: this should be "class return", not separate method
    def data_dict(self):
        logger.log(level=5, msg="Calling for BraggPeak data dict")
        return {"bp": self.peak, "properties": {"position": self.position, "height": self.height}}
