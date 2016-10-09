import logging

import numpy as np

import beprof.profile

import bragg_peak
import sobp

logging.basicConfig(format='%(levelname)-8s->    %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


bp = beprof.profile.Profile([(0, 0), (1, 0), (2, 0.2), (3, 0.5), (4, 1), (5, 0.2), (6, 0), (7, 0)])
# 3.0 - 4.0, every 0.05
mesh = np.arange(3, 4, 0.05)

kle = beprof.profile.Profile(np.array([bp.x, bp.y * 0.5]).T)
double_peak = beprof.profile.Profile(np.array([bp.x, bp.y + kle.y]).T)
# normalize
double_peak.rescale(double_peak.y.max())


logger.info("Creating BraggPeaks for tests...")
w = bragg_peak.BraggPeak(double_peak, 1, 1, 0.5)
q = bragg_peak.BraggPeak(bp, 2, 2, 0.7)

logger.info("Creating SOBP for tests...")
d = sobp.SOBP([w, q, w])

logger.debug("New SOBP:\n%s\nx axis:\n%s\ny axis:\n%s" % (d.sobp, d.sobp.x, d.sobp.y))

logger.info("Chi^2 = %.4f\non mesh:\n%s" % (d.calc_on_mesh(mesh), mesh))
