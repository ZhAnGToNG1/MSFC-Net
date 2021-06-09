from __future__ import absolute_import
from __future__ import division
from __future__ import print_function




from .detect_multi_scale import CtdetDetector


detector_factory = {
  'ctdet': CtdetDetector,
}