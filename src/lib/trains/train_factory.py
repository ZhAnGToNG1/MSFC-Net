from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .train_multi_scale import CtdetTrainer


train_factory = {
  'ctdet':CtdetTrainer,
}
