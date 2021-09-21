from .sample.ctdet_multiscale import CTDetDataset

from .dataset.DOTA import DOTA
from .dataset.DIOR import DIOR


dataset_factory = {
    'DOTA':DOTA,
    'DIOR':DIOR,
}
_sample_factory = {
  'ctdet': CTDetDataset,
}


def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        pass

    return Dataset
