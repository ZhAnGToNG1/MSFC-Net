import pycocotools.coco as coco
import numpy as np
import torch.utils.data as data


class DOTA(data.Dataset):
    num_classes = 16
    default_resolution = [512, 512]
    mean = np.array([0.339, 0.360, 0.358],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.181, 0.185, 0.192],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(DOTA, self).__init__()

        if split == 'val':
            self.img_dir = '/data/DOTA_h/trainval/image'
            self.annot_path = '/data/DOTA_h/trainval/trainval.json'
        else:
            self.img_dir = '/data/DOTA_h/images/train'
            self.annot_path = '/data/DOTA_h/train.json'

        self.max_objs = 512
        self.class_name = [
            'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter','container-crane']
        self._valid_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}


        self.split = split
        self.opt = opt


        print('==> initializing DOTA {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples
