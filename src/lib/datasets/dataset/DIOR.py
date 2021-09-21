import pycocotools.coco as coco
import numpy as np

import torch.utils.data as data

class DIOR(data.Dataset):
    num_classes = 20
    default_resolution = [512, 512]
    mean = np.array([0.393, 0.403, 0.365],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.157, 0.145, 0.141],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(DIOR, self).__init__()

        if split == 'val':
             self.img_dir = '/data/DIOR/images/val_dior'
             self.annot_path = '/data/DIOR/val.json'
        else:
            self.img_dir = '/data/DIOR/images/train'
            self.annot_path= '/data/DIOR/train/train.json'

        self.max_objs = 512
        self.class_name = [
            '__background__','golffield','Expressway-toll-station', 'vehicle', 'trainstation', 'chimney',
            'storagetank', 'ship', 'harbor',
            'airplane', 'groundtrackfield',  'tenniscourt', 'dam', 'basketballcourt',
            'Expressway-Service-area', 'stadium','airport','baseballfield',
            'bridge','windmill','overpass']
        self._valid_ids = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 , 16, 17, 18, 19, 20]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}


        self.split = split
        self.opt = opt
        print('==> initializing DIOR {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples



