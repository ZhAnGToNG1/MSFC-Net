from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

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
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)


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

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections



