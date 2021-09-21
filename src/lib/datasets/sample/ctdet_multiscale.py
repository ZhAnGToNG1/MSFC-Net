import torch.utils.data as data
import numpy as np
import cv2
import os
import math

from utils.image import get_affine_transform, affine_transform
from utils.image import guassian_radius, draw_umich_gaussian, draw_msra_gaussian




class CTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self,box):
        bbox = np.array([box[0],box[1],box[0]+box[2],box[1]+box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self,border,size):
        i = 1
        while size - border//i <= border//i:
            i *=2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids = [img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns),self.max_objs)

        img = cv2.imread(img_path)
        height , width = img.shape[0] , img.shape[1]
        c = np.array([img.shape[1]/2. , img.shape[0]/2.],dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split =='train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6,1.4,0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                #when not using random crop apply shift augmentation.
                sf = self.opt.scale  # 0.4
                cf = self.opt.shift  # 0.1
                c[0] += s * np.clip(np.random.randn()*cf , -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf , -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:,::-1,:]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(c,
                                           s,
                                           0,
                                           (input_w,input_h))

        inp = cv2.warpAffine(img,trans_input,
                             (input_w,input_h),
                             flags = cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2,0,1)


        ret3 = self.gt_transform(c, s, input_w, input_h, self.num_classes, num_objs, anns, 2, 2, flipped, width, inp ,thresh= 1)
        ret4 = self.gt_transform(c, s, input_w, input_h, self.num_classes, num_objs, anns, 4, 4, flipped, width, inp,thresh=1)
        ret5 = self.gt_transform(c, s, input_w, input_h, self.num_classes, num_objs, anns, 4, 16, flipped, width,inp,thresh=1)
        ret6 = self.gt_transform(c, s, input_w, input_h, self.num_classes, num_objs, anns, 16, 4, flipped, width,inp,thresh=1)


        ret = [ret3,ret4,ret5,ret6]
        return ret



    def gt_transform(self,c,s,input_w,input_h,
                     num_classes,num_objs,anns,h_ratio,w_ratio,flipped,width,inp,thresh):

        output_h = input_h // h_ratio
        output_w = input_w // w_ratio

        trans_h = input_h // 4
        trans_w = input_w // 4


        trans_output = get_affine_transform(c, s, 0, [trans_h, trans_w])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])


            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)


            bbox[0] = bbox[0] / (w_ratio/4)
            bbox[1] = bbox[1] / (h_ratio/4)
            bbox[2] = bbox[2] / (w_ratio/4)
            bbox[3] = bbox[3] / (h_ratio/4)


            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > thresh and w > thresh:
                radius = guassian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]

                reg[k] = ct - ct_int
                reg_mask[k] = 1
        ret = {'input': inp, 'hm': hm, 'reg': reg, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

        return ret


























































































































































































































































































































































































