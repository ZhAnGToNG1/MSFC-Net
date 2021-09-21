import numpy as np
import time
import torch

try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')

from models.decode_multi_scale import ctdet_decode
from utils.post_process import ctdet_post_process

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
    def __init__(self,opt):
        super(CtdetDetector , self).__init__(opt)

    def process(self, images, img_size,return_time = False):
        hms,regs,whs = [],[],[]
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            outputs = self.model(images)
            torch.cuda.synchronize()
            end_time = time.time()
            net_time = end_time - start_time

            for i in range(len(outputs)):
                output = outputs[i]
                hm = output['hm'].sigmoid_()
                wh = output['wh']
                reg = output['reg'] if self.opt.reg_offset else None

                hms.append(hm)
                regs.append(reg)
                whs.append(wh)
            dets = ctdet_decode(img_size,hms, whs ,regs = regs ,K = self.opt.K)


        if return_time:
            return dets,net_time
        else:
            return dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy() , [meta['c']], [meta['s']],
            meta['out_height'],meta['out_width'],self.opt.num_classes)

        for j in range(1,self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j] , dtype=np.float32).reshape(-1,5)
            dets[0][j][:,:4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1 , self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections] , axis = 0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.45, method=2)
        thresh = 0.08
        for j in range(1, self.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
        return results


    def show_results(self, debugger, image, results,image_name):
        debugger.add_img(image, img_id = 'ctdet')
        for j in range(1 , self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4] , img_id='ctdet')
        debugger.save_img(imgId='ctdet', path=self.opt.demo_dir, name=image_name)


