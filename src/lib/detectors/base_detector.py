import cv2
import numpy as np
import torch

from models.model import create_model, load_model
from utils.debugger import Debugger
import torchvision
class BaseDetector(object):
    def __init__(self,opt):
        if opt.gpus[0]>=0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('start testing...')
        print("Creating model...")
        self.model = create_model(opt.arch,opt.heads,opt.head_conv)
        self.model = load_model(self.model,opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()
        self.mean = np.array(opt.mean , dtype=np.float32).reshape(1,1,3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1,1,3)
        self.max_per_image = 800
        self.num_classes = opt.num_classes
        self.scales = opt.test_scales
        self.opt = opt
        self.pause = True
        if 'DIOR' in opt.dataset:
            print('DIOR dataset Normalize...')
            self.normalization = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize([0.393, 0.403, 0.365],[0.157, 0.145, 0.141])])
        else:
            print('DOTA dataset Normalize...')
            self.normalization = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize([0.339, 0.360, 0.358],[0.181, 0.185, 0.192])])


    def pre_process_simple(self,image):
        height,width = image.shape[0:2]
        c = np.array([height/2.,width/2.],dtype = np.float32)
        s = max(height, width) * 1.0
        inp_image = self.normalization(image)
        images = inp_image.reshape(1, 3, height, width)
        meta = {'c': c, 's': s,
                'out_height': height // self.opt.down_ratio,
                'out_width': width // self.opt.down_ratio}
        return images, meta

    
    def process(self, images, img_szie,return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def merge_outputs(self, detections):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results,image_name):
        raise NotImplementedError

    def run(self,image_or_path_or_tensor,meta = None):
        debugger = Debugger(dataset = self.opt.dataset, ipynb = (self.opt.debug == 3),
                            theme = self.opt.debugger_theme)
        image = cv2.imread(image_or_path_or_tensor)
        detections = []
        for scale in self.scales:
            height, width = image.shape[0:2]
            new_height = int(height * scale)
            new_width = int(width * scale)
            img = cv2.resize(image, (new_width, new_height))

            w_len = self.opt.patch_size
            h_len = self.opt.patch_size
            h_overlap = self.opt.patch_overlap
            w_overlap = self.opt.patch_overlap

            imgH = img.shape[0]
            imgW = img.shape[1]
            if imgH < h_len:
                temp = np.zeros([h_len, imgW, 3], np.uint8)
                temp[0:imgH, :, :] = img
                img = temp
                imgH = h_len
            if imgW < w_len:
                temp = np.zeros([imgH, w_len, 3], np.uint8)
                temp[:, 0:imgW, :] = img
                img = temp
                imgW = w_len

            for hh in range(0, imgH, h_len - h_overlap):
                if imgH - hh - 1 < h_len:
                    hh_ = imgH - h_len
                else:
                    hh_ = hh
                for ww in range(0, imgW, w_len - w_overlap):
                    if imgW - ww - 1 < w_len:
                        ww_ = imgW - w_len
                    else:
                        ww_ = ww


                    src_img = img[hh_:(hh_ + h_len), ww_:(ww_ + w_len), :]
                    src_img, meta1 = self.pre_process_simple(src_img)
                    src_img = src_img.to(self.opt.device)


                    dets ,net_time = self.process(src_img, self.opt.patch_size,return_time=True)
                    dets = self.post_process(dets, meta1, 1)

                    for i in dets:
                        if len(dets[i])!=0:
                            for j in range(len(dets[i])):
                                dets[i][j][0] += ww_
                                dets[i][j][1] += hh_
                                dets[i][j][2] += ww_
                                dets[i][j][3] += hh_

                    for k in range(1, self.num_classes + 1):
                        dets[k][:, :4] /= scale
                    detections.append(dets)

        results = self.merge_outputs(detections)

        if self.opt.debug >= 1:
            image_name = image_or_path_or_tensor.split('/')[-1].split('.')[0]
            self.show_results(debugger, image, results,image_name)
        return {'results': results}









