import os
import cv2
import numpy as np
from torch.autograd import Variable

def vis_feature(x,max_num = 256,out_path = ''):
    for i in range(0,x.shape[1]):
        if i >=max_num:
            break
        feature = x[0,i,:,:].view(x.shape[-2],x.shape[-1])

        feature = Variable(feature)
        feature = feature.cpu().numpy()

        feature = 1.0 / (1 + np.exp(-1 * feature))

        feature = np.round(feature*255)
        feature  = feature.astype(np.uint8)
        feature_img = cv2.applyColorMap(feature,cv2.COLORMAP_JET)
        #feature_img = cv2.cvtColor(feature_img,cv2.COLOR_BGR2RGB)
        dst_path = os.path.join(out_path, str(i) + '.png')
        cv2.imwrite(dst_path,feature_img)