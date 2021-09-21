import os
from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'png']

def demo(opt):
    opt.debug = 1
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo,file_name))
    else:
        image_names = [opt.demo]

    for (image_name) in image_names:
        print(image_name)
        ret = detector.run(image_name)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)