import os
from opts import opts
from logger import Logger
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from tqdm import tqdm
id_2_category = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout',
                 'harbor', 'swimming-pool', 'helicopter','container-crane']

def test(opt):
    # load model
    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Logger(opt)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    # load images
    ext = '.jpg' if opt.dataset == 'DIOR' else '.png'
    test_image_dir = opt.test_dir
    res_save_dir = os.path.join(opt.dota_results_dir,'detection-results')
    if not os.path.exists(res_save_dir):
        os.mkdir(res_save_dir)

    # test
    for image in tqdm(os.listdir(test_image_dir)):
        img_path = os.path.join(test_image_dir, image)
        ret = detector.run(img_path)
        img_name = image.split(ext)[0] + '.txt'
        res_pre_img = os.path.join(res_save_dir, img_name)
        results = ret['results']
        with open(res_pre_img, 'w') as fp:
            for cls_id in results:
                if len(results[cls_id]) == 0:
                    continue
                category = id_2_category[cls_id - 1]
                for obj in results[cls_id]:
                    x1, y1, x2, y2, score = obj
                    fp.write("%s %.2f %.2f %.2f %.2f %.2f\n" % (category, score, x1, y1, x2, y2))

if __name__ == '__main__':
    opt = opts().parse()
    test(opt)
