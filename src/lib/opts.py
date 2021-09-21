from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from datasets.dataset_factory import get_dataset


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('task' , default='ctdet',
                                 help='ctdet')
        self.parser.add_argument('--dataset',default='nwpu_vhr',
                                 help='nwpu_vhr')
        self.parser.add_argument('--exp_id',default='default')
        self.parser.add_argument('--test',action='store_true')
        self.parser.add_argument('--debug',type = int,default=0,
                                 help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # useful when lunching training with ipython notebook
                                  '4: save all visualizations to disk')
        self.parser.add_argument('--demo',default='',
                                 help='path to image/')
        self.parser.add_argument('--test_dir', default='',
                                 help='path to test image/')
        self.parser.add_argument('--load_model',default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')
        self.parser.add_argument('--gpus',default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])


        self.parser.add_argument('--arch',default='resnet_101',
                                 help='model architecture.')
        self.parser.add_argument('--head_conv',type = int, default=-1,
                                 help='conv layer channels for output head')
        self.parser.add_argument('--down_ratio',type = int,default=4,
                                 help='output stride.')
        self.parser.add_argument('--input_res', type = int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h' , type= int , default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w' , type = int ,default=-1,
                                 help='input width.')




        self.parser.add_argument('--lr', type = float, default=6.25e-5,
                                 help= 'learing rate for batch size 8')
        self.parser.add_argument('--lr_step', type=str, default='50,90',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--save_step', type=str, default='90,150',
                                 help='the epoch to save.')
        self.parser.add_argument('--num_epochs', type=int, default=140,
                                 help='total training epochs.')

        self.parser.add_argument('--batch_size', type=int, default=8,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=-1,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training')



        #test
        self.parser.add_argument('--test_scales', type=str, default='1',
                                 help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=100,
                                 help='max number of output objects.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        self.parser.add_argument('--patch_size',type =int,default=512,
                                 help = 'patch size for test')
        self.parser.add_argument('--patch_overlap',type =int,default=128,
                                 help=' the overlap of adjacent patches')


        self.parser.add_argument('--not_rand_crop', action='store_true',
                                 help='not use the random crop data augmentation'
                                      'from CornerNet.')
        self.parser.add_argument('--shift', type=float, default=0.1,
                                 help='when not using random crop'
                                      'apply shift augmentation.')
        self.parser.add_argument('--scale', type=float, default=0.4,
                                 help='when not using random crop'
                                      'apply scale augmentation.')
        self.parser.add_argument('--rotate', type=float, default=0,
                                 help='when not using random crop'
                                      'apply rotation augmentation.')
        self.parser.add_argument('--flip', type=float, default=0.5,
                                 help='probability of applying flip augmentation.')



        # loss
        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]

        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.save_step = [int(i) for i in opt.save_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]

        opt.reg_offset = True


        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 64
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
            opt.val_intervals = 5

        if opt.debug > 0:
            opt.num_workers = 0
            opt.batch_size = 1
            opt.gpus = [opt.gpus[0]]
            opt.master_batch_size = -1

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)


        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')

        opt.demo_dir = os.path.join(opt.root_dir, 'exp','demo','vis_results')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        opt.dota_results_dir = os.path.join(opt.root_dir, 'exp','mAP','DOTA','input')
        opt.dior_results_dir = os.path.join(opt.root_dir, 'exp', 'mAP', 'DIOR', 'input')
        print('The output will be saved to ', opt.save_dir)
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes
        print('the number of classes : ',opt.num_classes)


        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task =='ctdet':
            opt.heads = {'hm': dataset.num_classes,
                         'wh': 2}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
        elif opt.task == 'seg':
            opt.heads = {'hm': dataset.num_classes}
        else:
            assert 0, 'task not defined!'
        return opt

    def init(self,args=''):
        opt = self.parse(args)
        dataset = get_dataset(opt.dataset,opt.task)
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt