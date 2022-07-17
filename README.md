# Multi-Scale Semantic Fusion-Guided Fractal Convolutional Object Detection Network for Optical Remote Sensing Imagery

* The project contains the code for implementing the [MSFC-Net](https://ieeexplore.ieee.org/abstract/document/9535169) for optiacal remote sensing object detection.
* This work has been published on IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING
## Main results

### Object detection on DOTA validation

| Backbone     | ImageSize |     mAP      |   Model  |
|--------------|-----------|--------------|----------|
|VGG-19        |  512x512  |    77.31     |     [download](https://pan.baidu.com/s/1dHwfPNC-uc5P-j14_EFOOQ) code:sofk    |
|Detnet-50     |  512x512  |    77.44     |     [download](https://pan.baidu.com/s/1V6K8NkQzFzb-mnAhKNO8Kg) code:q5g6    |
|ResNet-50     |  512x512  |    77.65     |     [download](https://pan.baidu.com/s/10zCULdQTKpP-5N-k_ArW0w) code:k5s5    |
|Cspdarknet-53 |  512x512  |    77.84     |     [download](https://pan.baidu.com/s/1npukFXGkzRD7aMkTUs2VQw) code:sx23    |
|ResNet-101    |  512x512  |    77.83     |     [download](https://pan.baidu.com/s/1GEV3pckYD4Vf8IUQIma2Ew) code:wess    |
|ResNeXt-101   |  512x512  |    78.07     |     [download](https://pan.baidu.com/s/1lKxmV2NFxREBrvLUl0166Q) code:qwer    | 
|ResNeSt-101   |  512x512  |    80.26     |     [download](https://pan.baidu.com/s/1EbL4yCLBmZy5xszWNkN3Gg) code:dota    |


### Object detection on DIOR test

| Backbone     | ImageSize |     mAP      |   Model  |
|--------------|-----------|--------------|----------|
|ResNeSt-101   |  512x512  |    70.08     |     [download](https://pan.baidu.com/s/1igbcB1Y3mdOQpFG--zU_jA) code:dior    |


## Installation
Please refer to [INSTALL.md](readme/INSTALL.md) for installation.

## How to use
### Prepare dataset

inital data structure:
```
data/dota
├── train
│   ├── images
│   └── labelTxt
└── val
    ├── images
    └── labelTxt

```


1. Split the original images
```
python dota_devkit/ImgSplit_multi_process.py
(Manually enter the basepath and outpath)
Then, you should get data in following structure.
```
```
data/dota_split
├── train
│   ├── images
│   └── labelTxt
└──  val
    ├── images
    └── labelTxt
```
2. Create COCO format json
```
python dota_devkit/DOTA2COCO.py
(Manually enter the srcpath and destfile)
```

```
data/dota_split
├── train
│   ├── images
│   ├── labelTxt
│   └── train.json
│
└── val
    ├── images
    ├── labelTxt
    └── val.json
```
(The purpose of splitting val dataset is to find the optimal model, not for testing.)


### Test

For finishing the test, firstly, you only need to download the DOTA validation and DIOR test images, the ground truth have been provided in `MSFC-Net_ROOT/exp/mAP/DOTA(DIOR)/input/ground-truth/`. Secondly, please download above the model networks. Finally, you can test as follow:

For example, when testing on DOTA validation, run: 
(recommended to set CUDA_VISIBLE_DEVICES before running)
~~~
python test.py ctdet --exp_id test_dota --test_dir `your data path` --patch_size 512 --patch_overlap 128 --dataset DOTA --nms --arch msfc_101 --test_scales 1,0.4 --load_model `your model path`
                                                                                                                      --arch msfcvgg_19
                                                                                                                      --arch msfcresnet_101
                                                                                                                      --arch msfscspdarknet_53
                                                                                                                      --arch ...
~~~
when testing on DIOR test, run:
~~~
python test.py ctdet --exp_id test_dior --test_dir `your data path` --patch_size 512 --patch_overlap 112 --dataset DIOR --nms --arch msfc_101 --test_scales 1,0.4 --load_model `your model path`
~~~
### Train

python train.py ctdet --exp_id msfcnet --batch_size 10 --dataset DOTA --arch msfc_101 --num_epoches 100



### Evaluation

After testing the images, the results would be saved in `MSFC-Net_ROOT/exp/mAP/DOTA(DIOR)/input/detection-results/`.

The mAP is calculated by the https://github.com/Cartucho/mAP, for convenience, the code have been downloaded into the project `MSFC-Net_ROOT/exp/mAP/DOTA` and `MSFC-Net_ROOT/exp/mAP/DIOR`. 


For calculating the mAP, run:
~~~
python main.py -na
~~~
 
### Demo
You can follow the below steps to run a quick demo:

you need to download the network model (DOTA or DIOR), then run:
~~~
python demo.py ctdet --demo `MSFC-Net_ROOT/exp/demo/images/` --load_model `your model path` --nms --test_scales 1 --dataset DOTA --arch msfc_101
~~~




Contact: bit_zhangtong@163.com. If you have any questions.

The code is modified based on [CenterNet](https://github.com/xingyizhou/CenterNet).

### Citation
```bash
@ARTICLE{msfc2021zhang,  
 author={Zhang, Tong and Zhuang, Yin and Wang, Guanqun and Dong, Shan and Chen, He and Li, Lianlin},  
 journal={IEEE Transactions on Geoscience and Remote Sensing},   
 title={Multiscale Semantic Fusion-Guided Fractal Convolutional Object Detection Network for Optical Remote Sensing Imagery},   
 year={2022},  
 volume={60},  
 number={},  
 pages={1-20},  
 doi={10.1109/TGRS.2021.3108476}}
```

