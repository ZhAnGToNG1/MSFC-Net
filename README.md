# Multi-Scale Semantic Fusion-Guided Fractal Convolutional Object Detection Network for Optical Remote Sensing Imagery

> The project contains the code for implementing the MSFC-Net algorithm for optiacal remote sensing object detection.
> (Note: At present, only network structure and testing code are avaiable, training code will be provided later)

## Updaets

- (June 9, 2021) We released the test code and network structure. The corresponding network model is also avaiable for downloading.

## Main results

### Object detection on DOTA validation

| Backbone     | ImageSize |     mAP      |   Model  |
|--------------|-----------|--------------|----------|
|VGG-19        |  512x512  |    77.31     |     -    |
|ResNet-50     |  512x512  |    77.65     |     -    |
|ResNet-101    |  512x512  |    77.83     |     -    |
|ResNeXt-101   |  512x512  |    78.07     |     -    | 
|ResNeSt-101   |  512x512  |    80.26     |     [download]()     |
|ResNeSt-101   |  800x800  |    79.21     |     -    |

### Object detection on DIOR test

| Backbone     | ImageSize |     mAP      |   Model  |
|--------------|-----------|--------------|----------|
|ResNeSt-101   |  512x512  |    70.08     |     [download]()     |
|ResNeSt-101   |  800x800  |    70.63     |     -    |

## Installation
Please refer to [INSTALL.md](readme/INSTALL.md) for installation.

## How to use
### Test

For finishing the test, firstly, you only need to download the DOTA validation and DIOR test images, the ground truth have been provided in `MSFC-Net_ROOT/exp/mAP/DOTA(DIOR)/input/ground-truth/`. Secondly, please download above the model networks. Finally, you can test as follow:

For example, when testing on DOTA validation, run: 
(recommended to set CUDA_VISIBLE_DEVICES before running)
~~~
python ctdet --exp_id test_dota --test_dir `your data path` --patch_size 512 --patch_overlap 128 --dataset DOTA --nms --arch msfc_101 --test_scales 1,0.4 --load_model `your model path`
~~~
when testing on DIOR test, run:
~~~
python ctdet --exp_id test_dior --test_dir `your data path` --patch_size 512 --patch_overlap 112 --dataset DIOR --nms --arch msfc_101 --test_scales 1,0.4 --load_model `your model path`
~~~

### Evaluation

After testing the images, the mAP is calculated by the https://github.com/Cartucho/mAP, for convenience, the code have been download into the project `MSFC-Net_ROOT/exp/mAP/DOTA` and `MSFC-Net_ROOT/exp/mAP/DIOR`. 

The above test results would be save in `MSFC-Net_ROOT/exp/mAP/DOTA(DIOR)/input/detection-results/`.

For calculating the mAP, run:
~~~
python main.py -na
~~~
 

