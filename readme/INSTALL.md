# Installation

The code was implemented on Ubuntu 16.04, with Python 3.6, PyTorch 1.7.0, torchvision 0.8.0 and cuda 11.0.

(Note, due to hardware difference, you can choose the appropriate PyTorch (> 1.1.0) and cuda)


## Step 1.


    conda create --name msfcnet python=3.6

    Activate the envs.
   
    conda activate msfcnet

    
## Step 2.
    Install pytorch:

    conda install pytorch = x.x.x torchvision -c pytorch

    or directly downloads torch-1.7.1+cu110  and torchvision 0.8.0 from https://download.pytorch.org/whl/torch_stable.html
    
    and use pip install.
    
## Step 3.
    Install cocoapi:
    
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
    cd $COCOAPI/PythonAPI
    make
    python setup.py install --user
    
## Step 4.
    Install necessary dependencies:
    
    requirements:
    matplotlib
    GCC
    opencv-python
    numba
    tqdm
    scipy
    ...
    
## Step 5.
  ~~~
  Complie deformable convolution:

  Note: the `MSFC-Net_ROOT/src/lib/models/networks/DCNv2/` in the project only can be applied on cuda11.0 and RTX 3090. (from https://github.com/MatthewHowe/DCNv2)
 
  if cuda<11.0:
  
  Please downlaod DCNv2 from https://github.com/CharlesShang/DCNv2
  
  then,
  
  cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
  ./make.sh
  
  then,
  
  modify the import file 'from models.networks.DCNv2.DCN.dcn_v2 import dcn_v2_conv' in `MSFC-Net_ROOT/src/lib/models/n_utils/branch_conv.py` according to new version.
  
  ~~~
## Step 6.
    [Optional]Compile NMS:
    In our project, NMS has been compiled. If arise some problem, you can recompile it

    cd $CenterNet_ROOT/src/lib/external
    make


  
  
    
    
    
    

 
