# RHGNet
Pytorch implementations of "Learning Object-Centric Representation via Reverse Hierarchy Guidance"

## Environment

Codes are run under Pytorch 1.10.1. Numpy and opencv-python are needed for reading iamges.

## Model checkpoint 

We provide all checkpoints [here](https://drive.google.com/drive/folders/1SIkil9mclYwFrhxx41FlmK2cNxrWlIMW?usp=sharing). For a quick start, create a directory called 'checkpoint' under the root directory of this repository and place the downloaded model checkpoint into it.

## Data preparation

### CLEVR

Download the tfrecord files of CLEVR (clevr_with_masks_train.tfrecords) from [Multi Object Datasets](https://console.cloud.google.com/storage/browser/multi-object-datasets) and place it with 'clevr_trans.py'. Running clevr_trans.py (tensorflow needed for reading .tfrecords file) will create a 'CLEVR_with_mask' folder and decode the CLEVR images and masks under this folder. 

### CLEVRTex

CLEVRTex datasets can be directly downloaded from [CLEVRTex Project Page](https://www.robots.ox.ac.uk/~vgg/data/clevrtex/). Unzip the downloaded zip file will produce a 'clevrtex_full' folder with 50 sub-folder. Each sub-folder contains 1000 CLEVRTex images and their annotations.

## Usage

After model checkpoints and datasets are prepared, please adjust file paths in the demo file (demo_CLEVR.py, demo_CLEVRTex.py, etc.), including the model checkpoint path and the root directory of the datasets. 

Finally, run the demo file to evaluate the models' performance.