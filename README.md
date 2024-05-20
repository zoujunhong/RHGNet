# RHGNet
Pytorch implementations of "Learning Object-Centric Representation via Reverse Hierarchy Guidance"

## Environment

Codes are run under Pytorch 1.10.1. Numpy and opencv-python are needed for reading images.

## Model checkpoint

We provide all checkpoints [here](https://drive.google.com/drive/folders/1SIkil9mclYwFrhxx41FlmK2cNxrWlIMW?usp=sharing). 

## Data preparation

### Fast access to Datasets

We also provide a demo [here](https://drive.google.com/drive/folders/1SIkil9mclYwFrhxx41FlmK2cNxrWlIMW?usp=sharing) for each dataset, each contains 1000 images randomly sampled from the test dataset. For a quick start, directly download and unzip the zip file.

### Full CLEVR Dataset

Download the tfrecord files of CLEVR (clevr_with_masks_train.tfrecords) from [Multi Object Datasets](https://console.cloud.google.com/storage/browser/multi-object-datasets) and place it with 'clevr_trans.py'. Running clevr_trans.py (tensorflow needed for reading .tfrecords file) will create a 'CLEVR_with_mask' folder and decode the CLEVR images and masks under this folder. 

### Full CLEVRTex Dataset

CLEVRTex datasets can be directly downloaded from [CLEVRTex Project Page](https://www.robots.ox.ac.uk/~vgg/data/clevrtex/). Unzip the downloaded zip file will produce a 'clevrtex_full' folder with 50 sub-folder. Each sub-folder contains 1000 CLEVRTex images and their annotations.

## Usage

After model checkpoints and datasets are prepared,  run the demo files in the root directrory (demo_CLEVR.py, demo_CLEVRTex.py, etc.) to evaluate the model. When running the demo file, you can pass in 4 optional parameters:

- **demo**. Whether to use the demo dataset.
- **dataroot**. The path to the root directory of dataset.
- **checkpoint**. The path to checkpoint file.
- **visualization**. Whether to visualize the result. When visualzation is store true, the program will pause after each image is processed and save the reconstruction and segmentation result as demo/mask.png. Press 'enter' to go to the next image.

For example, if you want to run with CLEVR demo dataset and show the visualization result, just enter the command

```shell
python demo_CLEVR.py --demo --dataroot PATH_TO_DATASET --checkpoint PATH_TO_CKECKPOINT --visualization
```
