# VeinSegmenter v1.0

Vein Segmenter is a neural network focused on densely segmenting vein pixels 
from images of arms.

## Contents
- VeinSegmenter.ipynb: Jupyter notebook with model defintion, dataloader and train-validate-test pipeline code
- dataset_assembler: python script for preprocessing raw images and associated labels before training

## To Run
- Run the dataset assembler with data images organized into a valid directory structure
- Upload the dataset and notebook to your Google Drive
- Open the notebook in Google Colab. Request an instance with a GPU. No high RAM needed.
- To train, set the config to train mode. Set epochs to some integer > 0. Set the run name. Then, under "Run", choose "Run all" to train.
- To test, set epochs to 0. Make sure the macro LOAD_MODEL is True. Then, under Run, choose "Run all".

## Model
The model is based on the UNet architecture first proposed by Ronneberger et al. The dimension were heavily modified to account for large input image size. To boost performance and improve model efficieny, the conv layers used by the original architecture were replaced with ConvNeXt blocks, as proposed by Liu et al. The block use separate downsampling layers. Layernorm is swapped for Batchnorm layers. No stochastic depth or layer scaling is used.

### Design Rationale
U-Net is the gold standard solution for dense image segmentation in deep learning. However, it is slightly dated; the original paper is from 2016. The design, while compact, did not benefit from the advances proposed by MobileNet (depthwise separable convolutions), which provide a graceful way to tradeoff some performance for drastically smaller computation cost. For the large input images we use, efficiency is key to even fitting the model into GPU RAM and training in a reasonable amount of time. Furthermore, the stock standard 3 x 3 convolution paradigm suggested by earlier landmark models like VGGNet have since been superceded in performance by various other networks, most recently by ConvNeXt.

### Dimensions
Input: 1986 x 960
9 Layers:
Bottleneck:     conv 4 x 4, stride 4        (3 x 1984 x 960) 
Conv 1:         ConvNeXt Block              (64 x 496 x 240)
Conv 2:         ConvNeXt Block              (128 x 248 x 120)
Conv 3:         ConvNeXt Block              (256 x 124 x 60)
Conv 4:         ConvNeXt Block              (512 x 62 x 30)
Conv 5:         ConvNeXt Block              (1024 x 31 x 15)
Conv 6:         ConvNeXt Block              (512 x 62 x 30)
Conv 7:         ConvNeXt Block              (256 x 124 x 60)
Conv 8:         ConvNeXt Block              (128 x 248 x 120)
Conv 9:         ConvNeXt Block              (64 x 496 x 240)
Splatter:       conv 4 x 4, stride 4        (2 x 1984 x 960) 

Each of Conv 1-4 was connected to Conv 6-9 using concat operations along the channel dimension. 

## Pre-processing Guide
Place the dataset_assembler into a directory with the data images. The data images and their associated labels should be placed into directories by the name of the subject and "_arm".

Directory Format:
PATH/{name1}_arm
|-{name1}01.jpg
|-{name1}01_label0.npy
|-{name1}01_label1.npy
|-{name1}01_label2.npy
|-{name1}02.jpg
|-{name1}02_label0.npy
|-{name1}03.jpg
|-{name1}03_label0.npy
|-{name1}03_label1.npy
...
PATH/{name2}_arm
|-{name2}01.jpg
|-{name2}01_label0.npy
|-{name2}01_label1.npy
|-{name2}02.jpg
|-{name2}02_label0.npy
|-{name2}03.jpg
|-{name2}03_label0.npy
...
dataset_assembler.py

For now, dynamic pathing for the dataset root PATH has not been implemented. The numpy files are the segmentation masks. These are the same dimensions as the images in greyscale, with 1 for vein pixels and 0 for non-vein. These masks can be generated using Label Studio, which was our preferred method of labelling.

## Dataset
For various reasons, the dataset is not offered publicly at this time early in the project lifecycle. This may change towards the end of the project. For inquiries, please contact kevinliu@andrew.cmu.edu.