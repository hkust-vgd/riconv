# Rotation Invariant Convolutions for 3D Point Clouds Deep Learning	

Zhiyuan Zhang, Binh-Son Hua, David W. Rosen, Sai-Kit Yeung

International Conference on 3D Vision (3DV) 2019  

## Introduction
This is the implementation of the rotation invariant convolution and neural networks for point clouds as shown in our paper. The key idea is to build rotation invariant features and use them to build a convolution to consume a point set. For details, please refer to our [project](https://hkust-vgd.github.io/riconv/).
```
@inproceedings{zhang-riconv-3dv19,
    title = {Rotation Invariant Convolutions for 3D Point Clouds Deep Learning},
    author = {Zhiyuan Zhang and Binh-Son Hua and David W. Rosen and Sai-Kit Yeung},
    booktitle = {International Conference on 3D Vision (3DV)},
    year = {2019}
}
```

## Installation
The code is written in [TensorFlow](https://www.tensorflow.org/install/) and based on [PointNet](https://github.com/charlesq34/pointnet), [PointNet++](https://github.com/charlesq34/pointnet2), and [PointCNN](https://github.com/yangyanli/PointCNN). Please follow the instruction in [PointNet++](https://github.com/charlesq34/pointnet2) to compile the customized TF operators.  

The code has been tested with Python 3.6, TensorFlow 1.13.2, CUDA 10.0 and cuDNN 7.3 on Ubuntu 14.04.

## Usage
### Classification
Please download the preprocessed ModelNet40 dataset [here](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip).  

To train a network that takes XYZ coordinates as input to classify shapes in ModelNet40:
```
python3 train_val_cls.py
```
The evaluation is performed after every training epoch.

### Part Segmentation
Download the preprocessed ShapeNetPart dataset [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip).

To train a network that takes XYZ coordinates as input to segments object parts:
```
python train_val_seg.py
```
The evaluation is performed after every training epoch.

## License
This repository is released under MIT License (see LICENSE file for details).