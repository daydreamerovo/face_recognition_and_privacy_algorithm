 (Still Updating)

# Face-Landmark Detection and Privacy-Preserving Implementation Using Pytorch
This repo implements a five-landmark detection based on UKTFace (cropped, aligned), including nose, eyes, and mouth, with *Resnet18* and *ViT* as backbones for comparison,
Both are fine-tuned using LoRA adapter, and the effect of adding noise to images via FFT is discussed, followed by different methods to "Attack" the model. 
e.g., Wiener Filtering, trained U-net, and so on. Easy for starters to get familiar with face recognition algorithms and implementations.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

# Getting Started
Download this repo using `gh repo clone daydreamerovo/face_recognition_and_privacy_perserving` or download as a zip file. Original annotated images are done using **dlib** and using **shape_predictor_68_face_landmarks.dat** in the models file.

## Get Your Baseline
Run `python train.py` in the terminal to get a baseline for face detection, the best result's weight will be stored in the checkpoints file. If you want to change the backbone for training, simply use `python train.py --backbone vit` or `python train.py --backbone resnet`. The default backbone model is Resnet18. My baseline for two models are stored in checkpoints.

# Prerequisites
Download the prerequisites using `pip install -r requirements.txt`.

versions of modules I used:
**Python**: 3.9
**PyTorch**: 12.8; all tasks run on my RTX5060 Laptop.

# Installation
If `conda install -c conda-forge dlib` does not work for **dlib** installation, you have to install **Visual Studio tools for builders** and click **C++ for desktop** when installing. Then, using` pip install Cmake` to install Cmake; finally, use `pip install dlib`. 

Remember to operate under your created environments if you were using one!!!

**_dlib_** installation issue's solution for Chinese users: [Click here](https://blog.csdn.net/weixin_58961374/article/details/126970461).
