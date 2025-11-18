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
Run `python utils/run_protect.py --modes gaussian salt_pepper poisson --radius 30 --sigma 0.02 --amount 0.02` in the terminal to get a baseline for face detection, the best result's weight will be stored in the checkpoints file. If you want to change the backbone for training, simply use `python train.py --backbone vit` or `python train.py --backbone resnet`. The default backbone model is Resnet18. My baseline for two models is stored in checkpoints.

## Add Noise to Images in the Frequency Domain and Evaluate
Run `python run_protect.py` to add noise to the original images. You can change the noise mode by adding `--mode gaussian` for adding gaussian noise, Poisson and Salt Pepper noises are supported. Sigma, Amount, and radius are hyperparameters controlling how much noise is added and how much of the low-frequency part is kept. 

Run `python utils/update_noise_csv.py --src-csv data/landmarks_dataset.csv  --data-root ../data --modes gaussian salt_pepper poisson` to overwrite the file paths in the CSV file and point them to specific noisy image paths. Remember to check paths in the file, and change them into your own local path, e.g., the data path may in /project/data/ or something.

Run `python eval_noise.py  --meta-path data/landmarks_dataset.csv --checkpoint checkpoints/resnet18/best_model.pth --backbone resnet18 --batch-size 64` to get the NME of baseline models on those noise-added images. Change backbone or meta-path to noisy CSV paths like `--meta-path data/landmarks_dataset_gaussian.csv` to evaluate other modes.

Test on basline if trained with LoRA: `python eval_noise.py --meta-path data/landmarks_dataset_gaussian.csv --checkpoint checkpoints/vit/best_model.pth --backbone vit --batch-size 64 --use-lora --lora-adapter checkpoints/vit/lora`
# Prerequisites
Download the prerequisites using `pip install -r requirements.txt`.

versions of modules I used:
**Python**: 3.9
**PyTorch**: 12.8; all tasks run on my RTX5060 Laptop.

# Installation
If `conda install -c conda-forge dlib` does not work for **dlib** installation, you have to install **Visual Studio tools for builders** and click **C++ for desktop** when installing. Then, using` pip install Cmake` to install Cmake; finally, use `pip install dlib`. 

Remember to operate under your created environments if you were using one!!!

**_dlib_** installation issue's solution for Chinese users: [Click here](https://blog.csdn.net/weixin_58961374/article/details/126970461).
