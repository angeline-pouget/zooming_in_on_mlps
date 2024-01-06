# Zooming in on MLPs
​
## Overview
This repository contains the code accompanying our [report](https://docs.google.com/document/d/1t5WilwNGPhp19uKhuBTlZVlwhGhy3XiAn2nhRJZDp58/edit?usp=sharing) (CHANGE LINK HERE)  *Zooming in on MLPs* describing the research project completed in the context of the Deep Learning lecture at ETH Zurich in autumn 2023. In this work we compare the performance of different neural networks architectures, namely multi-layer perceptrons (MLP), convolutional neural networks (CNN) and vision transformers (ViT), with respect to several dimensions such as adversarial robustness or feature extraction ability. 

## Setup

### Environment Setup
For installing the *FFCV* dataloading framework, we refer to the original [repository](https://github.com/libffcv/ffcv). To install the remaining packages, activate the FFCV environment and run 
```
pip install -r requirements.txt
```
​
### Downloading Data
In order to use the efficiency of MLPs to the fullest, we are using a more optimised data loading framework than the standard one provided by *torch*. This is because the data transfer from CPU to GPU otherwise becomes the bottleneck of training, not the gradient computation. To ensure a faster data transfer, we use the *FFCV* framework, which requires converting your dataset first to the **beton** format. 

To simplify this, we provide the CIFAR-10, CIFAR-100 and ImageNet datasets in a beton format. To download these datasets, run
```
gdown --folder https://drive.google.com/drive/folders/1VVmQvM_NqxDlldpRvz323C0YIWt6p-AA
```

## Running Experiments
As described in our report, we ran a range of experiments to compare the different models against each other. In the following, we describe in detail how each of these experiments can be replicated using the code provided in this repository.

### 01 Model Finetuning
Due to resource constraints, it was not possible to train our own MLPs, CNNs and ViTs from scratch in the context of this project. Since we already had access to pre-trained MLP and CNN models on CIFAR-10 and CIFAR-100, we decided to fine-tune the linear classifiers of the "vit_small_patch16_224" vision transformer provided with pre-trained weights by the **timm** library. The code for this, as well as the final models that have been fine-tuned for 7 epochs each can be found in 01_vit_finetuning.ipynb.

### 02 Adversarial Accuracy
We implemented the untargeted FGSM and the PGD algorithm to test the adversarial accuracy of different models on CIFAR-10 and CIFAR-100. We then also looked at the transferability of adversarial examples between the different model architectures. The code to generate and evaluate adversarial examples can be found in 02_adversarial_robustness.ipynb.

### 03 Feature Extraction Ability
To assess the model's inclination towards recognizing either the shape or texture of images, an experiment was conducted using the CIFAR-10 dataset. The dataset underwent a series of transformations, and the model's accuracy for each class was measured. The results were then compared to those obtained from Convolutional Neural Network (CNN) and Vision Transformer (ViT) models. For detailed implementation and analysis, refer to the "03.1_texture_vs_shape_bias.ipynb" notebook. 

### 04 Feature Generalizability

### 05 Time Complexity and Performance
We compared the inference time, total size, parameter size and forward / backward pass size between the different models, also taking into account the classification accuracy. The code for this can be found in 05_inference_time.ipynb.

### 06 Improving the MLP

### 07 Results & Plotting
Any plots we generated for the report have been generated using 07_plots.ipynb.
