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
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1VVmQvM_NqxDlldpRvz323C0YIWt6p-AA
```

## Running Experiments
As described in our report, we ran a range of experiments to compare the different models against each other. In the following, we describe in detail how each of these experiments can be replicated using the code provided in this repository.

### Model Finetuning

### Adversarial Accuracy

### Feature Extraction Ability

### Feature Generalizability

### Time Complexity and Performance

### Improving the MLP
