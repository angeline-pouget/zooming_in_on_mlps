# Zooming in on MLPs

This repository contains the code accompanying our [report](https://docs.google.com/document/d/1t5WilwNGPhp19uKhuBTlZVlwhGhy3XiAn2nhRJZDp58/edit?usp=sharing) (CHANGE LINK HERE)  *Zooming in on MLPs* describing the research project completed in the context of the Deep Learning lecture at ETH Zurich in autumn 2023. In this work we compare the performance of different neural networks architectures, namely multi-layer perceptrons (MLP), convolutional neural networks (CNN) and vision transformers (ViT), with respect to several dimensions such as adversarial robustness or feature extraction ability. 

## Setup
For installing the *ffcv* dataloading framework, we refer to the original [repository](https://github.com/libffcv/ffcv). To install the remaining packages, activate the *ffcv* environment and run 
```
pip install -r requirements.txt
```
To download the CIFAR-10 and CIFAR-100 datasets in a *beton* format, run
```
gdown --folder https://drive.google.com/drive/folders/15e2AWuY_vVyGluxiVlgRZnswT-We0RfA?usp=sharing
```

## Running Experiments
As described in our report, we ran a range of experiments to compare the different models against each other. In the following, we describe in detail how each of these experiments can be replicated using the code provided in this repository. 

### 01 Model Finetuning
Due to resource constraints, it was not possible to train our own MLPs, CNNs and ViTs from scratch in the context of this project. Since we already had access to pre-trained MLP and CNN models on CIFAR-10 and CIFAR-100, we decided to fine-tune the linear classifiers of the "vit_small_patch16_224" vision transformer provided with pre-trained weights by the *timm* library. The code for this, as well as the final models that have been fine-tuned for 7 epochs each can be found in **01_vit_finetuning.ipynb** and **vit_models/**.

### 02 Adversarial Accuracy
We implemented the untargeted FGSM and the PGD algorithm to test the adversarial accuracy of different models on CIFAR-10 and CIFAR-100. We then also looked at the transferability of adversarial examples between the different model architectures. The code to generate and evaluate adversarial examples can be found in **02_adversarial_robustness.ipynb**.

### 03 Feature Extraction - Texture & Shape Bias
To assess the model's inclination towards recognizing either the shape or texture of images, an experiment was conducted using the CIFAR-10 dataset. The dataset underwent a series of transformations, and the model's accuracy for each class was measured. The results were then compared to those obtained from Convolutional Neural Network (CNN) and Vision Transformer (ViT) models. For detailed implementation and analysis, refer to the **03_texture_shape_bias.ipynb** notebook. 

### 04 Feature Extraction - CKA, Activation Maximization & Representation Inversion
We analyze and compare hidden layer representations of neural networks using Centered Kernel Alignment (CKA). By employing CKA, we can investigate how the representations propagate within the three architectures and quantify similarities and differences across all architectures. For detailed implementation and analysis, please refer to the 04a_cka.ipynb notebook.

We applied feature visualization techniques developed for CNN networks to other architectures, aiming to visualize the features learned from these networks. The methods implemented include activation maximization and feature inversion. In the former, the goal is to create an image that the model classifies into a given category with a probability of 1. On the other hand, in feature inversion, the goal is, given a hidden representation from a target layer, to construct a random image that produces the same hidden representation. For detailed implementation and analysis, please refer to the 04b_feature_visualization.ipynb notebook.

### 05 Model Calibration
We compare the predictive uncertainty and hence the model calibration of different models by comparing the distribution of confidence values (calculated with Softmax) for correctly classified CIFAR-10 images. We additionally compare the reliability plots. All the code can be found in **05_model_calibration.ipynb**.

### 06 Time Complexity and Performance
We compared the inference time, total size, parameter size and forward / backward pass size between the different models, also taking into account the classification accuracy. The code for this can be found in **06_inference_time.ipynb**.

### 07 Results & Plotting
Additional plots we generated for the report have been generated using **07_plots.ipynb**.
