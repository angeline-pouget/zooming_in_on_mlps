import time
import detectors

import torch
import timm
from tqdm import tqdm
from torchvision import transforms

from data_utils.data_stats import *
from data_utils.dataloader import get_loader
from models.networks import get_model


class Reshape(torch.nn.Module): 
    def __init__(self, shape=224): 
        super(Reshape, self).__init__()
        self.shape = shape 
        
    def forward(self, x): 
        shape = self.shape
        x = transforms.functional.resize(x, size=(shape, shape))
        if shape == 64:
            bs = x.shape[0]
            x = torch.reshape(x, shape=(bs,-1,))
        return x


def get_test_data_and_model(dataset, model, data_path='/scratch/ffcv/'):
    """
    This function retrieves the data, model and feature extractor (if needed) based on the provided information.

    Parameters:
    dataset (str): The name of the dataset to retrieve (can be cifar10, cifar100 or imagenet).
    model (str): The name of the model to retrieve (can be mlp, cnn or vit; only mlp is supported for dataset imagenet).
    data_path (str): The path to the data.

    Returns (as a tuple):
    data_loader (DataLoader): The retrieved data loader.
    model (Model): The retrieved model.

    Raises:
    AssertionError: If the dataset or model is not supported.
    """

    assert dataset in ('cifar10', 'cifar100', 'imagenet'), f'dataset {dataset} is currently not supported by this function'
    assert model in ('mlp', 'cnn', 'vit'), f'model {model} is currently not supported by this function'

    num_classes = CLASS_DICT[dataset]
    eval_batch_size = 1024
 
    if dataset == 'imagenet':
        data_resolution = 64
        assert model == 'mlp', f'imagenet dataset is only supported by mlp model'
    else:
        data_resolution = 32

    crop_resolution = data_resolution

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True

    if model == 'mlp':
        architecture = 'B_12-Wi_1024'
        checkpoint = 'in21k_' + dataset
        model = get_model(architecture=architecture, resolution=64, num_classes=num_classes, checkpoint=checkpoint)
        model = torch.nn.Sequential(Reshape(shape=64), model)

    if model == 'cnn':
        architecture = 'resnet18_' + dataset
        model = timm.create_model(architecture, pretrained=True)

    if model == 'vit':
        architecture = 'vit_small_patch16_224_' + dataset + '_v7.pth'
        model = torch.load(architecture)
        model = torch.nn.Sequential(Reshape(shape=224), model)

    data_loader = get_loader(
        dataset,
        bs=eval_batch_size,
        mode="test",
        augment=False,
        dev=device,
        mixup=0.0,
        data_path=data_path,
        data_resolution=data_resolution,
        crop_resolution=crop_resolution,
    )

    return data_loader, model