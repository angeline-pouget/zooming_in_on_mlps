'''
Script for dataset creation to investigate texture and shape biases
Datasets:
1) grayscale images
2) edge images
3) occluded images
4) shuffle images
5) stylized images
'''

import os
import cv2
import json
import glob
import random
import torch.nn as nn
from PIL import Image
from typing import List
import matplotlib.pyplot as plt
import torch.nn.functional as nnf

import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize

from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
from ffcv.fields import BytesField, IntField, RGBImageField

from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    Convert,
    ImageMixup,
    LabelMixup,
    RandomHorizontalFlip,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze

from .data_stats import *

# Custom transformations

class RandomOcclusions(torch.nn.Module):
    
    def __init__(self, masked_area = 0.0, patch_size = 0):
        
        super(RandomOcclusions, self).__init__()
        if patch_size > 16:
            raise ValueError(f'Patch size for the Random Occlusion transformation is out of range\n')
        if masked_area > 1:
            raise ValueError(f'Masked_area is a percentage and takes values in (0,1)\n')     
        self.patch_size  = patch_size
        self.masked_area = masked_area 
    
    def forward(self, imgs):
        
        if self.patch_size == 0: return imgs
        batch_size, c, dim_h, dim_w = imgs.size()
        n_points = int(dim_h*dim_w*self.masked_area/ self.patch_size**2)
        if n_points == 0: return imgs
        points_x = torch.randint(0, dim_h - self.patch_size + 1, size=(batch_size, n_points))
        points_y = torch.randint(0, dim_w - self.patch_size + 1, size=(batch_size, n_points))
        # sure there are more efficient ways...
        for i in range(batch_size):
            for x in range(self.patch_size):
                for y in range(self.patch_size):
                    imgs[i, :, points_x[i,:]+x, points_y[i,:]+y] = 0
        return imgs
    

class RandomBlackenPixels(torch.nn.Module):
    def __init__(self, probability=0.5):
        super(RandomBlackenPixels, self).__init__()
        self.probability = probability

    def forward(self, imgs):
        
        #batch_size, c, dim_h, dim_w = imgs.size()   
        
        #for i in range(batch_size):
        #    mask = torch.rand(imgs[:, :, 0, 0].shape) < self.probability
        #    print(mask.shape)
        #    imgs[i,:, mask] = 0.0
        
        #mask = torch.rand(imgs.size(0), 1, 1, 1) < self.probability
        #imgs[mask.expand_as(imgs)] = 0.0
        
        batch_size, c, dim_h, dim_w = imgs.size()   

        for i in range(batch_size):
            mask = torch.rand((c, dim_h, dim_w)) < self.probability
            imgs[i, mask] = 0.0

        
        return imgs


class ShufflePatches(torch.nn.Module):
    
    def __init__(self, patch_size = 0):
        super(ShufflePatches, self).__init__()
        self.ps = patch_size
    
    def forward(self, imgs):
        
        # imgs must be in format BxCxHxW
        # divide the batch of images into non-overlapping patches
        u = nnf.unfold(imgs, kernel_size=self.ps, stride=self.ps, padding=0)
        # permute the patches of each image in the batch
        pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim = 0)
        # fold the permuted patches back together
        f = nnf.fold(pu, imgs.shape[-2:], kernel_size=self.ps, stride = self.ps, padding = 0)
        return f

class CustomGrayscale(torch.nn.Module):
    def __init__(self):
        super(CustomGrayscale, self).__init__()
        self.grayscale = torchvision.transforms.Grayscale(num_output_channels=3)
    
    def forward(self, imgs):
        # imgs must be in format BxCxHxW
        return self.grayscale(imgs)

class EdgeDetector(torch.nn.Module):
    def __init__(self, smooth=False, kernel_size=3):
        super(EdgeDetector, self).__init__()
        self.smooth = smooth
        self.kernel_size = kernel_size
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        Gx = torch.tensor([[2.0, 0.0, -2.0],[4.0, 0.0, -4.0],[2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G  = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0).unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad = False) 
    
    def forward(self, imgs):
        if self.smooth:
            gblur = torchvision.transforms.GaussianBlur(self.kernel_size,sigma=(0.1, 2.0))
            imgs  = gblur.forward(imgs)
        imgs = torchvision.transforms.Grayscale(num_output_channels=1).forward(imgs)
        imgs = self.filter(imgs)
        imgs = torch.mul(imgs, imgs)
        imgs = torch.sum(imgs, dim=1, keepdim=True)
        imgs = torch.sqrt(imgs)
        imgs = imgs/torch.amax(imgs, dim=(2,3), keepdim = True)
        imgs = 1 - imgs
        
        # the edge of the image are recognized as edges... Fix this manually
        imgs[:, :, 0:1, :] = 1
        imgs[:, :, :, 0:1] = 1
        imgs[:, :, -2:, :] = 1
        imgs[:, :, :, -2:] = 1
        imgs = torch.cat([imgs]*3, dim = 1)
        return imgs

# Define an ffcv dataloader
def get_image_pipeline(crop_resolution, dtype, dev, mean, std, dataset_type=None):
    if dataset_type == None:
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToDevice(dev, non_blocking=True),
                                           ToTorchImage(),
                                           Convert(dtype),
                                           torchvision.transforms.Normalize(mean, std)
                                           ]
    elif dataset_type == 'create_stylized':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor()
                                           ]
    elif dataset_type == 'occluded':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToDevice(dev, non_blocking=True),
                                           ToTorchImage(),
                                           Convert(dtype),
                                           torchvision.transforms.Normalize(mean, std),
                                           RandomOcclusions(0.5,8)
                                           ]
    elif dataset_type == 'random_blacken':
        image_pipeline: List[Operation] = [ CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                            ToTensor(),
                                            ToDevice(dev, non_blocking=True),
                                            ToTorchImage(),
                                            Convert(dtype),
                                            torchvision.transforms.Normalize(mean, std),
                                            RandomBlackenPixels()
                                            ]
    elif dataset_type == 'shuffled':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToDevice(dev, non_blocking=True),
                                           ToTorchImage(),
                                           Convert(dtype),
                                           torchvision.transforms.Normalize(mean, std),
                                           ShufflePatches(8)
                                           ]
    elif dataset_type == 'grayscale':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToDevice(dev, non_blocking=True),
                                           ToTorchImage(),
                                           Convert(dtype),
                                           torchvision.transforms.Normalize(mean, std),
                                           CustomGrayscale()
                                           ]
    elif dataset_type == 'edged':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToDevice(dev, non_blocking=True),
                                           ToTorchImage(),
                                           Convert(dtype),
                                           EdgeDetector()
                                           ]
        

    return image_pipeline

def get_label_pipeline(dev):
    labe_pipeline: List[Operation] = [IntDecoder(), 
                                      ToTensor(),
                                      ToDevice(dev, non_blocking=True), 
                                      Squeeze()]  
    return labe_pipeline

def get_Shapeloader(
    dataset,
    bs,
    mode,
    augment,
    dev,
    data_resolution=None,
    crop_resolution=None,
    crop_ratio=(0.75, 1.3333333333333333),
    crop_scale=(0.08, 1.0),
    num_samples=None,
    dtype=torch.float32,
    mixup=None,
    data_path='./beton',
    dataset_type = None
):
    mode_name = MODE_DICT[dataset] if mode != 'train' else mode
    os_cache  = OS_CACHED_DICT[dataset]

    if data_resolution is None:
        data_resolution = DEFAULT_RES_DICT[dataset]
    if crop_resolution is None:
        crop_resolution = data_resolution

    real = '' if dataset != 'imagenet_real' or mode == 'train' else 'real_'
    sub_sampled = '' if num_samples is None or num_samples == SAMPLE_DICT[dataset] else '_ntrain_' + str(num_samples)

    beton_path = os.path.join(
        data_path,
        DATA_DICT[dataset],
        real + f'{mode_name}_{data_resolution}' + sub_sampled + '.beton',
    )

    print(f'Loading {beton_path}')

    mean = MEAN_DICT[dataset]
    std  = STD_DICT[dataset]

    image_pipeline = get_image_pipeline(crop_resolution, dtype, dev, mean, std, dataset_type)
    label_pipeline = get_label_pipeline(dev)
    
    
    if mode == 'train':
        num_samples = SAMPLE_DICT[dataset] if num_samples is None else num_samples
        indices = None
    else:
        indices = None

    return Loader(
        beton_path,
        batch_size=bs,
        num_workers=4,
        order=OrderOption.QUASI_RANDOM if mode == 'train' else OrderOption.SEQUENTIAL,
        drop_last=(mode == 'train'),
        pipelines={'image': image_pipeline, 'label': label_pipeline},
        os_cache=os_cache,
        indices=indices,
    )

# for stylized dataset we will use offline a pretrained model. If it works
class CustomDataset:
    def __init__(self, dataset, dir_path):
        
        labels_path = os.path.join(dir_path, f'{dataset}_labels.json')
        if not os.path.exists(labels_path):  raise FileNotFoundError(f'the file {labels_path} does not exists')

        self.dir_path    = dir_path
        with open(labels_path, 'r') as fp:
            self.labels_dict = json.load(fp)
            
    def __getitem__(self, idx):
        
        image_dict = self.labels_dict[str(idx)]
        img_name   = f'batch_{image_dict["batch"]}_pos_{image_dict["index"]}_class_{image_dict["class"]}.jpg'
        img_path   = os.path.join(self.dir_path, img_name)
        
        if not os.path.exists(img_path): raise FileNotFoundError(f'the file {img_path} does not exists')
        pass
    
    def __len__(self):
        return len(self.labels_dict)


def save_image(img, file_name, dir):
    if isinstance(img, torch.Tensor):   img = img.numpy()
    #img = np.transpose(img, (2, 1, 0))
    #img = img/np.max(img)
    
    #print("Array Shape:", img.shape)
    #print("Array Data Type:", img.dtype)
    #print(np.max(img))
    
    im = Image.fromarray(img, "RGB")
    path_to_save = os.path.join(dir, file_name)
    im.save(path_to_save)

def save_dataset(loader, temp_dir):
    
    batch_id   = 0
    counter    = 0
    label_dict = dict()
    
    with torch.no_grad():
        for imgs, targs in tqdm(loader, desc="Dataset Creation"):
            for idx in range(imgs.size(0)):

                file_name = f'batch_{batch_id}_pos_{idx}_class_{targs[idx]}.jpg'
                save_image(imgs[idx], file_name, temp_dir)
                label_dict[counter] = {'batch':batch_id, 'index':idx, 'class': targs[idx]}
                counter += 1
            batch_id += 1
            break

# Stylized Dateaset loader (loads all the images directly
# from the stylized_dataset folder, where they are stored in .jpeg form)

#stylized_dataset contains all images from cifar10 in a stylized fashion and was created 
#using the model from: https://github.com/bethgelab/stylize-datasets 

class StylizedDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images, self.labels = self._load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    def _load_data(self):
        images = []
        labels = []

        for filename in os.listdir(self.folder_path):
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                img_path = os.path.join(self.folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    parts = filename.split("class_")
                    if len(parts) > 1:
                        class_number_part = parts[1].split("-stylized")[0]
                        try:
                            class_number = int(class_number_part)
                            labels.append(class_number)
                            images.append(img_path)
                        except ValueError:
                            print(f"Skipping file {filename} due to invalid class number format.")

        return images, labels



