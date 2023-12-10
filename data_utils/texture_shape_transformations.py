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

# Define an ffcv dataloader
def get_pipeline(dataset_type, crop_resolution, dtype):
    if dataset_type == 'occluded':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToTorchImage(),
                                           RandomOcclusions(0.7,8)]
    elif dataset_type == 'shuffled':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToTorchImage(),
                                           Convert(dtype),
                                           ShufflePatches(8)
                                           ]
    elif dataset_type == 'grayscale':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToTorchImage(),
                                           CustomGrayscale()
                                           ]
    elif dataset_type == 'edged':
        image_pipeline: List[Operation] = [CenterCropRGBImageDecoder(output_size=(crop_resolution, crop_resolution), ratio=1),
                                           ToTensor(),
                                           ToTorchImage(),
                                           Convert(dtype),
                                           EdgeDetector()
                                           ]

    return image_pipeline

def get_loader(
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

    if dataset == 'imagenet_real' and mode != 'train':
        label_pipeline: List[Operation] = [NDArrayDecoder()]
    else:
        label_pipeline: List[Operation] = [IntDecoder()]
    
    image_pipeline = get_pipeline(dataset_type, crop_resolution, dtype)
    
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

def save_image(img, file_name, dir):
    if isinstance(img, torch.Tensor):   img = img.numpy()
    im = Image.fromarray(img)
    path_to_save = os.path.join(dir, file_name)
    im.save(path_to_save)

def main():
    mode            = 'test'
    dataset         = 'cifar10' # cifar10, cifar100, imagenet
    data_resolution = 32        # 32, 64
    crop_resolution = 64
    data_path       = '/scratch/beton'
    batch_size      = 512
    device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_path      = '/scratch/beton'
    dataset_type    = 'edged' # occluded, shuffled, grayscale, edged,stylized

    loader = get_loader(
        dataset   = dataset,
        bs        = batch_size,
        mode      = "test",
        augment   = "False",
        dev       = device,
        mixup     = 0.0,
        data_path = data_path,
        data_resolution = data_resolution,
        crop_resolution = crop_resolution,
        dataset_type    = dataset_type
        )
    
    temp_dir = os.path.join('/tmp', f'{dataset}_{dataset_type}')
    if os.path.exists(temp_dir) == False:   os.makedirs(temp_dir)

    batch_id   = 0
    counter    = 0
    label_dict = dict()
    
    with torch.no_grad():
        for imgs, targs in tqdm(loader, desc="Dataset Creation"):
            for idx in range(imgs.size(0)):

                file_name = f'batch_{batch_id}_pos_{idx}_class_{targs[idx].item(0)}.jpg'
                save_image(imgs[idx], file_name, temp_dir)
                label_dict[counter] = {'batch':batch_id, 'index':idx, 'class': targs[idx].item(0)}
                counter += 1
            
            batch_id += 1
            break
    
    # write label dict in a json file
    json_obj   = json.dumps(label_dict)
    label_path = os.path.join(temp_dir, f'{dataset}_labels.json')
    with open(label_path, "w") as outfile:
        outfile.write(json_obj)
    
    # dump dataset as beton
    #write_path = os.path.join(
    #    write_path, f"{dataset}_{dataset_type}", f"{mode}_{crop_resolution}.beton"
    #)
    #os.makedirs(os.path.dirname(write_path), exist_ok=True)

if __name__ == '__main__':
    main()