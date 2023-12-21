import os
import json
import tqdm
import torch
from data_utils.texture_shape_transformations import *

def main():
    mode            = 'test'
    dataset         = 'imagenet' # cifar10, cifar100, imagenet
    data_resolution = 64         # 32, 64
    crop_resolution = 64
    data_path       = '/scratch/beton'
    batch_size      = 1024
    device          = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    write_path      = '/scratch/beton'
    dataset_type    = None # occluded, shuffled, grayscale, edged,stylized

    loader = get_Shapeloader(
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