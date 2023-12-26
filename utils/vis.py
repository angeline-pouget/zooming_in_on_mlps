import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn.functional import softmax


post_transforms = transforms.Compose([transforms.Normalize(mean = [0, 0, 0], std = [1/0.229, 1/0.224, 1/0.225]),
                                      transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1, 1, 1])])

def scale(arr):
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

def postprocess_image(img_t, rgb = True):
    if rgb:
        img_t = post_transforms(img_t[0])     

    img_np = img_t.detach().cpu().numpy().transpose(1,2,0)
    img_np = scale(np.clip(img_np, 0, 1))
    
    return img_np


def generate_image(model, target_class, epochs, min_prob, lr, weight_decay, step_size = 100, gamma = 0.6,
                        noise_size = 224, model_type = None, p_freq = 50, init = torch.randn, device = 'cpu', figsize = (6, 6), save_path = None):
    
    """
        Starting from a random initialization, generates an image that maximizes the score for a specific class using
        gradient ascent
    """

    name, weights = next(model.named_parameters())
    in_size = weights.size()[1]
    
    if model_type == 'mlp':
        noise = init([1, noise_size*noise_size*3]).to(device)
    if model_type == 'cnn':
        noise = init([1, in_size, noise_size, noise_size])
    noise.requires_grad = True
    model     = model.to(device)
    opt       = torch.optim.SGD([noise], lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    
    for i in range(1, epochs + 1):
        opt.zero_grad()
        outs = model(noise)
        p    = softmax(outs[0], dim = 0)[target_class]
        
        if i % p_freq == 0 or i == epochs:        
            print('Epoch: {} Confidence score for class {}: {}'.format(i, target_class, p))
            
        if p > min_prob:
            print('Reached {} confidence score in epoch {}. Stopping early.'.format(p, i))
            break
            
        #obj = - outs[0][target_class]
        obj = -1 * torch.log(p)
        obj.backward()
        opt.step()
        scheduler.step()
    
    rgb      = in_size > 1
    fig, axs = plt.subplots(1, figsize = figsize)
    if model_type == 'mlp':
        noise = torch.reshape(noise, (1, 3, noise_size, noise_size))
    img_np   = postprocess_image(noise, rgb = True)
    if rgb:
        axs.imshow(img_np)

    axs.set_xticks([])
    axs.set_yticks([])

    if save_path:
        fig.savefig(save_path)
    
    return noise