import torch
import numpy as np
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.nn.functional import softmax
from data_utils.data_stats import *


#post_transforms = transforms.Compose([transforms.Normalize(mean = [0, 0, 0], std = [1/0.229, 1/0.224, 1/0.225]),
#                                      transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1, 1, 1])])
post_transforms = transforms.Compose([transforms.Normalize(mean = [0, 0, 0], std = [1/0.247, 1/0.243, 1/0.261]),
                                      transforms.Normalize(mean = [-0.491, -0.482, -0.446], std = [1, 1, 1])])

def scale(arr):
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

class Normalize(torch.nn.Module):
    def __init__(self, mean, std, model_type):
        super(Normalize, self).__init__()
        self.mean =  mean
        self.std  = std
        self.model_type = model_type
    
    def forward(self, x):
        x = (x-self.mean)/self.std
        if self.model_type == 'mlp':
            x = torch.reshape(x, (1, -1))
        return x
    
def denormalize(tensor, mean, std):
    return tensor*std + mean

def normalize(tensor, mean, std):
    return (tensor-mean)/std

def postprocess_image(img_t, rgb = True):
    if img_t.shape[-1] != 32:
        img_t = transforms.functional.resize(img_t, size=(32, 32))
    if rgb:
        img_t = post_transforms(img_t[0])     
    img_np = img_t.detach().cpu().numpy().transpose(1,2,0)
    img_np = scale(np.clip(img_np, 0, 1))
    
    return img_np

def alpha_norm(input_matrix, alpha):
        """
            Converts matrix to vector then calculates the alpha norm
        """
        alpha_norm = ((input_matrix.reshape(-1))**alpha).sum()
        return alpha_norm

def euclidian_loss(org_matrix, target_matrix):
        """
            Euclidian loss is the main loss function in the paper
            ||fi(x) - fi(x_0)||_2^2 / ||fi(x_0)||_2^2
        """
        distance_matrix    = target_matrix - org_matrix
        euclidian_distance = alpha_norm(distance_matrix, 2)
        normalized_euclidian_distance = euclidian_distance / alpha_norm(org_matrix, 2)
        return normalized_euclidian_distance

def total_variation_norm(input_matrix, beta):
        """
            Total variation norm is the second norm in the paper
            represented as R_V(x)
        """
        to_check   = input_matrix[:, :-1, :-1]  # Trimmed: right - bottom
        one_bottom = input_matrix[:, 1:, :-1]  # Trimmed: top - right
        one_right  = input_matrix[:, :-1, 1:]  # Trimmed: top - right
        total_variation = (((to_check - one_bottom)**2 + (to_check - one_right)**2)**(beta/2)).sum()
        return total_variation

def generate_image(model, target_class, epochs, min_prob, lr, weight_decay, 
                   step_size = 100, gamma = 0.6, noise_size = 224, model_type = None, img = None, 
                   p_freq = 50, device = 'cpu', dataset = None, figsize = (6, 6), save_path = None):
    
    """
        Starting from a random initialization, generates an image that maximizes the score for a specific class using
        gradient ascent
    """

    # the random number generator to be used
    init = torch.randn #N(0,1)
    #init = torch.rand #Unif[0,1]

    mean = MEAN_DICT[dataset]/255
    mean = torch.tensor(mean, dtype = torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std  = STD_DICT[dataset]/255
    std = torch.tensor(std, dtype = torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    if model_type == 'mlp':
        noise = init([1, 3, noise_size, noise_size]).to(device)
    elif model_type == 'cnn':
        noise = init([1, 3, noise_size, noise_size]).to(device)
    elif model_type == 'vit':
        noise = init([1, 3, noise_size, noise_size]).to(device)
    if img != None:
        # the loader will take care of the dimension to be correct
        noise = img.clone()
    
    # initially the image is normalized. If it is random noise is N(0,1),
    # if it is a real image, loader has already normalized it.
    noise = denormalize(noise, mean, std)
    noise = noise.clamp_(min=0.0, max = 1.0)

    init_image = torch.clone(noise)

    model = torch.nn.Sequential(Normalize(mean, std, model_type), model)
    model = model.to(device)
    
    for i in range(1, epochs + 1):

        noise.requires_grad_()
        model.zero_grad()
        
        outs = model(noise)
        loss = torch.nn.CrossEntropyLoss()(outs, torch.tensor([target_class], dtype = torch.int64))
        loss.backward()
        grad_tensor = noise.grad

        if torch.isnan(grad_tensor).any() == True:
            print("Nan found")
            break
        
        p = softmax(outs[0], dim = 0)[target_class]
        print("probability for target class", p.item())
        if p.item() > 0.95: break
        
        noise.requires_grad = False
        with torch.no_grad():
            noise       = noise - lr*grad_tensor
            noise.clamp_(min=0.0, max = 1.0)

    rgb      = True
    fig, axs = plt.subplots(1, figsize = figsize)
    if model_type == 'mlp':
        noise = torch.reshape(noise, (1, 3, noise_size, noise_size))
    img_np   = postprocess_image(noise, rgb = True)
    if rgb:
        axs.imshow(img_np)

    axs.set_xticks([])
    axs.set_yticks([])
    plt.title(f'Activation maximization for {model_type}')
    if save_path:
        fig.savefig(save_path)
    return init_image, noise


def feature_inversion(model, 
                      modules, 
                      img, 
                      noise_size,
                      epochs, 
                      lr, 
                      step_size = 100, 
                      gamma     = 0.6, 
                      mu        = 1e-1, 
                      device    = 'cuda', 
                      rgb       = True, 
                      figsize   = (16, 16), 
                      save_path = None):
    
    recreated_imgs = []
    orig_img       = img

    for module in modules:   
        recreated_imgs.append(feature_inversion_helper(model, 
                                                       module, 
                                                       orig_img, 
                                                       epochs     = epochs,
                                                       lr         = lr, 
                                                       step_size  = step_size,
                                                       gamma      = gamma, 
                                                       mu         = mu,
                                                       noise_size = noise_size, 
                                                       device     = device))
        
    fig, axs = plt.subplots(1, len(recreated_imgs), figsize = figsize)
    
    for i in range(len(recreated_imgs)):
        if rgb:
            axs[i].imshow(postprocess_image(recreated_imgs[i], rgb = rgb))
        else:
            axs[i].imshow(postprocess_image(recreated_imgs[i], rgb = rgb), cmap = 'gray')

    if save_path:
        fig.savefig(save_path)


def feature_inversion_helper(model,
                             module, 
                             orig_img, 
                             epochs, lr, 
                             step_size, 
                             gamma, 
                             mu, 
                             noise_size = 224, 
                             init = torch.randn, 
                             device = 'cpu',
                             dataset = 'cifar10'):
    
    """
        Performs feature inversion on one module
    """

    mean = MEAN_DICT[dataset]/255
    mean = torch.tensor(mean, dtype = torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    std  = STD_DICT[dataset]/255
    std = torch.tensor(std, dtype = torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    acts = [0]    
    def hook_fn(self, input, output):
        acts[0] = output    
    handle = module.register_forward_hook(hook_fn)
    
    _             = model(orig_img)
    orig_features = acts[0]
    
    noise = init([1, 3, noise_size, noise_size]).to(device)
    noise = denormalize(noise, mean, std)
    noise.clamp_(min=0.0, max = 1.0)
    noise = normalize(noise, mean, std)

    noise.requires_grad = True
    
    #opt         = torch.optim.SGD([noise], lr = lr,  momentum=0.9)

    alpha_reg_alpha  = 6
    alpha_reg_lambda = 1e-7

    tv_reg_beta   = 2
    tv_reg_lambda = 1e-8

    for i in range(epochs):

        noise.requires_grad = True
        
        #opt.zero_grad()
        model.zero_grad()
        
        _ = model(noise)
        curr_features = acts[0]

        euc_loss            = 1e-1 * euclidian_loss(orig_features, curr_features)
        reg_alpha           = alpha_reg_lambda * alpha_norm(noise, alpha_reg_alpha)
        reg_total_variation = tv_reg_lambda * total_variation_norm(noise, tv_reg_beta)
         
        loss = euc_loss + reg_alpha + reg_total_variation
        
        loss.backward(retain_graph = True)
        grad_tensor = noise.grad
        #opt.step()
        print(loss.item())

        #if i % 40 == 0:
        #    for param_group in opt.param_groups:
        #        param_group['lr'] *= 1/10

        if i % 40 == 0:
            lr = lr/10

        noise.requires_grad = False
        with torch.no_grad():
            noise       = noise - lr*grad_tensor
            #noise       = denormalize(noise, mean, std)
            noise.clamp_(min=0.0, max = 1.0)
            #noise       = normalize(noise, mean, std)
        
            
    handle.remove()
    return noise    