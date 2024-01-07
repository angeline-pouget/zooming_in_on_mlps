import torch
import torch.nn as nn


def _find_layers_mlp(model):
    #counts only the linear layers, need work for vits and cnns
    #print(model)
    name_list = []
    for name, module in model.named_modules():
        module_type = str(type(module))
        if module_type.find('Linear') !=-1:
            name_list.append(name)
    return name_list, len(name_list)

def _find_layers_cnn(model):
    #counts only the linear layers, need work for vits and cnns
    #print(model)
    name_list = []
    matches   = ['Linear', 'Conv2d', 'AdaptiveAvgPool2d', 'SelectAdaptivePool2d', 'Identity'] 
    for name, module in model.named_modules():
        module_type = str(type(module))
        if any(x in module_type for x in matches):
            name_list.append(name)
    return name_list, len(name_list)

def _find_layers_vit(model):
    #counts only the linear layers, need work for vits and cnns
    #print(model)
    name_list = []
    matches   = ['Attention', 'Linear']
    for name, module in model.named_modules():
        module_type = str(type(module))
        if any(x in module_type for x in matches):
            name_list.append(name)
    return name_list, len(name_list)

def find_layers(model, model_type = 'mlp'):
    if model_type == 'mlp':
        name_list, name_len = _find_layers_mlp(model)
    elif model_type == 'cnn':
        name_list, name_len = _find_layers_cnn(model)
    elif model_type == 'vit':
        name_list, name_len = _find_layers_vit(model)
    return name_list, name_len
    
def register_hooks(model, activations, model_type = 'mlp'):
    if model_type == 'mlp':
        matches = ['Linear']
    elif model_type == 'cnn':
        matches = ['Linear', 'Conv2d', 'AdaptiveAvgPool2d', 'SelectAdaptivePool2d', 'Identity']
    elif model_type == 'vit':
        matches = ['Attention', 'Linear']

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    for name, module in model.named_modules():
        module_type = str(type(module))
        if any(x in module_type for x in matches):
            module.register_forward_hook(get_activation(name)) 

def get_activations(imgs, model, activations):
    preds = model(imgs)
    acts  = [activations[k] for k in activations.keys()]
    return acts

def process_batch(model, imgs, cka, activations):
   acts = get_activations(imgs, model, activations)
   cka.update_state(acts)

class MinibatchCKA():
    def __init__(self, num_layers, num_layers2=None, across_models=False, dtype=torch.float32):
        self.dtype         = dtype
        self.across_models = across_models
        if num_layers2 is None:
            num_layers2 = num_layers
        self.hsic_accumulator = torch.zeros((num_layers, num_layers2), dtype=dtype)
    
        if across_models:
            self.hsic_accumulator_model1 = torch.zeros((num_layers, ), dtype=dtype)
            self.hsic_accumulator_model2 = torch.zeros((num_layers2,), dtype=dtype)
    
    def _generate_gram_matrix(self, x):
        """
        Generate Gram matrix and preprocess to compute unbiased HSIC.
        
        This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
        L. (2014). Partial distance correlation with methods for dissimilarities.
    
        Args:
            x: A [num_examples, num_features] matrix.

        Returns:
            A [num_examples ** 2] vector.
        """
        x    = torch.reshape(x, (x.shape[0], -1))
        x_n  = torch.norm(x, dim = 1)
        x    = x/x_n[:,None]
        gram = torch.matmul(x,x.T)
        n    = gram.shape[0] 
        gram[range(n), range(n)] = 0.0
        gram  = gram.to(self.dtype)
        means = torch.sum(gram, dim = 0)/(n-2)
        means-= torch.sum(means)/(2*(n-1))

        gram -= means[:, None]
        gram -= means[None, :]
        gram[range(n), range(n)] = 0.0
        gram = torch.reshape(gram, (-1,))
        return gram 

    def update_state(self, activations):
        """
        Accumulate minibatch HSIC values.
        
        Args:
        activations: A list of activations for all layers.
        """
        layer_grams = [self._generate_gram_matrix(x) for x in activations]
        layer_grams = torch.stack(layer_grams, dim=0)
        self.hsic_accumulator.add_(torch.matmul(layer_grams, layer_grams.T))

    def update_state_across_models(self, activations1, activations2):
        """
        Accumulate minibatch HSIC values from different models.
        
        Args:
            activations1: A list of activations for all layers in model 1.
            activations2: A list of activations for all layers in model 2.
        """
        assert self.hsic_accumulator.shape[0] == len(activations1), 'Dimensions mismatch for activations1'
        assert self.hsic_accumulator.shape[1] == len(activations2), 'Dimensions mismatch for activations2'
        
        layer_grams1 = [self._generate_gram_matrix(x) for x in activations1]
        layer_grams1 = torch.stack(layer_grams1, dim=0)  
        
        layer_grams2 = [self._generate_gram_matrix(x) for x in activations2]
        layer_grams2 = torch.stack(layer_grams2, dim=0) 

        self.hsic_accumulator.add_(torch.matmul(layer_grams1, layer_grams2.T)) 
        self.hsic_accumulator_model1.add_(torch.einsum('ij,ij->i', layer_grams1, layer_grams1))
        self.hsic_accumulator_model2.add_(torch.einsum('ij,ij->i', layer_grams2, layer_grams2))

    def result(self):

        if self.across_models == True:
            normalization1 = torch.sqrt(self.hsic_accumulator_model1)
            normalization2 = torch.sqrt(self.hsic_accumulator_model2)
            mean_hsic      = self.hsic_accumulator / normalization1[:, None]
            mean_hsic      = mean_hsic / normalization2[None, :]
        else:
            normalization = torch.sqrt(torch.diagonal(self.hsic_accumulator))
            mean_hsic     = self.hsic_accumulator/normalization[:, None]
            mean_hsic    /= normalization[None, :] 
        return mean_hsic
