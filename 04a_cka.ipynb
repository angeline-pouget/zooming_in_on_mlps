{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The autoreload extension is already loaded. To reload it, use:\n",
      "  %reload_ext autoreload\n"
     ]
    }
   ],
   "source": [
    "# These two lines ensure that we always import the latest version of a package, in case it has been modified.\n",
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "import timm\n",
    "import torch\n",
    "import detectors\n",
    "from utils import cka as CKA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from tqdm import tqdm\n",
    "\n",
    "from data_utils.data_stats import *\n",
    "from models.networks import get_model\n",
    "from data_utils.dataloader import get_loader\n",
    "from data_utils.dataset_to_beton import get_dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# define important parameters\n",
    "\n",
    "dataset         = 'cifar10'               # One of cifar10, cifar100, stl10, imagenet or imagenet21\n",
    "num_classes     = CLASS_DICT[dataset]\n",
    "data_path       = '/scratch/data/ffcv'\n",
    "model_path      = '/scratch/zooming_in_on_mlps/vit_models'\n",
    "eval_batch_size = 32\n",
    "data_resolution = 32 \n",
    "checkpoint      = None\n",
    "\n",
    "cka_internal    = True \n",
    "\n",
    "#model1_type     = 'mlp'                   \n",
    "#checkpoint      = 'in21k_cifar10'       \n",
    "#architecture    = 'B_12-Wi_1024'        \n",
    "#crop_resolution = 64            \n",
    "\n",
    "#model1_type     = 'cnn'               \n",
    "#architecture    = 'resnet18_' + dataset                      \n",
    "#crop_resolution = 32\n",
    "\n",
    "model1_type     = 'vit'                  \n",
    "architecture    = 'vit_small_patch16_224_' + dataset + '_7.pth'        \n",
    "crop_resolution = 224          \n",
    "\n",
    "\n",
    "# model 2\n",
    "#model2_type      = 'cnn'\n",
    "#architecture2    = 'resnet18_' + dataset\n",
    "#crop_resolution2 = 32\n",
    "\n",
    "model2_type      = 'vit'\n",
    "crop_resolution2 = 224\n",
    "architecture2    = 'vit_small_patch16_224_' + dataset + '_v7.pth'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_models_full(model_type, \n",
    "                    architecture, \n",
    "                    resolution  = crop_resolution, \n",
    "                    num_classes = CLASS_DICT[dataset], \n",
    "                    checkpoint  = checkpoint, \n",
    "                    model_path   = model_path):\n",
    "    if model_type == 'mlp':\n",
    "        model = get_model(architecture=architecture, resolution = resolution, \n",
    "                          num_classes=num_classes,checkpoint= checkpoint)\n",
    "    elif model_type == 'cnn':\n",
    "        model = timm.create_model(architecture, pretrained=True)\n",
    "    elif model_type == 'vit':\n",
    "        model = torch.load(os.path.join(model_path, architecture))\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "# load the models\n",
    "torch.backends.cuda.matmul.allow_tf32 = True\n",
    "device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "model1 = get_models_full(model1_type, architecture)\n",
    "if cka_internal == False:\n",
    "    model2 = get_models_full(model2_type, architecture2)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading /scratch/data/ffcv/cifar10/val_32.beton\n"
     ]
    }
   ],
   "source": [
    "# Get the custom test loader need to take care of different image sizes. Now fixed only for mlp with 64x64\n",
    "if cka_internal == True:\n",
    "    loader = get_loader(\n",
    "        dataset,\n",
    "        bs=eval_batch_size,\n",
    "        mode=\"test\",\n",
    "        augment=False,\n",
    "        dev=device,\n",
    "        mixup=0.0,\n",
    "        data_path=data_path,\n",
    "        data_resolution=data_resolution,\n",
    "        crop_resolution=crop_resolution,\n",
    "    )\n",
    "else:\n",
    "    loader1 = get_loader(\n",
    "        dataset,\n",
    "        bs=eval_batch_size,\n",
    "        mode=\"test\",\n",
    "        augment=False,\n",
    "        dev=device,\n",
    "        mixup=0.0,\n",
    "        data_path=data_path,\n",
    "        data_resolution=data_resolution,\n",
    "        crop_resolution=crop_resolution,\n",
    "    )\n",
    "    loader2 = get_loader(\n",
    "        dataset,\n",
    "        bs=eval_batch_size,\n",
    "        mode=\"test\",\n",
    "        augment=False,\n",
    "        dev=device,\n",
    "        mixup=0.0,\n",
    "        data_path=data_path,\n",
    "        data_resolution=data_resolution,\n",
    "        crop_resolution=crop_resolution2,\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "@torch.no_grad()\n",
    "def compute_cka_internal(model, loader, cka, model_type):\n",
    "\n",
    "    model.eval()\n",
    "    #register hooks\n",
    "    activations = {} \n",
    "    CKA.register_hooks(model, activations, model_type)\n",
    "    i = 0 \n",
    "    for ims, targs in tqdm(loader, desc=\"Evaluation\"):\n",
    "        if model_type == 'mlp': ims   = torch.reshape(ims, (ims.shape[0], -1))\n",
    "        CKA.process_batch(model, ims, cka, activations)\n",
    "        i += 1\n",
    "    return cka.result()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "@torch.no_grad()\n",
    "def compute_cka_across(model1, model2, loader, loader2, cka, model1_type, model2_type):\n",
    "\n",
    "    model1.eval()\n",
    "    model2.eval()\n",
    "    #register hooks\n",
    "    activations1 = {}\n",
    "    activations2 = {} \n",
    "    CKA.register_hooks(model1, activations1, model1_type)\n",
    "    CKA.register_hooks(model2, activations2, model2_type)\n",
    "\n",
    "    k = 0\n",
    "    for i, (ims, targs) in enumerate(tqdm(loader, desc=\"Evaluation\")):\n",
    "        for j, (ims2, targs2) in enumerate(loader2):\n",
    "            \n",
    "            if i != j:  continue\n",
    "            assert torch.equal(targs,targs2), f'Mismatch in batches {i}, {j}'\n",
    "            print(targs)\n",
    "            if model1_type == 'mlp': \n",
    "                ims   = torch.reshape(ims, (ims.shape[0], -1))\n",
    "            if model2_type == 'mlp':\n",
    "                ims2 = torch.reshape(ims2, (ims2.shape[0], -1))\n",
    "            \n",
    "            acts1 = CKA.get_activations(ims, model1, activations1)\n",
    "            acts2 = CKA.get_activations(ims2, model2, activations2)\n",
    "            cka.update_state_across_models(acts1, acts2)\n",
    "            break \n",
    "    return cka.result()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The parameter cka_internal has value True\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluation:   0%|          | 0/313 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Evaluation:  19%|█▉        | 60/313 [06:39<28:04,  6.66s/it]\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAf8AAAHHCAYAAACx2FF+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/P9b71AAAACXBIWXMAAA9hAAAPYQGoP6dpAAB2CUlEQVR4nO3deXxTVfo/8E+SZumW7gtlaVmUTQEFQcQFBcWNEUFFxQFxRxhFfl8Xvo6gziguXxnUQVBGARUGl3FDENQqrixaRUUF2SlLWyh0S9ukTc7vDyRj6HMKhxZCm8/bV17SJzf3nHtzk5N773POsSilFIiIiChiWMNdASIiIjq22PgTERFFGDb+REREEYaNPxERUYRh409ERBRh2PgTERFFGDb+REREEYaNPxERUYRh409ERBRh2PjTcaF///7o379/uKtRpx5btmyBxWLBnDlzjmk9wlXuAa+88go6deoEu92OxMTEsNThWFm2bBksFguWLVtm/No5c+bAYrFgy5YtjVafBx98EBaLpdHWRyRh49+MbNy4EbfeeivatWsHl8sFt9uNfv364emnn0ZVVVVwuZycHFx66aV1Xv/KK6/AZrPhwgsvRHV1dchzixcvhsViQVZWFgKBwFHflkgxf/58TJs2LdzVCLF27Vpcf/31aN++PWbNmoUXXngh3FVq0rp164Y2bdqgvpHU+/Xrh4yMDNTW1obED/wQONTjePjhTE1LVLgrQI1j0aJFuPLKK+F0OjFy5EicdNJJ8Pl8+PLLL3H33Xfj559/rvdLfN68ebj++usxcOBAvPPOO3C5XHWez8nJwZYtW/DJJ59g4MCBjVr/Dz/8sFHX11iys7NRVVUFu91+VNY/f/58rFmzBuPHjz+m5dZn2bJlCAQCePrpp9GhQ4djXn5zM2LECNx333344osvcPbZZ9d5fsuWLVi+fDnGjRuHqKgo/PWvf8V9990HABg6dGjIe1BRUYExY8bg8ssvx9ChQ4PxjIyMo78h1Kyw8W8GNm/ejKuvvhrZ2dn45JNP0KJFi+BzY8eOxYYNG7Bo0SLt6xcsWIBRo0bhvPPOw7vvvlun4fd4PHj33XcxZcoUzJ49G/PmzWv0xt/hcDTq+hqLxWKpsz+ac7kAUFRUBACNerm/srISMTExjba+puTaa6/FxIkTMX/+fLHx//e//w2lFEaMGAEAiIqKQlTU/q/mbt26oVu3bsFl9+zZgzFjxqBbt2647rrrjs0GULPEy/7NwBNPPIGKigq8+OKLIQ3/AR06dMCdd94pvvb111/Hddddh/79++O9994TG5y3334bVVVVuPLKK3H11VfjrbfeqnNbQDJu3DjExcWhsrKyznPXXHMNMjMz4ff7Acj3/J999ll07doVMTExSEpKQq9evTB//vzg89dffz1ycnLqrFu6Zzp79mycd955SE9Ph9PpRJcuXTBjxoxDbsPB994P3B+WHn+sy7vvvotLLrkEWVlZcDqdaN++Pf72t78Ft/fANi9atAhbt26tsw7dPf9PPvkEZ511FmJjY5GYmIjLLrsMv/76q7j9GzZswPXXX4/ExEQkJCRg9OjR4nvxRzk5OZg8eTIAIC0tDRaLBQ8++GDw+eeeew5du3aF0+lEVlYWxo4di5KSkpB19O/fHyeddBLy8vJw9tlnIyYmBv/7v/+rLfP6669HXFwctm3bhksvvRRxcXFo2bIlpk+fDgD46aefcN555yE2NhbZ2dkhx8ABmzZtwpVXXonk5GTExMTg9NNPF3/wbt++HUOGDEFsbCzS09Nx1113wev1ivVauXIlLrzwQiQkJCAmJgbnnHMOvvrqq3r3n6R169Y4++yz8eabb6KmpqbO8/Pnz0f79u3Rp08fALznT8cGG/9mYOHChWjXrh3OOOMMo9f95z//wYgRI3D22Wdj4cKFiI6OFpebN28ezj33XGRmZuLqq69GeXk5Fi5ceMj1Dx8+HB6Pp86XcGVlJRYuXIgrrrgCNptNfO2sWbNwxx13oEuXLpg2bRoeeugh9OjRAytXrjTaxgNmzJiB7Oxs/O///i+eeuoptG7dGrfffnuwgTlcnTt3xiuvvBLyePbZZ2G325Genh5cbs6cOYiLi8OECRPw9NNPo2fPnpg0aVLwci4A3H///ejRowdSU1OD66rv/v/HH3+MQYMGoaioCA8++CAmTJiAr7/+Gv369RMTzq666iqUl5djypQpuOqqqzBnzhw89NBD9W7ftGnTcPnllwf32SuvvBK8vPzggw9i7NixyMrKwlNPPYVhw4bh+eefxwUXXFCnUSsuLsZFF12EHj16YNq0aTj33HPrLdfv9+Oiiy5C69at8cQTTyAnJwfjxo3DnDlzcOGFF6JXr154/PHHER8fj5EjR2Lz5s3B1xYWFuKMM87A0qVLcfvtt+ORRx5BdXU1/vSnP+Htt98OLldVVYUBAwZg6dKlGDduHO6//3588cUXuOeee+rU55NPPsHZZ5+NsrIyTJ48GY8++ihKSkpw3nnnYdWqVfVui2TEiBEoLi7G0qVLQ+I//fQT1qxZEzzrJzpmFDVppaWlCoC67LLLDvs12dnZKisrS0VFRan+/fsrj8ejXbawsFBFRUWpWbNmBWNnnHHGYZUXCARUy5Yt1bBhw0Lir7/+ugKgPv/882DsnHPOUeecc07w78suu0x17dq13vWPGjVKZWdn14lPnjxZHXxoV1ZW1llu0KBBql27diGxg+uxefNmBUDNnj1brEMgEFCXXnqpiouLUz///HO95d16660qJiZGVVdXB2OXXHKJuA1SuT169FDp6emquLg4GPvhhx+U1WpVI0eODMYObP8NN9wQss7LL79cpaSkiNvxRwdev3v37mCsqKhIORwOdcEFFyi/3x+M//Of/1QA1EsvvRSMnXPOOQqAmjlz5iHLUmr/+whAPfroo8HYvn37VHR0tLJYLGrBggXB+Nq1axUANXny5GBs/PjxCoD64osvgrHy8nLVtm1blZOTE6zvtGnTFAD1+uuvB5fzeDyqQ4cOCoD69NNPlVL739MTTjhBDRo0SAUCgeCylZWVqm3btur8888PxmbPnq0AqM2bN9e7jXv37lVOp1Ndc801IfH77rtPAVDr1q0LxqTj94Ddu3fX2X6iI8Ez/yaurKwMABAfH2/0ur1796K2thatWrXSnvED+/MBrFYrhg0bFoxdc801+OCDD7Bv3756y7BYLLjyyiuxePFiVFRUBOOvvfYaWrZsiTPPPFP72sTERGzfvh3ffPONwVbp/XEbS0tLsWfPHpxzzjnYtGkTSktLj3i9f/vb3/D+++9jzpw56NKli1heeXk59uzZg7POOguVlZVYu3atcTm7du3C6tWrcf311yM5OTkY79atG84//3wsXry4zmtuu+22kL/POussFBcXB48ZEx9//DF8Ph/Gjx8Pq/W/Xxs333wz3G53nas7TqcTo0ePNirjpptuCv47MTERHTt2RGxsLK666qpgvGPHjkhMTMSmTZuCscWLF6N3794hx1NcXBxuueUWbNmyBb/88ktwuRYtWuCKK64ILhcTE4NbbrklpB6rV6/G+vXrce2116K4uBh79uzBnj174PF4MGDAAHz++efGPV6SkpJw8cUX47333oPH4wEAKKWwYMEC9OrVCyeeeKLR+ogaio1/E+d2uwHsb2BMDBgwAGPGjMGrr75aJ9P8j1599VX07t0bxcXF2LBhAzZs2IBTTjkFPp8Pb7zxxiHLGT58OKqqqvDee+8B2J+tvHjxYlx55ZX13te89957ERcXh969e+OEE07A2LFjj+h+6wFfffUVBg4cGLxXnpaWFrwPfaSN/5IlS/DQQw9h4sSJIT+OAODnn3/G5ZdfjoSEBLjdbqSlpQUTtI6kvK1btwLY3/gdrHPnzsHG6Y/atGkT8ndSUhIAHPJHm0n5DocD7dq1Cz5/QMuWLY2SOF0uF9LS0kJiCQkJaNWqVZ3jJCEhIWQbtm7dqt0vf6z71q1b0aFDhzrrO/i169evBwCMGjUKaWlpIY9//etf8Hq9R/QejhgxIpg8CwBff/01tmzZwkv+FBbM9m/i3G43srKysGbNGuPX/vOf/8S+ffvwzDPPICkpKSSxC9j/JXjgzPuEE06o8/p58+bVOWs62Omnn46cnBy8/vrruPbaa7Fw4UJUVVVh+PDh9b6uc+fOWLduHd5//30sWbIE//nPf/Dcc89h0qRJwfvWuh8Pf0yqA/aPfzBgwAB06tQJU6dORevWreFwOLB48WL84x//OKJxCzZv3owRI0bg/PPPx9///veQ50pKSnDOOefA7Xbj4YcfRvv27eFyufDdd9/h3nvvPWbjJOjyKVQ9/c0bS31XkyS6uoZjGw68P08++SR69OghLhMXF2e83ksvvRQJCQmYP38+rr32WsyfPx82mw1XX311Q6pLdETY+DcDl156KV544QUsX74cffv2PezXWa1WvPzyyygtLcVDDz2E5ORk3HHHHcHn582bB7vdHhz854++/PJLPPPMM9i2bVudM8yDXXXVVXj66adRVlaG1157DTk5OTj99NMPWb/Y2FgMHz4cw4cPh8/nw9ChQ/HII49g4sSJcLlcSEpKqpNpDqDOWejChQvh9Xrx3nvvhdT1008/PWQdJFVVVRg6dCgSExPx73//O+QyOLC/R0BxcTHeeuutkK5df0xSO+Bws7qzs7MBAOvWravz3Nq1a5GamorY2FiTzTDyx/LbtWsXjPt8PmzevLnRu36ayM7O1u6XA88f+P+aNWuglArZ7we/tn379gD2/7BuzO1yOp244oor8PLLL6OwsBBvvPEGzjvvPGRmZjZaGUSHi5f9m4F77rkHsbGxuOmmm1BYWFjn+Y0bN+Lpp58WX2u32/Hmm2+iX79+GD9+PF555ZXgc/PmzcNZZ52F4cOH44orrgh53H333QD291E+lOHDh8Pr9WLu3LlYsmRJyD1cneLi4pC/HQ4HunTpAqVUMLO8ffv2KC0txY8//hhcbteuXSEZ3sB/zx7/eLZYWlqK2bNnH7Iekttuuw2//fYb3n777eCl9EOV5/P58Nxzz9VZNjY29rAuIbdo0QI9evTA3LlzQ37wrFmzBh9++CEuvvjiI9iSwzdw4EA4HA4888wzIdv14osvorS0FJdccslRLb8+F198MVatWoXly5cHYx6PBy+88AJycnKCuRgXX3wxdu7ciTfffDO4XGVlZZ3Br3r27In27dvj//7v/0JyVQ7YvXv3Edd1xIgRqKmpwa233ordu3fzkj+FDc/8m4H27dtj/vz5GD58ODp37hwywt/XX3+NN954A9dff7329TExMVi0aBHOOecc3HDDDUhISEBGRgY2bNiAcePGia9p2bIlTj31VMybNw/33ntvvfU79dRT0aFDB9x///3wer2HvOQPABdccAEyMzODw57++uuv+Oc//4lLLrkkmNx49dVX495778Xll1+OO+64A5WVlZgxYwZOPPFEfPfddyHrcjgcGDx4MG699VZUVFRg1qxZSE9Px65duw5Zlz9atGgRXn75ZQwbNgw//vhjyA+PuLg4DBkyBGeccQaSkpIwatQo3HHHHbBYLHjllVfES9U9e/bEa6+9hgkTJuC0005DXFwcBg8eLJb95JNP4qKLLkLfvn1x4403oqqqCs8++ywSEhLq3LJpbGlpaZg4cSIeeughXHjhhfjTn/6EdevW4bnnnsNpp50W1gFn7rvvPvz73//GRRddhDvuuAPJycmYO3cuNm/ejP/85z/BKzM333wz/vnPf2LkyJHIy8tDixYt8Morr9QZfMhqteJf//oXLrroInTt2hWjR49Gy5YtsWPHDnz66adwu92H1dVVcs4556BVq1Z49913ER0dHTJKH9ExFbZ+BtTofvvtN3XzzTernJwc5XA4VHx8vOrXr5969tlnQ7qXZWdnq0suuaTO6wsKClSHDh2Uy+VSJ598sgKgNm7cqC3vwQcfVADUDz/8cMi63X///QqA6tChg/j8wV3snn/+eXX22WerlJQU5XQ6Vfv27dXdd9+tSktLQ1734YcfqpNOOkk5HA7VsWNH9eqrr4pdpd577z3VrVs35XK5VE5Ojnr88cfVSy+9VKeb1qG6+h3o2iU9/thl76uvvlKnn366io6OVllZWeqee+5RS5cuDelSppRSFRUV6tprr1WJiYkh69B1Mfz4449Vv379VHR0tHK73Wrw4MHql19+CVlG6qr3x7ofqlua7vVK7e/a16lTJ2W321VGRoYaM2aM2rdvX8gy55xzziG7af7RqFGjVGxsbJ24bj3S8btx40Z1xRVXqMTEROVyuVTv3r3V+++/X+e1W7duVX/6059UTEyMSk1NVXfeeadasmRJnfdFKaW+//57NXTo0OAxmJ2dra666iqVm5sbXOZw9+kf3X333QqAuuqqq8Tn2dWPjgWLUscg+4eIiIiOG7znT0REFGHY+BMREUUYNv5EREQRho0/ERFRmHz++ecYPHgwsrKyYLFY8M477xzyNcuWLcOpp54Kp9OJDh061Jn983Cw8SciIgoTj8eD7t27H/YMo5s3b8Yll1yCc889F6tXr8b48eNx00031Zkx8lCY7U9ERHQcsFgsePvttzFkyBDtMvfeey8WLVoUMqT71VdfjZKSEixZsuSwywr7ID87duzAvffeiw8++ACVlZXo0KEDZs+ejV69egHYP0ra5MmTMWvWLJSUlKBfv36YMWOGONa8JBAIYOfOnYiPjz/soVSJiOj4oZRCeXk5srKy6gyn3Viqq6vh8/kaZV3qoCGkgf3DOzudzgave/ny5XWGnR40aFC9E7RJwtr479u3D/369cO5556LDz74AGlpaVi/fn3IkKlPPPEEnnnmGcydOxdt27bFAw88gEGDBuGXX36By+U6ZBk7d+5E69atj+ZmEBHRMZCfn49WrVo1+nqrq6vRtm1LFBTsbZT1xcXF1RkaevLkyY0yEmdBQQEyMjJCYhkZGSgrK0NVVdVhT6oV1sb/8ccfR+vWrUPGWG/btm3w30opTJs2DX/9619x2WWXAQBefvllZGRk4J133jms2bAODAXbM/4GRFlCpxiNw6F/PPxRBarFeJJVnlDFZZFnJHNFmf9yjbfL8Z8q5LnZ4y3ytple+4iPkgsuq60R49FWeZsTHHJcp9TnF+OVgVoxHmOVD+VUl1m5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akK2uVOnTmjTpk2z2Ga/348FCxbA4/Ggb9++zXp7x44di0suuSRk24Dm+x6vX78eWVlZaNeuHUaMGIFt27YBaL7b+95776FXr1648sorkZ6ejlNOOQWzZs0KPh8J31/h0uwb/z179sDv99eZBTAjIwMFBQVhqtWxc2Abm+v2BwIBjB8/Hv369cNJJ50EYP82OxwOJCYmhizb1Lf5p59+QlxcHJxOJ2677Ta8/fbb6NKlS7Pd3gULFuC7777DlClT6jzXHLe5T58+mDNnDpYsWYIZM2Zg8+bNOOuss1BeXt4stxcANm3ahBkzZuCEE07A0qVLMWbMGNxxxx2YO3cugOb//RVOx/2UvkT1GTt2LNasWRNyb7S56tixI1avXo3S0lK8+eabGDVqFD777LNwV+uoyM/Px5133omPPvrouJ8atbFcdNFFwX9369YNffr0QXZ2Nl5//XVER0eHsWZHTyAQQK9evfDoo48CAE455RSsWbMGM2fOxKhRo8Jcu+at2Z/5p6amwmaz1cmKLSwsRGZmZphqdewc2MbmuP3jxo3D+++/j08//RStWrUKxjMzM+Hz+VBSUhKyfFPfZofDgQ4dOqBnz56YMmUKunfvjqeffrpZbm9eXh6Kiopw6qmnIioqClFRUfjss8/wzDPPICoqChkZGc1umw+WmJiIE088ERs2bGiW7zEAtGjRAl26dAmJde7cOXi7ozl/f4Vbs2/8HQ4Hevbsidzc3GAsEAggNzcXffv2DWPNjo22bdsiMzMzZPvLysqwcuXKJrv9SimMGzcOb7/9Nj755BO0bds25PmePXvCbreHbPO6deuwbdu2JrvNkkAgAK/X2yy3d8CAAfjpp5+wevXq4KNXr14YMWJE8N/NbZsPVlFRgY0bN6JFixbN8j0GgH79+tXppvvbb78hOzsbQPP8/jpuhDvj8FhYsGCBcjqdas6cOeqXX35Rt9xyi0pMTFQFBQXhrlqjKC8vV99//736/vvvFQA1depU9f3336utW7cqpZR67LHHVGJionr33XfVjz/+qC677DLVtm1bVVVVFeaaH5kxY8aohIQEtWzZMrVr167go7KyMrjMbbfdptq0aaM++eQT9e2336q+ffuqvn37hrHWDXPfffepzz77TG3evFn9+OOP6r777lMWi0V9+OGHSqnmt72SP2b7K9X8tvn//b//p5YtW6Y2b96svvrqKzVw4ECVmpqqioqKlFLNb3uVUmrVqlUqKipKPfLII2r9+vVq3rx5KiYmRr366qvBZZrb99fxIiIaf6WUevbZZ1WbNm2Uw+FQvXv3VitWrAh3lRrNp59+qgDUeYwaNUoptb+7zAMPPKAyMjKU0+lUAwYMUOvWrQtvpRtA2lYAavbs2cFlqqqq1O23366SkpJUTEyMuvzyy9WuXbvCV+kGuuGGG1R2drZyOBwqLS1NDRgwINjwK9X8tldycOPf3LZ5+PDhqkWLFsrhcKiWLVuq4cOHqw0bNgSfb27be8DChQvVSSedpJxOp+rUqZN64YUXQp5vbt9fxwtO6UtERBRhmv09fyIiIgrFxp+IiCjCsPEnIiKKMGz8iYiIIgwbfyIiogjDxp+IiCjCsPEnIiKKMGz8iYiIIgwbf6Kj5Prrr8eQIUPCXY0j0rVrV3z44YcAgAsuuAAvv/xymGtERI2JjT9RBPH5fIdcpqSkBL/99htOP/10+P1+LF++HP369TsGtSOiY4WNP1GYTJ06FSeffDJiY2PRunVr3H777aioqAAAeDweuN1uvPnmmyGveeeddxAbG4vy8nIA++e9v+qqq5CYmIjk5GRcdtll2LJlS3D5A1cfHnnkEWRlZaFjx46HrNeKFSvQtWtXuN1urF69GrGxsWjfvn3jbTgRhR0bf6IwsVqteOaZZ/Dzzz9j7ty5+OSTT3DPPfcAAGJjY3H11Vdj9uzZIa+ZPXs2rrjiCsTHx6OmpgaDBg1CfHw8vvjiC3z11VeIi4vDhRdeGHKGn5ubi3Xr1uGjjz7C+++/r61Pt27dkJiYiKFDh+Lnn39GYmIizj77bOzZsweJiYno1q3b0dkRRHTshXtmIaLmatSoUeqyyy477OXfeOMNlZKSEvx75cqVymazqZ07dyqllCosLFRRUVFq2bJlSimlXnnlFdWxY0cVCASCr/F6vSo6OlotXbo0WIeMjAzl9XoPWX5+fr7avHmz6tq1q5o1a5bavHmzOvfcc9Xjjz+uNm/erPLz8w97W4jo+MYzf6Iw+fjjjzFgwAC0bNkS8fHx+POf/4zi4mJUVlYCAHr37o2uXbti7ty5AIBXX30V2dnZOPvsswEAP/zwAzZs2ID4+HjExcUhLi4OycnJqK6uxsaNG4PlnHzyyXA4HIesT6tWreByubBx40ZcffXVaNGiBb755htcc801yMnJQatWrY7CXiCicGDjTxQGW7ZswaWXXopu3brhP//5D/Ly8jB9+nQAoUl5N910E+bMmQNg/yX/0aNHw2KxAAAqKirQs2dPrF69OuTx22+/4dprrw2uIzY29pD1ue222xAXF4d27drB6/UiMzMTycnJqKioQOfOnREXF4dt27Y14h4gonBi408UBnl5eQgEAnjqqadw+umn48QTT8TOnTvrLHfddddh69ateOaZZ/DLL79g1KhRwedOPfVUrF+/Hunp6ejQoUPIIyEhwag+Dz/8MFavXo1LL70U48ePx+rVq3H99dfjuuuuC/6oyMrKavB2E9HxgY0/0VFUWlpa58w8Pz8fHTp0QE1NDZ599lls2rQJr7zyCmbOnFnn9UlJSRg6dCjuvvtuXHDBBSGX3keMGIHU1FRcdtll+OKLL7B582YsW7YMd9xxB7Zv325UzwM/IH788UcMHjwYHTp0wK+//oqLL744+IMiKiqqwfuDiI4PbPyJjqJly5bhlFNOCXk89NBD6N69O6ZOnYrHH38cJ510EubNm4cpU6aI67jxxhvh8/lwww03hMRjYmLw+eefo02bNhg6dCg6d+6MG2+8EdXV1XC73cZ1LSgowObNm3H66afD5/NhxYoVwfwCImpeLEopFe5KEJHeK6+8grvuugs7d+48rMQ9IqJD4XU8ouNUZWUldu3ahcceewy33norG34iajS87E90nHriiSfQqVMnZGZmYuLEieGuDhE1I7zsT0REFGF45k9ERBRh2PgTERFFGDb+REREEYaNPxERUYRh409ERBRh2PgTERFFGDb+REREEYaNPxERUYRh409ERBRh/j/UYdN7T2oPnAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "print(\"The parameter cka_internal has value\", cka_internal)\n",
    "if cka_internal:\n",
    "    _, n_layers = CKA.find_layers(model1, model1_type)\n",
    "    cka         = CKA.MinibatchCKA(n_layers, None, across_models=False)\n",
    "    heatmap     = compute_cka_internal(model1, loader, cka, model1_type) \n",
    "else:\n",
    "    _, n_layers1 = CKA.find_layers(model1, model1_type)\n",
    "    _, n_layers2 = CKA.find_layers(model2, model2_type)\n",
    "\n",
    "    cka          = CKA.MinibatchCKA(n_layers1, n_layers2, across_models=True)\n",
    "    heatmap      = compute_cka_across(model1, model2, loader1, loader2, cka, model1_type, model2_type)\n",
    "\n",
    "heatmap = heatmap.numpy()\n",
    "plt.imshow(heatmap, cmap = 'magma', origin = 'lower')\n",
    "plt.xlabel('Layer #')\n",
    "plt.ylabel('Layer #')\n",
    "if cka_internal:\n",
    "    if model1_type == 'mlp':\n",
    "        tit = 'MLP'\n",
    "    if model1_type == 'cnn':\n",
    "        tit = 'CNN'\n",
    "    if model1_type == 'vit':\n",
    "        tit = 'ViT'\n",
    "    plt.title(f'CKA visualization for model {tit}')\n",
    "else:\n",
    "    plt.title(f'CKA across model {model1_type} and {model2_type}')\n",
    "plt.colorbar()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ffcv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
