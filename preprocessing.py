import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.utils.data.dataset
import torchvision.transforms.v2 as transforms
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import InterpolationMode


class ContrastStretching(nn.Module):
    def __init__(self, p=1):
        super(ContrastStretching, self).__init__()
        self.p = p

    def forward(self, tensor: torch.Tensor):
        img = tensor.clone()
        if torch.rand(1).item() < self.p:
            min_val = img.min()
            max_val = img.max()
            img = (img - min_val) / (max_val - min_val) * 255
        
        return img

class UnsharpMasking(nn.Module):
    def __init__(self, sigma=3, kernel_size=5, multiplier=2, p=0.5):
        super(UnsharpMasking, self).__init__()
        self.sigma = sigma
        self.p = p
        self.kernel_size = kernel_size
        self.multiplier = multiplier

    def forward(self, tensor):
        image = tensor.clone()
        if torch.rand(1).item() < self.p:
            blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
            image = tensor + (tensor - blur(tensor)) * self.multiplier
        return image


class GaussianNoise(nn.Module):
    def __init__(self, mean=0.0, std=1.0,rate = 0.5):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.rate =rate
        self.mean = mean

    def forward(self, tensor: torch.Tensor):
        tmean = tensor.mean()
        tstd = tensor.std()
        tensor = (tensor  - tensor.mean())/tensor.std()
        output = tensor + torch.randn(tensor.size()) * self.rate + self.mean
        return output * tstd + tmean




class RandomBrightness(nn.Module):
    def __init__(self, p=1,rate = 0.5):
        super(RandomBrightness, self).__init__()
        self.p = p
        self.rate = rate

    def forward(self, tensor: torch.Tensor):
        img = tensor.clone()
        number = torch.float32
        if isinstance(self.rate,tuple):
            number = torch.FloatTensor(1).uniform_(self.rate[0],self.rate[1]).item()
        else:
            number = self.rate
        if torch.rand(1).item() < self.p:
            img  = img *  number
        
        return img


class SubsetAugmentation(Dataset):

    def __init__(self, subset: torch.utils.data.dataset.Subset):
        self.subset = subset

    def __getitem__(self, index):
        idx = index // 13
        resized_image, classid, classname, img_path = self.subset.__getitem__(idx)
        resized_image = resized_image * 255.0
        trans_image, augment_name = self.augment_image(resized_image, index)
        trans_image = trans_image / 255.0
        return trans_image, classid, classname, img_path

    def __len__(self):
        return 13 * len(self.subset)

    def augment_image(self, img: torch.Tensor, index: int) -> tuple[torch.Tensor, str]:
        v = index % 13
        transformation_list = [
            transforms.RandomRotation(
                degrees=90, interpolation=InterpolationMode.BILINEAR
            ),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            UnsharpMasking(p=0.95),
            ContrastStretching(),
            GaussianNoise(),
            RandomBrightness(rate=(0.4,1.5)),
            transforms.RandomErasing(
                p=0.95, scale=(0.02, 0.10), ratio=(0.3, 2.3), value=0
            ),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.95),
        ]

        img_tensor = img
        transformation_name = ""
        if v == 0:
            transformation_name = "Original"
            img_tensor = img_tensor
        elif v in range(1, 10):
            transformation = transformation_list[v - 1]
            transformation_name = transformation.__class__.__name__
            img_tensor = transformation(img_tensor)  
        elif v == 10:
            img_tensor = img_tensor.to(torch.uint8)
            transformation = transforms.RandomEqualize()
            transformation_name = transformation.__class__.__name__
            img_tensor = transformation(img_tensor)
            img_tensor = img_tensor.to(torch.float32)
        elif v == 11:
            sample = [ContrastStretching(),transforms.RandomRotation(degrees=90),UnsharpMasking(p=1.0),]
            transformation = transforms.Compose(sample)
            img_tensor = transformation(img_tensor)
            transformation_name = "Compose1"
         
        elif v == 12:
            sample  = [ContrastStretching(),transforms.RandomRotation(degrees=90),GaussianNoise(rate = 0.3)]
            transformation = transforms.Compose(sample)
            img_tensor = transformation(img_tensor)
            transformation_name = "Compose2"
            

        return img_tensor, transformation_name