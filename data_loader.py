import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import resnet
import vgg
import os
from datetime import datetime
import re
import math
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA           #加载PCA算法包
import evaluate


def create_train_loader(
                    dataset_path,
                    default_image_size,
                    mean,
                    std,
                    batch_size,
                    num_workers,
                    ):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(default_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    folder = datasets.ImageFolder(dataset_path, transform)
    data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

def create_validation_loader(
                    dataset_path,
                    default_image_size,
                    mean,
                    std,
                    batch_size,
                    num_workers,
                    scale=0.875
):
    transform = transforms.Compose([
        #transforms.RandomResizedCrop(default_image_size),
        #todo:这里错啦！！！！！！！！！！！！
        transforms.Resize(int(math.floor(max(default_image_size) / scale))),
        transforms.CenterCrop(max(default_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    folder = datasets.ImageFolder(dataset_path, transform)
    data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader