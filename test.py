import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import vgg
import os
from datetime import datetime
from prune import select_and_prune_filter
import numpy as np
import config as conf


pretrained = False
dataset_name = 'imagenet'
learning_rate = conf.learning_rate
num_epochs = conf.num_epochs
batch_size = conf.batch_size
checkpoint_step = conf.checkpoint_step
checkpoint_path = None
highest_accuracy_path = None
global_step_path = None
default_image_size = 224
momentum = conf.momentum
num_workers = conf.num_workers
percent_of_pruning = 0.3
ord = 2

#prepare the data
if dataset_name is 'imagenet':
    mean=conf.imagenet['mean']
    std=conf.imagenet['std']
    train_set_path=conf.imagenet['train_set_path']
    train_set_size=conf.imagenet['train_set_size']
    validation_set_path=conf.imagenet['validation_set_path']
# Data loading code
transform = transforms.Compose([
    transforms.RandomResizedCrop(default_image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,std=std),
])
train = datasets.ImageFolder(train_set_path, transform)
val = datasets.ImageFolder(validation_set_path, transform)
validation_loader = torch.utils.data.DataLoader(val, batch_size=256, shuffle=False, num_workers=num_workers)
i=0
for step, data in enumerate(validation_loader, 0):
    i+=1
    print(i)