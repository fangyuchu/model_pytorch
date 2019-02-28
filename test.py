import config as conf
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


a=[i for i in range(10)]
b=[2,3,4,5]
c=np.array(a,dtype=np.float32)
print(c[[i for i in range(len(a)) if i not in b]])