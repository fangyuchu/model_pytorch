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
from sklearn.decomposition import PCA           #加载PCA算法包


a=np.array([[[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]],[[19,20,21],[22,23,24],[25,26,27]],[[28,29,30],[31,32,33],[34,35,36]]]])
b=np.swapaxes(a[0],0,1)
c=np.swapaxes(b,1,2)
d=np.resize(c,(9,4))
pca=PCA(n_components=2)     #加载PCA算法，设置降维后主成分数目为2
e=pca.fit_transform(d)#对样本进行降维
f=np.resize(e,(3,3,2))
g=np.swapaxes(f,1,2)
h=np.swapaxes(g,0,1)

print(h)