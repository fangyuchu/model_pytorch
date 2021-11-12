from __future__ import print_function
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
from framework import config as conf
from framework.imagenet_loader import HybridTrainPipe,HybridValPipe,DALIDataloader
from PIL import Image
import os
import os.path
import numpy as np
import sys
from torch.utils.data.sampler import SubsetRandomSampler

def create_train_loader(
        batch_size,
        num_workers,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        dataset_name='',
        default_image_size=224,
        train_val_split_ratio='all',# ex.  0.1
):
    mean=getattr(conf,dataset_name)['mean']
    std = getattr(conf, dataset_name)['std']
    dataset_path=getattr(conf,dataset_name)['train_set_path']

    if dataset_name == 'cifar10':
        train_folder = datasets.CIFAR10(root=dataset_path, train=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std),
                                  ])
                                        , download=True)
        val_folder = datasets.CIFAR10(root=dataset_path, train=True,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean, std=std),
                                             ]), download=True)
    elif dataset_name == 'cifar100':
        train_folder = datasets.CIFAR100(root=dataset_path, train=True,
                                         transform=transforms.Compose([
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomCrop(32, 4),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std),
                                         ]), download=True)
        val_folder = datasets.CIFAR100(root=dataset_path, train=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std),
                                       ]), download=True)
    else:
        train_folder = datasets.ImageFolder(root=dataset_path,
                                            transform=transforms.Compose([
                                                transforms.RandomResizedCrop(default_image_size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=mean, std=std),
                                            ]))
        val_folder = datasets.ImageFolder(root=dataset_path,
                                          transform=transforms.Compose([
                                              transforms.Resize(int(math.floor(default_image_size / 0.875))),
                                              transforms.CenterCrop(default_image_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std),
                                          ]))

    if train_val_split_ratio =='all':
        train_loader=torch.utils.data.DataLoader(train_folder, batch_size=batch_size, num_workers=num_workers,pin_memory=True,shuffle=True)
        val_loader = None
    else:
        #create indices to split train and val set
        train_set_size = getattr(conf,dataset_name)['train_set_size']
        indices = list(range(train_set_size))
        split = int(np.floor(train_val_split_ratio * train_set_size))
        # np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(valid_idx)
        train_loader = torch.utils.data.DataLoader(train_folder, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,pin_memory=True)
        val_loader=torch.utils.data.DataLoader(val_folder, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,pin_memory=True)
    if dataset_name == 'imagenet':
        return get_imagenet_loader(dataset_path, batch_size, 'train', False),None
    return train_loader,val_loader

def create_test_loader(
        batch_size,
        num_workers,
        dataset_name,
        dataset_path=None,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=0.875,
        default_image_size=224,
        shuffle=False
):
    if 'cifar10' in dataset_name and 'cifar100' not in dataset_name:
        if dataset_path is None:
            dataset_path=conf.cifar10['test_set_path']
        mean=conf.cifar10['mean']
        std=conf.cifar10['std']
        if 'trainset' in dataset_name:
            dataset= datasets.CIFAR10(root=dataset_path, train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),download=True)
        else:
            dataset=datasets.CIFAR10(root=dataset_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]), download=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True)
    elif 'cifar100' in dataset_name:
        if dataset_path is None:
            dataset_path=conf.cifar100['test_set_path']
        mean=conf.cifar100['mean']
        std=conf.cifar100['std']
        if 'trainset' in dataset_name:
            dataset= datasets.CIFAR100(root=dataset_path, train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),download=True)
        else:
            dataset=datasets.CIFAR100(root=dataset_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]), download=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True)
    else:
        if 'imagenet' in dataset_name :
            mean=conf.imagenet['mean']
            std=conf.imagenet['std']
            if dataset_name =='imagenet_trainset':
                dataset_path=conf.imagenet['train_set_path']
            if dataset_name == 'imagenet' and dataset_path is None:
                dataset_path=conf.imagenet['test_set_path']
            if dataset_name == 'tiny_imagenet' and dataset_path is None:
                dataset_path = conf.tiny_imagenet['test_set_path']
            if dataset_name == 'tiny_imagenet_trainset':
                dataset_path = conf.tiny_imagenet['train_set_path']

        transform = transforms.Compose([
            transforms.Resize(int(math.floor(default_image_size / scale))),
            transforms.CenterCrop(default_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        folder = datasets.ImageFolder(dataset_path, transform)
        data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

        if dataset_name =='imagenet':
            return get_imagenet_loader(dataset_path,batch_size,'test',False)
    return data_loader


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'



def get_imagenet_loader(root, batch_size, type='train', mobile_setting=True):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=8, device_id=0,
                                    data_dir=root , crop=224, world_size=1, local_rank=0)
        train_loader = DALIDataloader(pipeline=pip_train, size=1281167, batch_size=batch_size,
                                      onehot_label=True)

        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        # folder = datasets.ImageFolder(conf.imagenet['train_set_path'], transform)
        # train_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=True,
        #                                           num_workers=8, pin_memory=True)

        return train_loader
    elif type =='test':
        pip_test = HybridValPipe(batch_size=batch_size, num_threads=4, device_id=0, data_dir=root ,
                                 crop=224, size=256, world_size=1, local_rank=0)

        test_loader = DALIDataloader(pipeline=pip_test, size=50000, batch_size=batch_size,
                                     onehot_label=True)
        return test_loader
    # #from pruning from scratch
    # crop_scale = 0.25 if mobile_setting else 0.08
    # jitter_param = 0.4
    # lighting_param = 0.1
    # if type == 'train':
    #     transform = transforms.Compose([
    #         transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
    #         transforms.ColorJitter(
    #             brightness=jitter_param, contrast=jitter_param,
    #             saturation=jitter_param),
    #         Lighting(lighting_param),
    #         transforms.RandomHorizontalFlip(),
    #     ])
    #
    # elif type == 'test':
    #     transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #     ])
    #
    # dataset = datasets.ImageFolder(root, transform)
    # # sampler = data.distributed.DistributedSampler(dataset)
    # if type == 'train':
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True,
    #         num_workers=4, pin_memory=True
    #     )
    #     return data_loader
    #
    # elif type == 'test':
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False,
    #         num_workers=4, pin_memory=True
    #     )
    #     return data_loader




