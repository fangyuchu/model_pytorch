from __future__ import print_function
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
from framework import config as conf
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
        validation_ratio=0.1,
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
                                  ]), download=True)
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

    #create indices to split train and val set
    train_set_size = getattr(conf,dataset_name)['train_set_size']
    indices = list(range(train_set_size))
    split = int(np.floor(validation_ratio * train_set_size))
    # np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(train_folder, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,pin_memory=True)
    val_loader=torch.utils.data.DataLoader(val_folder, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,pin_memory=True)
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
            dataset_path=conf.cifar10['dataset_path']
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
            dataset_path=conf.cifar100['dataset_path']
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
    return data_loader

class data_prefetcher():
    '''
    dl=data_loader.create_train_loader(dataset_name='imagenet',batch_size=1,num_workers=1)
    prefetcher=data_loader.data_prefetcher(dl)
    data, label = prefetcher.next()
    iteration = 0
    while data is not None:
        iteration += 1
        # 训练代码
        data, label = prefetcher.next()
    '''
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)



if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class CIFAR100(data.Dataset):
    #use train and validate set to train the network. the same as 'Where to Prune,using lstm ...'
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            # downloaded_list = self.train_list
            downloaded_list = self.train_list + self.test_list
        else:
            downloaded_list = self.test_list


        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str