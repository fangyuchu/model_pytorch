import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import config as conf



def create_train_loader(
                    batch_size,
                    num_workers,
                    dataset_path=None,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    dataset_name='',
                    default_image_size=224,
):
    if dataset_name == 'cifar10':
        if dataset_path is None:
            dataset_path=conf.cifar10['train_set_path']
        mean=conf.cifar10['mean']
        std=conf.cifar10['std']
        data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=dataset_path, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]), download=True),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True)
    else:
        if dataset_name == 'imagenet' and dataset_path is None:
            dataset_path=conf.imagenet['train_set_path']
            mean=conf.imagenet['mean']
            std=conf.imagenet['std']
        if dataset_name == 'tiny_imagenet' and dataset_path is None:
            dataset_path=conf.tiny_imagenet['train_set_path']
            mean=conf.tiny_imagenet['mean']
            std=conf.tiny_imagenet['std']
        transform = transforms.Compose([
            transforms.RandomResizedCrop(default_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        folder = datasets.ImageFolder(dataset_path, transform)
        data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    return data_loader

def create_validation_loader(
                    batch_size,
                    num_workers,
                    dataset_path=None,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    scale=0.875,
                    dataset_name='',
                    default_image_size=224,
                    shuffle=False
):
    if dataset_name == 'cifar10':
        if dataset_path is None:
            dataset_path=conf.cifar10['validation_set_path']
        mean=conf.cifar10['mean']
        std=conf.cifar10['std']
        data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=dataset_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),download=True),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True)
    elif dataset_name == 'cifar10_trainset':                                                    #use train set as validation set
        if dataset_path is None:
            dataset_path=conf.cifar10['train_set_path']
        mean=conf.cifar10['mean']
        std=conf.cifar10['std']
        data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=dataset_path, train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),download=True),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True)
    else:
        if 'imagenet' in dataset_name :
            mean=conf.imagenet['mean']
            std=conf.imagenet['std']
            if dataset_name =='imagenet_trainset':
                dataset_path=conf.imagenet['train_set_path']
            if dataset_name is 'imagenet' and dataset_path is None:
                dataset_path=conf.imagenet['validation_set_path']
            if dataset_name == 'tiny_imagenet' and dataset_path is None:
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