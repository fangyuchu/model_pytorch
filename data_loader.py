import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import config as conf



def create_train_loader(
                    batch_size,
                    num_workers,
                    dataset_path=None,
                    mean=None,
                    std=None,
                    dataset_name='',
                    default_image_size=224,
):
    if dataset_name is 'cifar10':
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
        if dataset_name is 'imagenet' and dataset_path is None:
            dataset_path=conf.imagenet['train_set_path']
            mean=conf.imagenet['mean']
            std=conf.imagenet['std']
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
                    mean=None,
                    std=None,
                    scale=0.875,
                    dataset_name='',
                    default_image_size=224,
):
    if dataset_name is 'cifar10':
        if dataset_path is None:
            dataset_path=conf.cifar10['validation_set_path']
        mean=conf.cifar10['mean']
        std=conf.cifar10['std']
        data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=dataset_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),download=True),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        if dataset_name is 'imagenet' and dataset_path is None:
            dataset_path=conf.imagenet['validation_set_path']
            mean=conf.imagenet['mean']
            std=conf.imagenet['std']
        transform = transforms.Compose([
            transforms.Resize(int(math.floor(default_image_size / scale))),
            transforms.CenterCrop(default_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        folder = datasets.ImageFolder(dataset_path, transform)
        data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return data_loader
