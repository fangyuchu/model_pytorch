import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math



def create_train_loader(
                    dataset_path,
                    mean,
                    std,
                    batch_size,
                    num_workers,
                    dataset_name='imagenet',
                    default_image_size=224,
):
    if dataset_name is 'cifar10':
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
                    dataset_path,
                    mean,
                    std,
                    batch_size,
                    num_workers,
                    scale=0.875,
                    dataset_name='imagenet',
                    default_image_size=224,
):
    if dataset_name is 'cifar10':
        data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=dataset_path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),download=True),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(int(math.floor(default_image_size / scale))),
            transforms.CenterCrop(default_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        folder = datasets.ImageFolder(dataset_path, transform)
        data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return data_loader
