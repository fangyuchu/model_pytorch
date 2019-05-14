import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math



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
    data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
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
        transforms.Resize(int(math.floor(default_image_size / scale))),
        transforms.CenterCrop(default_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    folder = datasets.ImageFolder(dataset_path, transform)
    data_loader = torch.utils.data.DataLoader(folder, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return data_loader
