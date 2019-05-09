import pretrainedmodels
import torch
import train
import config as conf
import torchvision.datasets as datasets
import torchvision.transforms as transforms


print(pretrainedmodels.model_names)
model_name = 'vgg16_bn' # could be fbresnet152 or inceptionresnetv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').to(device)
model.eval()

mean = conf.imagenet['mean']
std = conf.imagenet['std']
train_set_path = conf.imagenet['train_set_path']
train_set_size = conf.imagenet['train_set_size']
#validation_set_path = conf.imagenet['validation_set_path']
validation_set_path='/home/victorfang/Desktop/imagenet所有数据/imagenet_validation_new'

#validation_loader=train.create_data_loader(validation_set_path,224,mean,std,conf.batch_size,conf.num_workers)

validation_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(validation_set_path, transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])),
    batch_size=conf.batch_size, shuffle=True,
    num_workers=conf.num_workers, pin_memory=True)

train.evaluate_net(model,validation_loader,save_net=False)