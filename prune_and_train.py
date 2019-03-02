import config as conf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import resnet
import vgg
import os
import re
from datetime import datetime
from prune import select_and_prune_filter


def prune_and_train(
                    model_name,
                    pretrained=False,
                    dataset_name='imagenet',
                    learning_rate=conf.learning_rate,
                    num_epochs=conf.num_epochs,
                    batch_size=conf.batch_size,
                    checkpoint_step=conf.checkpoint_step,
                    checkpoint_path=None,
                    highest_accuracy_path=None,
                    global_step_path=None,
                    default_image_size=224,
                    momentum=conf.momentum,
                    num_workers=conf.num_workers,
                    percent_of_pruning=0.3,
                    ord=2,
                  ):
    #implemented according to "Pruning Filters For Efficient ConvNets" by Hao Li
    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ',end='')
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


    temp=re.search(r'(\d+)',model_name).span()[0]
    model=model_name[:temp]                                                     #name of the model.ex: vgg,resnet...
    del temp
    #define the model
    net=getattr(globals()[model],model_name)(pretrained=pretrained).to(device)

    num_conv=0                                                                  #num of conv layers in the net
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv+=1

    for i in range(1,num_conv+1):
        net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=percent_of_pruning, ord=ord)  # prune the model

    #define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=learning_rate
                          )  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

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
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if checkpoint_path is None:
        checkpoint_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/checkpoint'
    if highest_accuracy_path is None:
        highest_accuracy_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/highest_accuracy.txt'
    if global_step_path is None:
        global_step_path=conf.root_path+model_name+','+str(percent_of_pruning)+'pruned'+'/global_step.txt'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)

    highest_accuracy = 0
    if os.path.exists(highest_accuracy_path):
        f = open(highest_accuracy_path, 'r')
        highest_accuracy = float(f.read())
        f.close()
        print('highest accuracy from previous training is %f' % highest_accuracy)
    global_step=0
    if os.path.exists(global_step_path):
        f = open(global_step_path, 'r')
        global_step = int(f.read())
        f.close()
        print('global_step at present is %d' % global_step)
        model_saved_at=checkpoint_path+'/global_step='+str(global_step)+'.pth'
        print('load model from'+model_saved_at)
        net.load_state_dict(torch.load(model_saved_at))
    else:
        print('{} test the model after pruned'.format(datetime.now()))                      #no previous checkpoint
        with torch.no_grad():                                                               #test the firstly pruned model
            correct = 0
            total = 0
            for val_data in validation_loader:
                net.eval()
                images, labels = val_data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            correct = float(correct.cpu().numpy().tolist())
            accuracy = correct / total
            print("{} Validation Accuracy after pruned = {:.4f}".format(datetime.now(), accuracy))

    print("{} Start training ".format(datetime.now())+model_name+"...")
    for epoch in range(math.floor(global_step*batch_size/train_set_size),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            if global_step % math.ceil(train_set_size / batch_size)==0:
                break

            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            global_step += 1

            if step % checkpoint_step == 0 and step != 0:
                print("{} Start validation".format(datetime.now()))
                print("{} global step = {}".format(datetime.now(), global_step))
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for val_data in validation_loader:
                        net.eval()
                        images, labels = val_data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    correct = float(correct.cpu().numpy().tolist())
                    accuracy = correct / total
                    print("{} Validation Accuracy = {:.4f}".format(datetime.now(), accuracy))
                    if accuracy>highest_accuracy:
                        highest_accuracy=accuracy
                        #save model
                        print("{} Saving model...".format(datetime.now()))
                        torch.save(net.state_dict(), '%s/global_step=%d.pth' % (checkpoint_path, global_step))
                        print("{} Model saved ".format(datetime.now()))
                        #save highest accuracy
                        f = open(highest_accuracy_path, 'w')
                        f.write(str(highest_accuracy))
                        f.close()
                        #save global step
                        f=open(global_step_path,'w')
                        f.write(str(global_step))
                        print("{} model saved at global step = {}".format(datetime.now(), global_step))
                        f.close()
                    print('{} continue training'.format(datetime.now()))


if __name__ == "__main__":
    prune_and_train(model_name='vgg16_bn',pretrained=True,checkpoint_step=5000,percent_of_pruning=0.1,num_epochs=20)