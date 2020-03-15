import torch.nn as nn
from network import resnet,resnet_cifar,resnet_tinyimagenet,vgg
from network.net_with_predicted_mask import predicted_mask_net
import re
from prune import prune_module
import torch


def get_net_information(net,dataset_name,net_name):
    '''

    :param net:
    :param dataset_name:
    :param net_name:
    :return:
    '''
    checkpoint = {}
    checkpoint['net_name']=net_name
    checkpoint['dataset_name']=dataset_name
    checkpoint['state_dict']=net.state_dict()
    structure=[]                                                                                #number of filters for each conv
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            structure+=[mod.out_channels]
    checkpoint['structure']=structure

    if isinstance(net,predicted_mask_net):
        checkpoint['feature_len']=net.feature_len
        checkpoint['gcn_rounds']=net.gcn_rounds
    # try:
    #     #for predicted_mask_net
    #     #isinstanec(net,predicted_mask_met) is False. I don't know why
    #     checkpoint['feature_len'] = net.feature_len
    #     checkpoint['gcn_rounds'] = net.gcn_rounds
    # except AttributeError:
    #     pass

    return checkpoint

def restore_net(checkpoint,pretrained=True,data_parallel=False,transformed_net=False):
    structure=checkpoint['structure']
    dataset_name=checkpoint['dataset_name']
    net_name=checkpoint['net_name']

    # define the network
    if 'vgg' in net_name:
        net = getattr(globals()['vgg'], net_name)(pretrained=False,dataset_name=dataset_name)
    elif 'resnet' in net_name:
        if 'imagenet' == dataset_name:
            net = getattr(globals()['resnet'], net_name)(pretrained=False)
        elif 'tiny_imagenet' == dataset_name:
            net = getattr(globals()['resnet'], net_name)(pretrained=False,num_classes=200)
        elif 'cifar10' == dataset_name:
            net = getattr(globals()['resnet_cifar'], net_name)()
        elif 'cifar100'==dataset_name:
            net = getattr(globals()['resnet_cifar'], net_name)(num_classes=100)
        else:
            raise Exception('Please input right dataset_name.')
    else:
        raise Exception('Unsupported net type:'+net_name)

    #prune the network according to checkpoint['structure']
    num_layer=0
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            index=[i for i in range(mod.out_channels-structure[num_layer])]
            if 'vgg' in net_name:
                net= prune_module.prune_conv_layer_vgg(model=net, layer_index=num_layer, filter_index=index)
            elif 'resnet' in net_name:
                net=prune_module.prune_conv_layer_resnet(net=net,
                                                         layer_index=num_layer,
                                                         filter_index=index,
                                                         )
            num_layer+=1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    if transformed_net is True:
        t_net=predicted_mask_net(net,net_name,dataset_name,checkpoint['feature_len'],checkpoint['gcn_rounds'])
        net=t_net

    if pretrained:
        try:
            net.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            net=nn.DataParallel(net)
            net.load_state_dict(checkpoint['state_dict'])
            net=net._modules['module']
    if data_parallel:
        net=nn.DataParallel(net)
    return net


# def conversion(dataset_name,net_name,checkpoint_path='',checkpoint=None):

#转化之前的cifar上的resnet
#     checkpoint=torch.load(checkpoint_path)
#     # net=checkpoint.pop('net')
#     net=resnet_cifar.resnet32()
#     checkpoint['state_dict']['fc.weight']=checkpoint['state_dict'].pop('linear.weight')
#     checkpoint['state_dict']['fc.bias']=checkpoint['state_dict'].pop('linear.bias')
#
#     net.load_state_dict(checkpoint['state_dict'])
#     checkpoint.update(get_net_information(net,dataset_name,net_name))
#     torch.save(checkpoint,checkpoint_path)


if __name__ == "__main__":
    # conversion(checkpoint_path='../data/baseline/resnet32_cifar10,accuracy=0.92380.tar',net_name='resnet56',dataset_name='cifar10')


    checkpoint=torch.load('/home/victorfang/model_pytorch/data/baseline/resnet56_cifar10,accuracy=0.94230.tar')

    net=resnet_cifar.resnet56()
    # net.load_state_dict(checkpoint['state_dict'])
    c=get_net_information(net=net,dataset_name='cifar10',net_name='resnet56')
    net=restore_net(checkpoint,True)
    from framework import evaluate,data_loader
    evaluate.evaluate_net(net,data_loader.create_validation_loader(512,4,'cifar10'),False)
    # c=get_net_information(net=net,dataset_name=dataset_name,net_name='resnet50')




