import torch
from framework import data_loader, train, evaluate
from network import vgg_channel_weight, vgg

#
# c=torch.load('/home/disk_new/model_saved/vgg16_bn_weighted_channel/checkpoint/flop=18923530,accuracy=0.93600.tar')
#
# network=c['network']
# network.load_state_dict(c['state_dict'])
# for mod in network.features:
#     if isinstance(mod,nn.Conv2d):
#         print()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net= vgg.vgg16_bn(pretrained=True)
print()

for mod in net.modules():
    if isinstance(mod,torch.nn.Conv2d):
        weight=mod.weight.detach()
        c=torch.ones(5)
        a=torch.sum(weight)+torch.sum(c)
        b=a*2
        b.backward()
        mod.weight[0,0,0,0]=10000
        print()




checkpoint=torch.load('./data/model_saved/reform_vgg16_bn/checkpoint/flop=530442,accuracy=0.94060.tar')
net= vgg_channel_weight.vgg16_bn(pretrained=False, dataset='cifar10').to(device)
net.load_state_dict(checkpoint['state_dict'])


net.train_channel_weight(if_train=False)
net.prune_channel_weight(percent=[0 for i in range(13)])
net.to(device)

# print()


checkpoint=torch.load('./data/baseline/vgg16_bn_cifar10,accuracy=0.941.tar')
net=checkpoint['net']

net.load_state_dict(checkpoint['state_dict'])

vgg_channel_weight.reform_net(net)
net.to(device)
# evaluate.evaluate_net(network,data_loader=data_loader.create_validation_loader(batch_size=512,num_workers=8,dataset_name='cifar10'),save_net=False)



train.train(net=net,
            net_name='reform_vgg16_bn',
            dataset_name='cifar10',
            learning_rate=0.01,
            num_epochs=250,
            batch_size=256,
            checkpoint_step=4000,
            load_net=True,
            test_net=False,
            num_workers=4,
            learning_rate_decay=True,
            learning_rate_decay_factor=0.1,
            learning_rate_decay_epoch=[50,100,150,200],
            # criterion=vgg_channel_weight.CrossEntropyLoss_weighted_channel(network=network,penalty=1e-5))
            criterion=vgg_channel_weight.CrossEntropyLoss_weighted_channel(net=net, penalty=1e-1, piecewise=4)
            # criterion=nn.CrossEntropyLoss()

            )

print()