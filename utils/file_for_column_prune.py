import os,sys
sys.path.append('../')
import torch
from torch import nn
import torch.optim as optim
from framework import evaluate,data_loader,measure_flops,train
from network import vgg,storage,net_with_predicted_mask,resnet_cifar,resnet_cifar,resnet,mobilenet
from framework import config as conf
import logger
from network.modules import conv2d_with_mask_and_variable_shortcut
from prune.prune_module import get_module
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset='imagenet'
net_type='resnet50'
# net_type ='mobilenet_v1'
dataset='cifar10'
net_type='vgg16_bn'


#todo 现在做的还不行的话就考虑把reg的权重增大，使得column mask迫近0
def new_forward(conv):
    if isinstance(conv, conv2d_with_mask_and_variable_shortcut):
        def lambda_forward(x):
            masked_weight = conv.weight.view(conv.weight.size(0), -1) * conv.column_mask
            x = nn.functional.conv2d(x, masked_weight.reshape(conv.weight.shape), conv.bias, conv.stride, conv.padding, conv.dilation,
                                       conv.groups)
            if torch.sum(conv.mask != 0) <= conv.add_shortcut_num:  # ratio of masked conv is large
                downsample = conv.downsample(input)  # add shortcut
                # add zero if num of output feature maps differentiate between conv and shortcut
                if downsample.size()[1] < x.size()[1]:  # downsample has less feature maps
                    downsample = nn.functional.pad(downsample, (0, 0, 0, 0, 0, x.shape[1] - downsample.shape[1]))

                elif downsample.size()[1] > x.size()[1]:
                    x = nn.functional.pad(x, (0, 0, 0, 0, 0, downsample.shape[1] - x.shape[1]))
                x = x + downsample
            else:  # shortcut will not be added if only few filters are pruned
                pass
            return x

    elif isinstance(conv, nn.Conv2d):
        def lambda_forward( x ):
            masked_weight = conv.weight.view(conv.weight.size(0), -1)*conv.column_mask
            x = nn.functional.conv2d(x, masked_weight.reshape(conv.weight.shape), conv.bias, conv.stride, conv.padding, conv.dilation,
                                       conv.groups)
            return x
    return lambda_forward



def trasform_to_column_prune_conv(net):
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            mod.column_mask = nn.Parameter(torch.ones(mod.kernel_size[0] * mod.kernel_size[0] * mod.in_channels),requires_grad=True) # add column mask
            mod.forward=new_forward(mod)    #refactor the forward
            _modules = get_module(model=net, name=name)
            _modules[name.split('.')[-1]] = mod

def regularizer_func(net,writer,global_step):
    mean = 0
    std = 0
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            mean = mean+torch.mean(mod.column_mask.abs())
            std = std+torch.std(mod.column_mask.abs())
    writer.add_scalar(tag='reg/column_mask_mean',
                      scalar_value=float(mean.detach()),
                      global_step=global_step)
    writer.add_scalar(tag='reg/column_mask_std',
                      scalar_value=float(std.detach()),
                      global_step=global_step)
    coefficient = 0.001
    if std ==0:
        std=0 # avoid the nan problem of StdBackward
    reg = coefficient * (mean-std)
    writer.add_scalar(tag='reg/column_mask_reg',
                      scalar_value=float(reg.detach()),
                      global_step=global_step)
    return reg
    # return 0

def mask_net(net,ratio=0.1):
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            _, mask_index = torch.topk(mod.column_mask, k=int(ratio*mod.column_mask.numel()), dim=0, largest=False)
            with torch.no_grad():
                index = torch.ones_like(mod.column_mask)
                index[mask_index] = 0
                mod.column_mask[:] = mod.column_mask * index # set the smallest masks to zero
                mod.column_mask.requires_grad=False # mask will not be trained after being masked


if dataset == 'cifar10':

    weight_decay = {'default':5e-4,'extractor':5e-4}
    momentum = {'default':0.9,'extractor':0.9}
    learning_rate={'default':0.0001,'column_mask':0.1}
    batch_size=128

    if net_type == 'vgg16_bn':
        checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_vgg16bn_predicted_mask_and_variable_shortcut_net_newinner_doubleschedule_80_12/checkpoint/flop=63805650,accuracy=0.93150.pth')
        checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_vgg16bn_predicted_mask_and_variable_shortcut_net_newinner_doubleschedule_70_13/checkpoint/flop=95213418,accuracy=0.93270.pth')
        net = checkpoint['net']
        net.load_state_dict(checkpoint['state_dict'])
        trasform_to_column_prune_conv(net)
        net.cuda()
        measure_flops.measure_model(net,dataset_name='cifar10')
        evaluate.evaluate_net(net,data_loader=data_loader.create_test_loader(batch_size=512,num_workers=2,dataset_name='cifar10'),save_net=False)

        # exp_name='gat_column_vgg16bn_cifar10_12_80_train_column_mask_0.001maskCoefficient_160epoch'
        # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # # save the output to log
        # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path, exist_ok=True)
        # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        #
        # train.train(net=net,
        #             net_name='vgg16_bn',
        #             exp_name=exp_name,
        #             description='',
        #             dataset_name='cifar10',
        #             weight_decay=weight_decay,
        #             momentum=momentum,
        #             learning_rate=learning_rate,
        #             num_epochs=160,
        #             learning_rate_decay=True,
        #             learning_rate_decay_epoch=[80,120],
        #             learning_rate_decay_factor=0.1,
        #             batch_size=batch_size,
        #             evaluate_step=5000,
        #             resume=False,
        #             test_net=True,
        #             num_workers=4,
        #             # weight_decay=5e-4,
        #             # learning_rate_decay=False,
        #             top_acc=1,
        #             paint_loss=True,
        #             save_at_each_step=False,
        #             gradient_clip_value=None,
        #             regularizer_func=regularizer_func
        #             )



        # checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_column_vgg16bn_cifar10_12_80_train_column_mask/checkpoint/final_model_flop=63805650,accuracy=0.91180.pth')
        # checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_column_vgg16bn_cifar10_12_80_train_column_mask_0.001maskCoefficient/checkpoint/final_model_flop=63559466,accuracy=0.90840.pth')
        # checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_column_vgg16bn_cifar10_12_80_train_column_mask_0.0001maskCoefficient_160epoch/checkpoint/final_model_flop=63559466,accuracy=0.93020.pth')
        checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_column_vgg16bn_cifar10_12_80_train_column_mask_0.001maskCoefficient_160epoch/checkpoint/final_model_flop=63559466,accuracy=0.93000.pth')
        checkpoint = torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_column_vgg16bn_cifar10_13_pruned_train_90_biglr_wrong/checkpoint/flop=92365566,accuracy=0.93360.pth')
        net.load_state_dict(checkpoint['state_dict'])
        r=0.62
        mask_net(net, ratio=r)
        batch_size=64
        exp_name='gat_column_vgg16bn_cifar10_13_70_train_column_mask_0.001maskCoefficient_0.1masklr_160epoch'+'_'+str(r)+'masked_320epoch_finetune_bs64_88'

        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary

        print(exp_name)

        measure_flops.measure_model(net,dataset_name='cifar10')
        evaluate.evaluate_net(net, data_loader=data_loader.create_test_loader(batch_size=512, num_workers=2,dataset_name='cifar10'), save_net=False)
        train.train(net=net,
                    net_name='vgg16_bn',
                    exp_name=exp_name,
                    dataset_name='cifar10',
                    optimizer=optim.SGD,
                    weight_decay=weight_decay,
                    requires_grad={'default':True,'column_mask':False},
                    momentum=momentum,
                    learning_rate=0.01,
                    num_epochs=320,
                    batch_size=batch_size,
                    evaluate_step=5000,
                    resume=False,
                    test_net=False,
                    num_workers=4,
                    learning_rate_decay=True,
                    learning_rate_decay_epoch=[160,240],#[40,60],
                    learning_rate_decay_factor=0.1,
                    scheduler_name='MultiStepLR',
                    top_acc=1,
                    data_parallel=False,
                    paint_loss=False,
                    save_at_each_step=False,
                    use_tensorboard=True
                    # gradient_clip_value=gradient_clip_value
                    )
    if net_type == 'resnet18':
        net = resnet.resnet18(num_classes=10).cuda()
        prune_ratio=0.7
        i = 3
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='resnet18',
                                                                               dataset_name='cifar10',
                                                                               mask_update_epochs=20,
                                                                               mask_update_freq=1000,
                                                                               flop_expected=556651530*(1 - prune_ratio),
                                                                               mask_training_start_epoch=1,
                                                                               mask_training_stop_epoch=20,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=0.9
                                                                               )
        net = net.to(device)
        checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet18', str(i) + '.pth'),map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        net.mask_net()
        net.print_mask()
        net.prune_net()

        # checkpoint = torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/model_saved/gat_resnet18_doubleschedule_90_3/checkpoint/flop=52612434,accuracy=0.92960.pth')
        checkpoint = torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/model_saved/gat_resnet18_doubleschedule_70_3/checkpoint/flop=165712492,accuracy=0.94330.pth')
        net=net.net
        evaluate.evaluate_net(net, data_loader=data_loader.create_test_loader(batch_size=512, num_workers=2,dataset_name='cifar10'),save_net=False)
        net.load_state_dict(checkpoint['state_dict'])
        trasform_to_column_prune_conv(net)
        net.cuda()
        measure_flops.measure_model(net, dataset_name='cifar10')
        evaluate.evaluate_net(net, data_loader=data_loader.create_test_loader(batch_size=512, num_workers=2,dataset_name='cifar10'),save_net=False)

        exp_name='gat_column_resnet18_cifar10_'+str(i)+'_'+str(int(prune_ratio*100))+'_train_column_mask_0.001maskCoefficient_0.1masklr_160epoch'
        # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # # save the output to log
        # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path, exist_ok=True)
        # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        # print(exp_name,learning_rate)
        # train.train(net=net,
        #             net_name='resnet18',
        #             exp_name=exp_name,
        #             description='',
        #             dataset_name='cifar10',
        #             weight_decay=weight_decay,
        #             momentum=momentum,
        #             learning_rate=learning_rate,
        #             num_epochs=160,
        #             learning_rate_decay=True,
        #             learning_rate_decay_epoch=[80,120],
        #             learning_rate_decay_factor=0.1,
        #             batch_size=batch_size,
        #             evaluate_step=5000,
        #             resume=False,
        #             test_net=True,
        #             num_workers=4,
        #             # weight_decay=5e-4,
        #             # learning_rate_decay=False,
        #             top_acc=1,
        #             paint_loss=True,
        #             save_at_each_step=False,
        #             gradient_clip_value=None,
        #             regularizer_func=regularizer_func
        #             )


        checkpoint = torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/model_saved/gat_column_resnet18_cifar10_3_70_train_column_mask_0.001maskCoefficient_0.1masklr_160epoch/checkpoint/last_model.pth')
        # checkpoint = torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/model_saved/gat_column_resnet18_cifar10_3_90_train_column_mask_0.001maskCoefficient_0.1masklr_160epoch/checkpoint/last_model.pth')

        net.load_state_dict(checkpoint['state_dict'])

        r=0.7
        mask_net(net, ratio=r)
        exp_name = exp_name+'_'+str(r)+'masked_320epoch_finetune_91_2'
        print(exp_name)
        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'),
                                   sys.stderr)  # redirect std err, if necessary


        measure_flops.measure_model(net, dataset_name='cifar10')
        evaluate.evaluate_net(net,
                              data_loader=data_loader.create_test_loader(batch_size=512, num_workers=2,dataset_name='cifar10'),
                              save_net=False)
        train.train(net=net,
                    net_name='resnet18',
                    exp_name=exp_name,
                    dataset_name='cifar10',
                    optimizer=optim.SGD,
                    weight_decay=weight_decay,
                    requires_grad={'default': True, 'column_mask': False},
                    momentum=momentum,
                    learning_rate=0.01,
                    num_epochs=320,
                    batch_size=batch_size,
                    evaluate_step=5000,
                    resume=False,
                    test_net=False,
                    num_workers=4,
                    learning_rate_decay=True,
                    learning_rate_decay_epoch=[160, 240],  # [40,60],
                    learning_rate_decay_factor=0.1,
                    scheduler_name='MultiStepLR',
                    top_acc=1,
                    data_parallel=False,
                    paint_loss=False,
                    save_at_each_step=False,
                    use_tensorboard=True
                    # gradient_clip_value=gradient_clip_value
                    )
        print()