import os,sys
sys.path.append('../')
import torch
from torch import nn
import torch.optim as optim
from framework import evaluate,data_loader,measure_flops,train,draw
from network import vgg,storage,net_with_predicted_mask,resnet_cifar,resnet_cifar,resnet
from framework import config as conf
from network.modules import conv2d_with_mask_and_variable_shortcut
import logger
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ablation_exp_name='no_shortcut'

if ablation_exp_name == 'no_gat':
    net_name='resnet56'
    net_name='vgg16_bn'

    # no_gat
    if net_name == 'resnet56':
        optimizer_net = optim.SGD
        optimizer_extractor = optim.SGD
        learning_rate = {'default': 0.1, 'extractor': 0.0001}
        weight_decay = {'default':5e-4,'extractor':5e-4}
        momentum = {'default':0.9,'extractor':0.9}
        batch_size=128
        #网络参数
        add_shortcut_ratio=0.9#不是这儿！！！
        mask_update_freq = 1000
        mask_update_epochs = 900
        mask_training_start_epoch=1
        mask_training_stop_epoch=20
        exp_name='gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_nogat2'
        print(exp_name)
        description=exp_name+'  '+'专门训练mask,没有warmup，训练20epoch，没有gat，mask直接由梯度更新'
        total_flop=126550666#125485706
        prune_ratio=0.87
        flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        gradient_clip_value=5
        learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
        num_epochs=160*1+mask_training_stop_epoch
        #
        net=resnet_cifar.resnet56(num_classes=10).to(device)
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='resnet56',
                                                                               dataset_name='cifar10',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio,
                                                                               gcn_layer_num=0,
                                                                               no_gat=True
                                                                               )
        for mod in net.modules():
            if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                mod.mask = nn.Parameter(torch.ones(size=mod.mask.shape),requires_grad=True)
        net=net.cuda()
        # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # # save the output to log
        # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path, exist_ok=True)
        # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        #
        # print( weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)
        # train.train_extractor_network(net=net,
        #                               net_name='resnet56',
        #                               exp_name=exp_name,
        #                               description=description,
        #                               dataset_name='cifar10',
        #
        #                               optim_method_net=optimizer_net,
        #                               optim_method_extractor=optimizer_extractor,
        #                               weight_decay=weight_decay,
        #                               momentum=momentum,
        #                               learning_rate=learning_rate,
        #
        #                               num_epochs=num_epochs,
        #                               batch_size=batch_size,
        #                               evaluate_step=5000,
        #                               load_net=False,
        #                               test_net=False,
        #                               num_workers=4,
        #                               # weight_decay=5e-4,
        #                               learning_rate_decay=True,
        #                               learning_rate_decay_epoch=learning_rate_decay_epoch,
        #                               learning_rate_decay_factor=0.1,
        #                               scheduler_name='MultiStepLR',
        #                               top_acc=1,
        #                               paint_loss=True,
        #                               save_at_each_step=False,
        #                               )
        #
        checkpoint=torch.load('/home/disk_new/model_saved/gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_nogat2/checkpoint/masked_net.pth',map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        #mask net
        first_conv=False
        mask=None
        for mod in net.modules():
            if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                if first_conv is False:
                    first_conv = True
                    mask = mod.mask.detach()
                else:
                    mask = torch.cat((mask,mod.mask.detach()))
        mask= nn.Parameter(mask,requires_grad=False)
        prune_num = net.find_prune_num(mask)
        _, mask_index = torch.topk(torch.abs(mask), k=prune_num, dim=0, largest=False)
        index = torch.ones(mask.shape).to(mask.device)
        index[mask_index] = 0
        mask = mask * index
        lo = hi = 0
        last_conv_mask = None
        for name, mod in net.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut) and 'downsample' not in name:
                hi += mod.out_channels
                mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
                lo = hi
                last_conv_mask = mod.mask
            else:
                if isinstance(mod, nn.BatchNorm2d) and last_conv_mask is not None:
                    # prune the corresponding mean and var according to mask
                    mod.running_mean[last_conv_mask == 0] = 0
                    mod.running_var[last_conv_mask == 0] = 1
                last_conv_mask = None
        net.print_mask()
        net.prune_net()

        exp_name = 'gat_resnet56_doubleschedule_nogat' + str(int(prune_ratio * 100))
        description = exp_name + '  ' + ''

        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value)

        net.current_epoch = net.mask_training_stop_epoch + 1
        learning_rate_decay_epoch = [i for i in [80,120]]
        num_epochs = 160
        train.train(net=net,
                    net_name='resnet56',
                    exp_name=exp_name,
                    description=description,
                    dataset_name='cifar10',
                    optimizer=optim.SGD,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    evaluate_step=5000,
                    resume=False,
                    test_net=True,
                    num_workers=4,
                    learning_rate_decay=True,
                    learning_rate_decay_epoch=learning_rate_decay_epoch,
                    learning_rate_decay_factor=0.1,
                    scheduler_name='MultiStepLR',
                    top_acc=1,
                    data_parallel=False,
                    paint_loss=True,
                    save_at_each_step=False,
                    gradient_clip_value=gradient_clip_value
                    )
    elif net_name =='vgg16_bn':
        optimizer_net = optim.SGD
        optimizer_extractor = optim.SGD
        learning_rate = {'default': 0.1, 'extractor': 0.0001}
        weight_decay = {'default': 5e-4, 'extractor': 5e-4}
        momentum = {'default': 0.9, 'extractor': 0.9}
        batch_size = 128
        # 网络参数
        add_shortcut_ratio = 0.9  # 不是这儿！！！
        mask_update_freq = 1000
        mask_update_epochs = 900
        mask_training_start_epoch = 1
        mask_training_stop_epoch = 20
        exp_name = 'gat_vgg16_bn_predicted_mask_and_variable_shortcut_net_mask_newinner_nogat2'
        print(exp_name)
        description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch，没有gat，mask直接由梯度更新'
        total_flop = 316813412  # 125485706
        prune_ratio = 0.98
        flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        gradient_clip_value = 5
        learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [80, 120]]
        num_epochs = 160 * 1 + mask_training_stop_epoch
        #
        net = vgg.vgg16_bn(dataset_name = 'cifar100').to(device)
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='vgg16_bn',
                                                                               dataset_name='cifar100',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio,
                                                                               gcn_layer_num=0,
                                                                               no_gat=True
                                                                               )
        for mod in net.modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                mod.mask = nn.Parameter(torch.ones(size=mod.mask.shape), requires_grad=True)
        net = net.cuda()
        # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # # save the output to log
        # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path, exist_ok=True)
        # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        #
        # print( weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)
        # train.train_extractor_network(net=net,
        #                               net_name='vgg16_bn',
        #                               exp_name=exp_name,
        #                               description=description,
        #                               dataset_name='cifar100',
        #
        #                               optim_method_net=optimizer_net,
        #                               optim_method_extractor=optimizer_extractor,
        #                               weight_decay=weight_decay,
        #                               momentum=momentum,
        #                               learning_rate=learning_rate,
        #
        #                               num_epochs=num_epochs,
        #                               batch_size=batch_size,
        #                               evaluate_step=5000,
        #                               load_net=False,
        #                               test_net=False,
        #                               num_workers=4,
        #                               # weight_decay=5e-4,
        #                               learning_rate_decay=True,
        #                               learning_rate_decay_epoch=learning_rate_decay_epoch,
        #                               learning_rate_decay_factor=0.1,
        #                               scheduler_name='MultiStepLR',
        #                               top_acc=1,
        #                               paint_loss=True,
        #                               save_at_each_step=False,
        #                               )
        #
        checkpoint = torch.load(
            '/home/victorfang/PycharmProjects/model_pytorch/data/model_saved/gat_vgg16_bn_predicted_mask_and_variable_shortcut_net_mask_newinner_nogat1/checkpoint/masked_net.pth',
            map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        # mask net
        first_conv = False
        mask = None
        for mod in net.modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut):
                if first_conv is False:
                    first_conv = True
                    mask = mod.mask.detach()
                else:
                    mask = torch.cat((mask, mod.mask.detach()))
        mask = nn.Parameter(mask, requires_grad=False)
        prune_num = net.find_prune_num(mask)
        _, mask_index = torch.topk(torch.abs(mask), k=prune_num, dim=0, largest=False)
        index = torch.ones(mask.shape).to(mask.device)
        index[mask_index] = 0
        mask = mask * index
        lo = hi = 0
        last_conv_mask = None
        for name, mod in net.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut) and 'downsample' not in name:
                hi += mod.out_channels
                mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
                lo = hi
                last_conv_mask = mod.mask
            else:
                if isinstance(mod, nn.BatchNorm2d) and last_conv_mask is not None:
                    # prune the corresponding mean and var according to mask
                    mod.running_mean[last_conv_mask == 0] = 0
                    mod.running_var[last_conv_mask == 0] = 1
                last_conv_mask = None
        net.print_mask()
        net.prune_net()

        exp_name = 'gat_vgg16_bn_doubleschedule_nogat' + str(int(prune_ratio * 100))
        description = exp_name + '  ' + ''

        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'),
                                   sys.stderr)  # redirect std err, if necessary
        print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value)

        net.current_epoch = net.mask_training_stop_epoch + 1
        learning_rate_decay_epoch = [i for i in [80, 120]]
        num_epochs = 160
        train.train(net=net,
                    net_name='vgg16_bn',
                    exp_name=exp_name,
                    description=description,
                    dataset_name='cifar100',
                    optimizer=optim.SGD,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    evaluate_step=5000,
                    resume=False,
                    test_net=True,
                    num_workers=4,
                    learning_rate_decay=True,
                    learning_rate_decay_epoch=learning_rate_decay_epoch,
                    learning_rate_decay_factor=0.1,
                    scheduler_name='MultiStepLR',
                    top_acc=1,
                    data_parallel=False,
                    paint_loss=True,
                    save_at_each_step=False,
                    gradient_clip_value=gradient_clip_value
                    )

elif ablation_exp_name == 'no_shortcut':
    net_name='vgg16_bn'
    if net_name=='resnet56':
        # no shortcut
        prune_ratio=0.83
        exp_name='gat_resnet56_noshortcut_'+str(prune_ratio*100)
        print(exp_name)
        description='不用shortcut来训练剪完的网络'
        batch_size=128
        #网络参数
        add_shortcut_ratio=0.9#不是这儿！！！
        mask_update_freq = 1000
        mask_update_epochs = 900
        mask_training_start_epoch=1
        mask_training_stop_epoch=20
        total_flop=126550666#125485706
        flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        net=resnet_cifar.resnet56(num_classes=10).to(device)
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='resnet56',
                                                                               dataset_name='cifar10',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio,
                                                                               gcn_layer_num=2,
                                                                               no_gat=False
                                                                               )
        net=net.cuda()
        checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/masked_net/resnet56/9.pth',map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        net.set_shortcut_ratio(1)
        #mask net
        first_conv=False
        mask=None
        for name,mod in net.named_modules():
            if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                if first_conv is False:
                    first_conv = True
                    mask = torch.ones_like(mod.mask.detach())
                elif 'conv_a' in name:
                    mask = torch.cat((mask,mod.mask.detach()))
                else:
                    mask = torch.cat((mask,torch.ones_like(mod.mask.detach())))
        mask= nn.Parameter(mask,requires_grad=False)
        prune_num = net.find_prune_num(mask)
        _, mask_index = torch.topk(torch.abs(mask), k=prune_num, dim=0, largest=False)
        index = torch.ones(mask.shape).to(mask.device)
        index[mask_index] = 0
        mask = mask * index
        lo = hi = 0
        last_conv_mask = None
        for name, mod in net.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut) and 'downsample' not in name:
                hi += mod.out_channels
                mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
                lo = hi
                last_conv_mask = mod.mask
            else:
                if isinstance(mod, nn.BatchNorm2d) and last_conv_mask is not None:
                    # prune the corresponding mean and var according to mask
                    mod.running_mean[last_conv_mask == 0] = 0
                    mod.running_var[last_conv_mask == 0] = 1
                last_conv_mask = None
        net.print_mask()
        net.prune_net()
        net = net.net
        for mod in net.modules():
            if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                raise Exception('There is a traitor here!!!')
        train.train(net=net,
                    net_name='resnet56',
                    exp_name=exp_name,
                    description=description,
                    dataset_name='cifar10',
                    optimizer=optim.SGD,
                    weight_decay=5e-4,
                    momentum=0.9,
                    learning_rate=0.1,
                    num_epochs=160,
                    batch_size=batch_size,
                    evaluate_step=5000,
                    resume=False,
                    test_net=True,
                    num_workers=4,
                    learning_rate_decay=True,
                    learning_rate_decay_epoch=[i for i in [80,120]],
                    learning_rate_decay_factor=0.1,
                    scheduler_name='MultiStepLR',
                    top_acc=1,
                    data_parallel=False,
                    paint_loss=True,
                    save_at_each_step=False,
                    gradient_clip_value=None
                    )
    elif net_name=='vgg16_bn':
        # no shortcut
        prune_ratio=0.98
        exp_name='gat_resnet56_noshortcut_'+str(prune_ratio*100)
        print(exp_name)
        description='不用shortcut来训练剪完的网络'
        batch_size=128
        #网络参数
        add_shortcut_ratio=0.9#不是这儿！！！
        mask_update_freq = 1000
        mask_update_epochs = 900
        mask_training_start_epoch=1
        mask_training_stop_epoch=20
        total_flop=316813412#125485706
        flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        net=vgg.vgg16_bn(dataset_name='cifar100').to(device)
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='vgg16_bn',
                                                                               dataset_name='cifar100',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio,
                                                                               gcn_layer_num=2,
                                                                               no_gat=False
                                                                               )
        net=net.cuda()
        checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/masked_net/vgg16_cifar100/9.pth',map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        net.set_shortcut_ratio(1)
        #mask net
        first_conv=False
        mask=None
        for name,mod in net.named_modules():
            if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                if first_conv is False:
                    mask = mod.mask.detach()
                    first_conv = True
                mask = torch.cat((mask,mod.mask.detach()))
        mask= nn.Parameter(mask,requires_grad=False)
        prune_num = net.find_prune_num(mask)
        _, mask_index = torch.topk(torch.abs(mask), k=prune_num, dim=0, largest=False)
        index = torch.ones(mask.shape).to(mask.device)
        index[mask_index] = 0
        mask = mask * index
        lo = hi = 0
        last_conv_mask = None
        for name, mod in net.net.named_modules():
            if isinstance(mod, conv2d_with_mask_and_variable_shortcut) and 'downsample' not in name:
                hi += mod.out_channels
                mod.set_mask(mask[lo:hi].view(-1))  # update mask for each conv
                lo = hi
                last_conv_mask = mod.mask
            else:
                if isinstance(mod, nn.BatchNorm2d) and last_conv_mask is not None:
                    # prune the corresponding mean and var according to mask
                    mod.running_mean[last_conv_mask == 0] = 0
                    mod.running_var[last_conv_mask == 0] = 1
                last_conv_mask = None
        net.print_mask()
        net.prune_net()
        net = net.net
        for mod in net.modules():
            if isinstance(mod,conv2d_with_mask_and_variable_shortcut):
                raise Exception('There is a traitor here!!!')
        train.train(net=net,
                    net_name='vgg16_bn',
                    exp_name=exp_name,
                    description=description,
                    dataset_name='cifar100',
                    optimizer=optim.SGD,
                    weight_decay=5e-4,
                    momentum=0.9,
                    learning_rate=0.1,
                    num_epochs=160,
                    batch_size=batch_size,
                    evaluate_step=5000,
                    resume=False,
                    test_net=True,
                    num_workers=4,
                    learning_rate_decay=True,
                    learning_rate_decay_epoch=[i for i in [80,120]],
                    learning_rate_decay_factor=0.1,
                    scheduler_name='MultiStepLR',
                    top_acc=1,
                    data_parallel=False,
                    paint_loss=True,
                    save_at_each_step=False,
                    gradient_clip_value=None
                    )

elif ablation_exp_name == 'draw_net_mask':
    prune_ratio=0.85
    description='不用shortcut来训练剪完的网络'
    batch_size=128
    #网络参数
    add_shortcut_ratio=0.9#不是这儿！！！
    mask_update_freq = 1000
    mask_update_epochs = 900
    mask_training_start_epoch=1
    mask_training_stop_epoch=20
    total_flop=126550666#125485706
    flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
    net=resnet_cifar.resnet56(num_classes=10).to(device)
    net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                           net_name='resnet56',
                                                                           dataset_name='cifar10',
                                                                           mask_update_epochs=mask_update_epochs,
                                                                           mask_update_freq=mask_update_freq,
                                                                           flop_expected=flop_expected,
                                                                           mask_training_start_epoch=mask_training_start_epoch,
                                                                           mask_training_stop_epoch=mask_training_stop_epoch,
                                                                           batch_size=batch_size,
                                                                           add_shortcut_ratio=add_shortcut_ratio,
                                                                           gcn_layer_num=2,
                                                                           no_gat=False
                                                                           )
    net=net.cuda()
    checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/masked_net/resnet56/1.pth',map_location='cpu')
    # checkpoint = torch.load('/home/disk_new/model_saved/gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_bn_mean2gamma5_12/checkpoint/flop=127615626,accuracy=0.79990.tar',map_location='cpu')
    for key in list(checkpoint['state_dict'].keys()):
        if 'zero_vec' in key or 'eye_mat' in key or 'gat_layers.0.adj' in key or 'gat_layers.1.adj' in key:
            checkpoint['state_dict'].pop(key)
    net.load_state_dict(checkpoint['state_dict'])
    draw.draw_masked_net(net,'resnet56_1','/home/victorfang')

    # net=vgg.vgg16_bn(dataset_name='cifar10').to(device)
    # net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
    #                                                                        net_name='vgg16_bn',
    #                                                                        dataset_name='cifar10',
    #                                                                        mask_update_epochs=mask_update_epochs,
    #                                                                        mask_update_freq=mask_update_freq,
    #                                                                        flop_expected=flop_expected,
    #                                                                        mask_training_start_epoch=mask_training_start_epoch,
    #                                                                        mask_training_stop_epoch=mask_training_stop_epoch,
    #                                                                        batch_size=batch_size,
    #                                                                        add_shortcut_ratio=add_shortcut_ratio
    #                                                                        )
    # net=net.to(device)
    # checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/data/masked_net/vgg16/9.pth',map_location='cpu')
    # for key in list(checkpoint['state_dict'].keys()):
    #     if 'zero_vec' in key or 'eye_mat' in key or 'gat_layers.0.adj' in key or 'gat_layers.1.adj' in key:
    #         checkpoint['state_dict'].pop(key)
    # net.load_state_dict(checkpoint['state_dict'])
    # draw.draw_masked_net(net,'vgg16_9','/home/victorfang')
    print()
elif ablation_exp_name == 'gat_layer_num':
    gat_layer_num=[1,2,3,4,5]
    # gat_layer_num=[2,4,5]
    optimizer_net = optim.SGD
    optimizer_extractor = optim.SGD
    learning_rate = {'default': 0.1, 'extractor': 0.0001}
    weight_decay = {'default': 5e-4, 'extractor': 5e-4}
    momentum = {'default': 0.9, 'extractor': 0.9}
    batch_size = 128
    # 网络参数
    add_shortcut_ratio = 0.9  # 不是这儿！！！
    mask_update_freq = 1000
    mask_update_epochs = 900
    mask_training_start_epoch = 1
    mask_training_stop_epoch = 20


    total_flop = 126550666  # 125485706
    prune_ratio = 0.9
    flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
    gradient_clip_value = 5
    learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [80, 120]]
    num_epochs = 160 * 1 + mask_training_stop_epoch
    for layer_num in gat_layer_num:
        net = resnet_cifar.resnet56(num_classes=10).to(device)
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='resnet56',
                                                                               dataset_name='cifar10',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio,
                                                                               gcn_layer_num=layer_num,
                                                                               no_gat=False
                                                                               )
        # exp_name = 'gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_gat_layer_num_'+str(layer_num)
        # print(exp_name)
        # net = net.cuda()
        # checkpoint_path = os.path.join(conf.root_path, 'model_saved', 'ablation_gat_layer_num',exp_name)
        # # save the output to log
        # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path, exist_ok=True)
        # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        #
        # print( weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)
        # train.train_extractor_network(net=net,
        #                               net_name='resnet56',
        #                               exp_name=exp_name,
        #                               dataset_name='cifar10',
        #
        #                               optim_method_net=optimizer_net,
        #                               optim_method_extractor=optimizer_extractor,
        #                               weight_decay=weight_decay,
        #                               momentum=momentum,
        #                               learning_rate=learning_rate,
        #
        #                               num_epochs=num_epochs,
        #                               batch_size=batch_size,
        #                               evaluate_step=5000,
        #                               load_net=False,
        #                               test_net=False,
        #                               num_workers=4,
        #                               # weight_decay=5e-4,
        #                               learning_rate_decay=True,
        #                               learning_rate_decay_epoch=learning_rate_decay_epoch,
        #                               learning_rate_decay_factor=0.1,
        #                               scheduler_name='MultiStepLR',
        #                               top_acc=1,
        #                               paint_loss=True,
        #                               save_at_each_step=False,
        #                               )
        exp_name = 'gat_resnet56_gatlayernum_' + str(int(prune_ratio * 100)) + '_' + str(layer_num)
        description = exp_name + '  ' + ''
        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value,)

        saved_checkpoint_path = '/home/victorfang/model_pytorch/data/model_saved/gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_gat_layer_num_'+str(layer_num)+'/checkpoint/masked_net.pth'

        checkpoint = torch.load(saved_checkpoint_path,map_location='cpu')

        # checkpoint = torch.load('/home/disk_new/model_saved/gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_bn_mean2gamma5_12/checkpoint/flop=127615626,accuracy=0.81070.tar',map_location='cpu')
        # exp_name = 'test'

        net.load_state_dict(checkpoint['state_dict'])
        net.cuda()
        net.mask_net()
        net.print_mask()
        net.prune_net()
        net.current_epoch = net.mask_training_stop_epoch + 1
        learning_rate_decay_epoch = [2*i for i in [80,120]]
        num_epochs = 160*2
        train.train(net=net,
                    net_name='resnet56',
                    exp_name=exp_name,
                    description=description,
                    dataset_name='cifar10',
                    optimizer=optim.SGD,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    evaluate_step=5000,
                    resume=True,
                    test_net=False,
                    num_workers=2,
                    learning_rate_decay=True,
                    learning_rate_decay_epoch=learning_rate_decay_epoch,
                    learning_rate_decay_factor=0.1,
                    scheduler_name='MultiStepLR',
                    top_acc=1,
                    data_parallel=False,
                    paint_loss=True,
                    save_at_each_step=False,
                    gradient_clip_value=gradient_clip_value
                    )


# #add_shortcut_ratio
# optimizer_extractor = optim.SGD
# learning_rate = {'default': 0.1, 'extractor': 0.001}
# weight_decay = {'default':5e-4,'extractor':5e-4}
# momentum = {'default':0.9,'extractor':0.9}
# batch_size=128
# #网络参数
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch=1
# mask_training_stop_epoch=20
#
# total_flop=126550666
# prune_ratio=0.85
# flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value=2
# learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
# num_epochs=160*1+mask_training_stop_epoch
#
# l=[0.0001,0.5]
# l+=list(range(1,10))
# l=[4,5]
# for ratio in l:
#     add_shortcut_ratio=1-ratio/10
#     print("add_shortcut_ratio: ",add_shortcut_ratio)
#     net=resnet_cifar.resnet56(num_classes=10).to(device)
#     net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                            net_name='resnet56',
#                                                                            dataset_name='cifar10',
#                                                                            mask_update_epochs=mask_update_epochs,
#                                                                            mask_update_freq=mask_update_freq,
#                                                                            flop_expected=flop_expected,
#                                                                            gcn_layer_num=2,
#                                                                            mask_training_start_epoch=mask_training_start_epoch,
#                                                                            mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                            batch_size=batch_size,
#                                                                            add_shortcut_ratio=0.9
#                                                                            )
#     net=net.to(device)
#     i = 2
#     exp_name = 'ablation_add_shortcut_ratio_predicted_mask_and_variable_shortcut_net/resnet56_2_ratio:' + str(int(add_shortcut_ratio * 100))
#     description = exp_name + '  ' + ''
#
#     checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
#     # save the output to log
#     print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path, exist_ok=True)
#     sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
#     sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
#     print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value, i)
#
#
#     checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56',str(i) + '.tar'),map_location='cpu')
#     net.load_state_dict(checkpoint['state_dict'])
#
#     net.set_shortcut_ratio(add_shortcut_ratio)
#
#     net.mask_net()
#     net.print_mask()
#     net.prune_net()
#     net.current_epoch = net.mask_training_stop_epoch + 1
#     learning_rate_decay_epoch = [1*i for i in [80,120]]
#     num_epochs = 160*1
#     train.train(net=net,
#                 net_name='resnet56',
#                 exp_name=exp_name,
#                 description=description,
#                 dataset_name='cifar10',
#                 optimizer=optim.SGD,
#                 weight_decay=weight_decay,
#                 momentum=momentum,
#                 learning_rate=learning_rate,
#                 num_epochs=num_epochs,
#                 batch_size=batch_size,
#                 evaluate_step=5000,
#                 resume=False,
#                 test_net=True,
#                 num_workers=2,
#                 learning_rate_decay=True,
#                 learning_rate_decay_epoch=learning_rate_decay_epoch,
#                 learning_rate_decay_factor=0.1,
#                 scheduler_name='MultiStepLR',
#                 top_acc=1,
#                 data_parallel=False,
#                 paint_loss=True,
#                 save_at_each_step=False,
#                 gradient_clip_value=gradient_clip_value
#                 )
#     net.print_mask()


# #gcn round
# optimizer_net = optim.SGD
# optimizer_extractor = optim.SGD
# learning_rate = {'default': 0.1, 'extractor': 0.001}
# weight_decay = {'default': 5e-4, 'extractor': 5e-4}
# momentum = {'default': 0.9, 'extractor': 0.9}
# batch_size=128
# #网络参数
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch=1
# mask_training_stop_epoch=20
#
#
# total_flop=126550666
# prune_ratio=0.85
# flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value=None
# learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
# num_epochs=160*1+mask_training_stop_epoch
#
# rounds = [1,2,3,4,5]
# for r in rounds:
#     print("gcn round: ",r)
#     net=resnet_cifar.resnet56(num_classes=10).to(device)
#     net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                            net_name='resnet56',
#                                                                            dataset_name='cifar10',
#                                                                            mask_update_epochs=mask_update_epochs,
#                                                                            mask_update_freq=mask_update_freq,
#                                                                            flop_expected=flop_expected,
#                                                                            gcn_rounds=r,
#                                                                            mask_training_start_epoch=mask_training_start_epoch,
#                                                                            mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                            batch_size=batch_size,
#                                                                            add_shortcut_ratio=0.9
#                                                                            )
#     net=net.to(device)
#     exp_name = 'ablation_gcn_round_predicted_mask_and_variable_shortcut_net_'+str(prune_ratio*100)+'pruned/resnet56_gcnRound_' + str(r)
#     description = exp_name + '  ' + ''
#     print(exp_name)
#
#     checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
#     # save the output to log
#     print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path, exist_ok=True)
#     sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
#     sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
#     print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value,r)
#     if r !=9:
#         train.train_extractor_network(net=net,
#                                       net_name='resnet56',
#                                       exp_name=exp_name,
#                                       description=description,
#                                       dataset_name='cifar10',
#
#                                       optim_method_net=optimizer_net,
#                                       optim_method_extractor=optimizer_extractor,
#                                       weight_decay=weight_decay,
#                                       momentum=momentum,
#                                       learning_rate=learning_rate,
#
#                                       num_epochs=num_epochs,
#                                       batch_size=batch_size,
#                                       evaluate_step=5000,
#                                       load_net=False,
#                                       test_net=False,
#                                       num_workers=4,
#                                       # weight_decay=5e-4,
#                                       learning_rate_decay=True,
#                                       learning_rate_decay_epoch=learning_rate_decay_epoch,
#                                       learning_rate_decay_factor=0.1,
#                                       scheduler_name='MultiStepLR',
#                                       top_acc=1,
#                                       paint_loss=True,
#                                       save_at_each_step=False,
#                                       gradient_clip_value=gradient_clip_value
#                                       )
#     print('load checkpoint from:',os.path.join(checkpoint_path,'checkpoint','masked_net.tar'))
#     ck=torch.load(os.path.join(checkpoint_path,'checkpoint','masked_net.tar'))
#     net.load_state_dict(ck['state_dict'])
#
#     net.mask_net()
#     net.print_mask()
#     net.prune_net()
#     net.current_epoch = net.mask_training_stop_epoch + 1
#     learning_rate_decay_epoch = [1*i for i in [80,120]]
#     measure_flops.measure_model(net,dataset_name='cifar10',print_flop=True)
#     num_epochs = 160*1
#     exp_name+='_train'
#     train.train(net=net,
#                 net_name='resnet56',
#                 exp_name=exp_name,
#                 description=description,
#                 dataset_name='cifar10',
#                 optimizer=optim.SGD,
#                 weight_decay=weight_decay,
#                 momentum=momentum,
#                 learning_rate=learning_rate,
#                 num_epochs=num_epochs,
#                 batch_size=batch_size,
#                 evaluate_step=5000,
#                 load_net=False,
#                 test_net=True,
#                 num_workers=2,
#                 learning_rate_decay=True,
#                 learning_rate_decay_epoch=learning_rate_decay_epoch,
#                 learning_rate_decay_factor=0.1,
#                 scheduler_name='MultiStepLR',
#                 top_acc=1,
#                 data_parallel=False,
#                 paint_loss=True,
#                 save_at_each_step=False,
#                 gradient_clip_value=gradient_clip_value
#                 )


# #only_gcn_inner
# optimizer_net = optim.SGD
# optimizer_extractor = optim.SGD
# learning_rate = {'default': 0.1, 'extractor': 0.001}
# weight_decay = {'default': 1e-4, 'extractor': 5e-4}
# momentum = {'default': 0.9, 'extractor': 0.9}
# batch_size=128
# #网络参数
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch=1
# mask_training_stop_epoch=20
#
# total_flop=126550666
# prune_ratio=0.83
# flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value=None
# learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
# num_epochs=160*1+mask_training_stop_epoch
#
# sets=[['both',False,False],['only_gcn',True,False],['only_inner',False,True]]
# sets=[['only_gcn',True,False],['only_inner',False,True]]
# for s in sets:
#     print(s)
#     net=resnet_cifar.resnet56(num_classes=10).to(device)
#     net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                            net_name='resnet56',
#                                                                            dataset_name='cifar10',
#                                                                            mask_update_epochs=mask_update_epochs,
#                                                                            mask_update_freq=mask_update_freq,
#                                                                            flop_expected=flop_expected,
#                                                                            gcn_rounds=2,
#                                                                            mask_training_start_epoch=mask_training_start_epoch,
#                                                                            mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                            batch_size=batch_size,
#                                                                            add_shortcut_ratio=0.9,
#                                                                            only_gcn=s[1],
#                                                                            only_inner_features=s[2]
#                                                                            )
#     net=net.to(device)
#     exp_name = 'ablation_only_gcn_inner_predicted_mask_and_variable_shortcut_net/resnet56_'+str(prune_ratio*100)+'/'+s[0]
#     print(exp_name)
#     description = exp_name + '  ' + ''
#
#     # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
#     # # save the output to log
#     # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
#     # if not os.path.exists(checkpoint_path):
#     #     os.makedirs(checkpoint_path, exist_ok=True)
#     # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
#     # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
#     # print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value,s)
#     #
#     # train.train_extractor_network(net=net,
#     #                               net_name='resnet56',
#     #                               exp_name=exp_name,
#     #                               description=description,
#     #                               dataset_name='cifar10',
#     #
#     #                               optim_method_net=optimizer_net,
#     #                               optim_method_extractor=optimizer_extractor,
#     #                               weight_decay=weight_decay,
#     #                               momentum=momentum,
#     #                               learning_rate=learning_rate,
#     #
#     #                               num_epochs=num_epochs,
#     #                               batch_size=batch_size,
#     #                               evaluate_step=5000,
#     #                               load_net=False,
#     #                               test_net=False,
#     #                               num_workers=4,
#     #                               # weight_decay=5e-4,
#     #                               learning_rate_decay=True,
#     #                               learning_rate_decay_epoch=learning_rate_decay_epoch,
#     #                               learning_rate_decay_factor=0.1,
#     #                               scheduler_name='MultiStepLR',
#     #                               top_acc=1,
#     #                               paint_loss=True,
#     #                               save_at_each_step=False,
#     #                               gradient_clip_value=gradient_clip_value
#     #                               )
#
#     ck_path=os.path.join('/home/disk_new/model_saved/ablation_only_gcn_inner_predicted_mask_and_variable_shortcut_net/',s[0]+'.tar')
#     checkpoint=torch.load(ck_path)
#     net.load_state_dict(checkpoint['state_dict'])
#     print('load net from ',ck_path)
#     net.mask_net()
#     net.print_mask()
#     net.prune_net()
#     net.current_epoch = net.mask_training_stop_epoch + 1
#     learning_rate_decay_epoch = [1*i for i in [80,120]]
#     measure_flops.measure_model(net,dataset_name='cifar10',print_flop=True)
#     num_epochs = 160*1
#     exp_name+='_train'
#     train.train(net=net,
#                 net_name='resnet56',
#                 exp_name=exp_name,
#                 description=description,
#                 dataset_name='cifar10',
#                 optimizer=optim.SGD,
#                 weight_decay=weight_decay,
#                 momentum=momentum,
#                 learning_rate=learning_rate,
#                 num_epochs=num_epochs,
#                 batch_size=batch_size,
#                 evaluate_step=5000,
#                 load_net=False,
#                 test_net=True,
#                 num_workers=2,
#                 learning_rate_decay=True,
#                 learning_rate_decay_epoch=learning_rate_decay_epoch,
#                 learning_rate_decay_factor=0.1,
#                 scheduler_name='MultiStepLR',
#                 top_acc=1,
#                 data_parallel=False,
#                 paint_loss=True,
#                 save_at_each_step=False,
#                 gradient_clip_value=gradient_clip_value
#                 )


# #resnet110
# optimizer_net = optim.SGD
# optimizer_extractor = optim.SGD
# learning_rate = {'default': 0.1, 'extractor': 0.0001}
# weight_decay = {'default': 1e-4, 'extractor': 5e-4}
# momentum = {'default': 0.9, 'extractor': 0.9}
# batch_size=128
# #网络参数
# mask_update_freq = 1000
# mask_update_epochs = 900
# mask_training_start_epoch=1
# mask_training_stop_epoch=20
#
#
# flop_expected=12831484
# gradient_clip_value=None
# learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
# num_epochs=160*1+mask_training_stop_epoch
#
# for i in range(1,4):
#     print(i)
#     net=resnet_cifar.resnet110(num_classes=10).to(device)
#     net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
#                                                                            net_name='resnet110',
#                                                                            dataset_name='cifar10',
#                                                                            mask_update_epochs=mask_update_epochs,
#                                                                            mask_update_freq=mask_update_freq,
#                                                                            flop_expected=flop_expected,
#                                                                            gcn_rounds=2,
#                                                                            mask_training_start_epoch=mask_training_start_epoch,
#                                                                            mask_training_stop_epoch=mask_training_stop_epoch,
#                                                                            batch_size=batch_size,
#                                                                            add_shortcut_ratio=0.9,
#                                                                            )
#     net=net.to(device)
#     exp_name = 'resnet110/resnet110_like_resnet56_90pruned_'+str(i)
#     print(exp_name)
#     description = exp_name + '  ' + ''
#
#     # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
#     # # save the output to log
#     # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
#     # if not os.path.exists(checkpoint_path):
#     #     os.makedirs(checkpoint_path, exist_ok=True)
#     # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
#     # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
#     # print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value,i)
#     #
#     # train.train_extractor_network(net=net,
#     #                               net_name='resnet110',
#     #                               exp_name=exp_name,
#     #                               description=description,
#     #                               dataset_name='cifar10',
#     #
#     #                               optim_method_net=optimizer_net,
#     #                               optim_method_extractor=optimizer_extractor,
#     #                               weight_decay=weight_decay,
#     #                               momentum=momentum,
#     #                               learning_rate=learning_rate,
#     #
#     #                               num_epochs=num_epochs,
#     #                               batch_size=batch_size,
#     #                               evaluate_step=5000,
#     #                               load_net=False,
#     #                               test_net=False,
#     #                               num_workers=4,
#     #                               # weight_decay=5e-4,
#     #                               learning_rate_decay=True,
#     #                               learning_rate_decay_epoch=learning_rate_decay_epoch,
#     #                               learning_rate_decay_factor=0.1,
#     #                               scheduler_name='MultiStepLR',
#     #                               top_acc=1,
#     #                               paint_loss=True,
#     #                               save_at_each_step=False,
#     #                               gradient_clip_value=gradient_clip_value
#     #                               )
#
#     checkpoint=torch.load('/home/swim/fang/model_pytorch/data/model_saved/resnet110/resnet110_like_resnet56_90pruned_1/checkpoint/masked_net.tar')
#     net.load_state_dict(checkpoint['state_dict'])
#
#     net.mask_net()
#     net.print_mask()
#     net.prune_net()
#     net.current_epoch = net.mask_training_stop_epoch + 1
#     learning_rate_decay_epoch = [1*i for i in [80,120]]
#     measure_flops.measure_model(net,dataset_name='cifar10',print_flop=True)
#     num_epochs = 160*1
#     exp_name+='_train'
#     train.train(net=net,
#                 net_name='resnet56',
#                 exp_name=exp_name,
#                 description=description,
#                 dataset_name='cifar10',
#                 optimizer=optim.SGD,
#                 weight_decay=weight_decay,
#                 momentum=momentum,
#                 learning_rate=learning_rate,
#                 num_epochs=num_epochs,
#                 batch_size=batch_size,
#                 evaluate_step=5000,
#                 load_net=False,
#                 test_net=True,
#                 num_workers=2,
#                 learning_rate_decay=True,
#                 learning_rate_decay_epoch=learning_rate_decay_epoch,
#                 learning_rate_decay_factor=0.1,
#                 scheduler_name='MultiStepLR',
#                 top_acc=1,
#                 data_parallel=False,
#                 paint_loss=True,
#                 save_at_each_step=False,
#                 gradient_clip_value=gradient_clip_value
#                 )