# -*- coding: utf-8 -*-
import os,sys
sys.path.append('../')
import torch
from torch import nn
import torch.optim as optim
from framework import evaluate,data_loader,measure_flops,train
from network import vgg,storage,net_with_predicted_mask,resnet_cifar,resnet_cifar,resnet,mobilenet
from framework import config as conf
import logger
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset='imagenet'
net_type='resnet50'
dataset='cifar10'
# net_type='resnet56'
net_type='vgg16_bn'
# # #for cifar
# # #训练参数
if dataset == 'cifar10':
    optimizer_net = optim.SGD
    optimizer_extractor = optim.SGD
    learning_rate = {'default': 0.0001, 'extractor': 0.0001}
    weight_decay = {'default':5e-4,'extractor':5e-4}
    momentum = {'default':0.9,'extractor':0.9}
    batch_size=128
    #网络参数
    add_shortcut_ratio=0.9#不是这儿！！！
    mask_update_freq = 1000
    mask_update_epochs = 900
    mask_training_start_epoch=1
    mask_training_stop_epoch=20

    if net_type=='resnet56':
        learning_rate = {'default': 0.0001, 'extractor': 0.0001}
        exp_name='gat_pretrained_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_bn_meanstd_2'
        description=exp_name+'  '+'专门训练mask,没有warmup，训练20epoch'

        total_flop=126550666#125485706
        prune_ratio=0.7
        flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        gradient_clip_value=5
        learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
        num_epochs=160*1+mask_training_stop_epoch
        #
        net=resnet_cifar.resnet56(num_classes=10).to(device)
        # checkpoint=torch.load('/home/victorfang/semantic_adversarial_nas/data/model_saved/resnet56_baseline_colorjitter_randomrotate/checkpoint/flop=127615626,accuracy=0.93570.pth')
        # net.load_state_dict(checkpoint['state_dict'])

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
                                                                               gcn_layer_num=2
                                                                               )
        net=net.to(device)
        # # net.prune_net()
        # # net = net.net
        # # state=net.state_dict()
        # # for key in state.keys():
        # #     print()
        # #     if torch.sum(checkpoint['state_dict'][key] != state[key]) >0:
        # #         print(key)
        # # data_loader=data_loader.create_test_loader(batch_size=512,num_workers=4,dataset_name='cifar10')
        # # evaluate.evaluate_net(net,data_loader,False)
        #
        #
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
        #                               gradient_clip_value=gradient_clip_value
        #                               )
        # #

        i = 2
        exp_name = 'gat_pretrained_resnet56_predicted_mask_and_variable_shortcut_net_newinner_doubleschedule' + str(int(prune_ratio * 100)) + '_' + str(i)
        description = exp_name + '  ' + ''

        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary


        checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56','pretrained_'+str(i) + '.pth'),map_location='cpu')

        # checkpoint = torch.load('/home/disk_new/model_saved/gat_resnet56_predicted_mask_and_variable_shortcut_net_mask_newinner_bn_mean2gamma5_12/checkpoint/flop=127615626,accuracy=0.81070.tar',map_location='cpu')
        # exp_name = 'test'

        # for key in list(checkpoint['state_dict'].keys()):
        #     if 'zero_vec' in key or 'eye_mat' in key or 'gat_layers.0.adj' in key or 'gat_layers.1.adj' in key:
        #         checkpoint['state_dict'].pop(key)

        net.load_state_dict(checkpoint['state_dict'])
        net.mask_net()
        net.print_mask()
        net.prune_net()
        net.current_epoch = net.mask_training_stop_epoch + 1
        learning_rate=0.1
        learning_rate_decay_epoch = [2*i for i in [80,120]]
        num_epochs = 160*2
        # learning_rate=0.01
        # learning_rate_decay_epoch = [40,80]
        # num_epochs = 120
        print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value, i)
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
        #
        eval_loader = data_loader.create_test_loader(batch_size=batch_size, num_workers=4, dataset_name='cifar10')
        evaluate.evaluate_net(net, eval_loader, save_net=False)
    elif net_type=='vgg16_bn':
        exp_name='gat_pretrained_vgg16bn_predicted_mask_and_variable_shortcut_net_mask_newinner_2'
        batch_size=128
        description=exp_name+'  '+'专门训练mask,没有warmup，训练20epoch'
        total_flop=314017290
        prune_ratio=0.7
        flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        gradient_clip_value=None
        learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
        num_epochs=160*1+mask_training_stop_epoch

        # learning_rate = {'default': 0.1, 'extractor': 0.0001}
        # weight_decay = {'default': 5e-4, 'extractor': 0}

        net=vgg.vgg16_bn(dataset_name='cifar10').cuda()
        checkpoint = torch.load(conf.root_path+'/baseline/vgg16_bn_cifar10_acc=0.93460.pth')
        net.load_state_dict(checkpoint['state_dict'])

        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='vgg16_bn',
                                                                               dataset_name='cifar10',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio
                                                                               )
        net=net.to(device)
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
        #                               data_distributed=False,
        #                               paint_loss=True,
        #                               save_at_each_step=True,
        #                               gradient_clip_value=gradient_clip_value
        #                               )


        i = 2
        exp_name = 'gat_pretrained_vgg16bn_predicted_mask_and_variable_shortcut_net_newinner_finetune_' + str(int(prune_ratio * 100)) + '_' + str(i)
        description = exp_name + '  ' + ''

        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary


        checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net','vgg16', 'pretrained_'+str(i) + '.pth'),map_location='cpu')
        # checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_vgg16bn_predicted_mask_and_variable_shortcut_net_mask_newinner_mean5gamma0reg_bn_test/checkpoint/flop=314570250,accuracy=0.81950.tar',map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        # train.add_forward_hook(net,module_name='extractor.network.0')
        net.mask_net()
        net.print_mask()
        net.prune_net()
        net.current_epoch = net.mask_training_stop_epoch + 1
        # learning_rate = 0.1
        # learning_rate_decay_epoch = [2*i for i in [80,120]]
        # num_epochs = 160*2

        learning_rate = 0.01
        learning_rate_decay_epoch=[40,80]
        num_epochs = 120
        net=net.net

        print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value, i)

        train.train(net=net,
                    net_name='vgg16_bn',
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
        #
        eval_loader = data_loader.create_test_loader(batch_size=batch_size, num_workers=0, dataset_name='cifar10')
        evaluate.evaluate_net(net, eval_loader, save_net=False)
    else:
        raise AttributeError



elif dataset == 'cifar100':
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

    if net_type =='vgg16_bn':
        exp_name='test'#'gat_vgg16bn_cifar100_net_mask_newinner_9'
        description=exp_name+'  '+'专门训练mask,没有warmup，训练20epoch'
        batch_size=128
        total_flop=316813412
        prune_ratio=0.6
        flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        gradient_clip_value=None
        learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
        num_epochs=160*1+mask_training_stop_epoch

        # weight_decay = {'default': 5e-4, 'extractor': 0}


        net=vgg.vgg16_bn(dataset_name='cifar100').to(device)
        measure_flops.measure_model(net,dataset_name='cifar100')
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
                                                                               gcn_layer_num=2
                                                                               )
        net=net.to(device)
        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        print( weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)
        train.train_extractor_network(net=net,
                                      net_name='vgg16_bn',
                                      exp_name=exp_name,
                                      description=description,
                                      dataset_name='cifar100',

                                      optim_method_net=optimizer_net,
                                      optim_method_extractor=optimizer_extractor,
                                      weight_decay=weight_decay,
                                      momentum=momentum,
                                      learning_rate=learning_rate,

                                      num_epochs=num_epochs,
                                      batch_size=batch_size,
                                      evaluate_step=5000,
                                      load_net=False,
                                      test_net=False,
                                      num_workers=4,
                                      # weight_decay=5e-4,
                                      learning_rate_decay=True,
                                      learning_rate_decay_epoch=learning_rate_decay_epoch,
                                      learning_rate_decay_factor=0.1,
                                      scheduler_name='MultiStepLR',
                                      top_acc=1,
                                      data_distributed=False,
                                      paint_loss=True,
                                      save_at_each_step=False,
                                      gradient_clip_value=gradient_clip_value
                                      )

        #
        # i = 5
        # exp_name = 'gat_vgg16bn_cifar100_predicted_mask_and_variable_shortcut_net_newinner_doubleschedule_' + str(int(prune_ratio * 100)) + '_' + str(i)
        # description = exp_name + '  ' + ''
        #
        # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # # save the output to log
        # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path, exist_ok=True)
        # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        # print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value, i)
        #
        #
        # checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net','vgg16_cifar100', str(i) + '.pth'),map_location='cpu')
        # # checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/gat_vgg16bn_cifar100_net_mask_newinner_mean5gamma5reg_test/checkpoint/masked_net.pth',map_location='cpu')
        # net.load_state_dict(checkpoint['state_dict'])
        #
        # net.mask_net()
        # net.print_mask()
        # net.prune_net()
        # net.current_epoch = net.mask_training_stop_epoch + 1
        # learning_rate_decay_epoch = [2*i for i in [80,120]]
        # learning_rate_decay_factor=0.1
        # net=net.net
        # num_epochs = 320#160*1
        # train.train(net=net,
        #             net_name='vgg16_bn',
        #             exp_name=exp_name,
        #             description=description,
        #             dataset_name='cifar100',
        #             optimizer=optim.SGD,
        #             weight_decay=weight_decay,
        #             momentum=momentum,
        #             learning_rate=learning_rate,
        #             num_epochs=num_epochs,
        #             batch_size=batch_size,
        #             evaluate_step=5000,
        #             resume=False,
        #             test_net=False,
        #             num_workers=2,
        #             learning_rate_decay=True,
        #             learning_rate_decay_epoch=learning_rate_decay_epoch,
        #             learning_rate_decay_factor=learning_rate_decay_factor,
        #             scheduler_name='MultiStepLR',
        #             top_acc=1,
        #             data_parallel=False,
        #             paint_loss=False,
        #             save_at_each_step=False,
        #             gradient_clip_value=gradient_clip_value,
        #             use_tensorboard=True
        #             )
        #
        # eval_loader = data_loader.create_test_loader(batch_size=batch_size, num_workers=0, dataset_name='cifar10')
        # evaluate.evaluate_net(net, eval_loader, save_net=False)
    elif net_type == 'resnet56':
        exp_name = 'resnet56_cifar100_predicted_mask_and_variable_shortcut_net_mask_newinner_5'
        description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'

        total_flop = 126556516
        prune_ratio = 0.90
        flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        gradient_clip_value = None
        learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [80, 120]]
        num_epochs = 160 * 1 + mask_training_stop_epoch

        net = resnet_cifar.resnet56(num_classes=100).cuda()
        measure_flops.measure_model(net, dataset_name='cifar100')
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='resnet56',
                                                                               dataset_name='cifar100',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               gcn_rounds=2,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio
                                                                               )
        net = net.to(device)
        # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # # save the output to log
        # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path, exist_ok=True)
        # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
        # print( weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)
        # train.train_extractor_network(net=net,
        #                               net_name='resnet56',
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
        #                               num_workers=2,
        #                               # weight_decay=5e-4,
        #                               learning_rate_decay=True,
        #                               learning_rate_decay_epoch=learning_rate_decay_epoch,
        #                               learning_rate_decay_factor=0.1,
        #                               scheduler_name='MultiStepLR',
        #                               top_acc=1,
        #                               data_distributed=False,
        #                               paint_loss=True,
        #                               save_at_each_step=False,
        #                               gradient_clip_value=gradient_clip_value
        #                               )
        #

        i = 5
        exp_name = 'resnet56_cifar100_predicted_mask_and_variable_shortcut_net_newinner_repeat' + str(
            int(prune_ratio * 100)) + '_' + str(i)
        description = exp_name + '  ' + ''

        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'),
                                   sys.stderr)  # redirect std err, if necessary
        print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value, i)

        checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet56_cifar100', str(i) + '.tar'),
                                map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])

        a = 0
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                a += mod.out_channels
        print(a)

        net.mask_net()
        net.print_mask()
        net.prune_net()
        net.current_epoch = net.mask_training_stop_epoch + 1
        learning_rate_decay_epoch = [2 * i for i in [80, 120]]
        learning_rate_decay_factor = 0.1

        num_epochs = 320  # 160*1
        train.train(net=net,
                    net_name='resnet56',
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
                    test_net=False,
                    num_workers=2,
                    learning_rate_decay=True,
                    learning_rate_decay_epoch=learning_rate_decay_epoch,
                    learning_rate_decay_factor=learning_rate_decay_factor,
                    scheduler_name='MultiStepLR',
                    top_acc=1,
                    data_parallel=False,
                    paint_loss=False,
                    save_at_each_step=False,
                    gradient_clip_value=gradient_clip_value,
                    use_tensorboard=True
                    )

        eval_loader = data_loader.create_test_loader(batch_size=batch_size, num_workers=0, dataset_name='cifar10')
        evaluate.evaluate_net(net, eval_loader, save_net=False)




elif dataset=='imagenet':

    if net_type == 'resnet50':
        # resnet50
        optimizer_net = optim.SGD
        optimizer_extractor = optim.SGD
        learning_rate = {'default': 0.0001, 'extractor': 0.0001}
        weight_decay = {'default': 1e-4, 'extractor': 1e-4}
        momentum = {'default': 0.9, 'extractor': 0.9}
        batch_size = 256
        # 网络参数
        add_shortcut_ratio = 0.9  # 不是这儿！！！
        mask_update_freq = 1000
        mask_update_epochs = 900
        mask_training_start_epoch = 1
        mask_training_stop_epoch = 3

        exp_name = 'gat_pretrained_resnet50_predicted_mask_and_variable_shortcut_net_mask_newinner_bn_2'
        description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'

        total_flop = 4133641192
        prune_ratio = 0.5
        flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        gradient_clip_value = None
        learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [30, 60]]
        num_epochs = 90 * 1 + mask_training_stop_epoch

        net = resnet.resnet50(pretrained=True).cuda()

        batch_size=128
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='resnet50',
                                                                               dataset_name='imagenet',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio,
                                                                               gcn_layer_num=2
                                                                               )

        net = net.cuda()


        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary

        print( weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)

        train.train_extractor_network(net=net,
                                      net_name='resnet50',
                                      exp_name=exp_name,
                                      description=description,
                                      dataset_name='imagenet',
                                      optim_method_net=optimizer_net,
                                      optim_method_extractor=optimizer_extractor,
                                      weight_decay=weight_decay,
                                      momentum=momentum,
                                      learning_rate=learning_rate,

                                      num_epochs=num_epochs,
                                      batch_size=batch_size,
                                      evaluate_step=2000,
                                      load_net=False,
                                      test_net=False,
                                      num_workers=8,
                                      learning_rate_decay=True,
                                      learning_rate_decay_epoch=learning_rate_decay_epoch,
                                      learning_rate_decay_factor=0.1,
                                      scheduler_name='MultiStepLR',
                                      top_acc=1,
                                      data_distributed=False,
                                      paint_loss=True,
                                      save_at_each_step=False,
                                      gradient_clip_value=gradient_clip_value
                                      )

        # # #
        # i = 9
        # exp_name = 'gat_resnet50_predicted_mask_and_variable_shortcut_net_newinner_newtrain_' + str(
        #     int(prune_ratio * 100)) + '_' + str(i) #+'_4gpu'
        # description = exp_name + '  ' + ''
        #
        # checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # # save the output to log
        # print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        # if not os.path.exists(checkpoint_path):
        #     os.makedirs(checkpoint_path, exist_ok=True)
        # sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        # sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'),
        #                            sys.stderr)  # redirect std err, if necessary
        # print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value, i)
        # checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'resnet50', str(i) + '.pth'),map_location='cpu')
        # net.load_state_dict(checkpoint['state_dict'])
        #
        # net.mask_net()
        # # net.print_mask()
        # net.prune_net()
        # net.print_mask()
        # net.current_epoch = net.mask_training_stop_epoch + 1
        # pruned_flop = net.measure_self_flops()
        # print('prune_ratio:', 1 - pruned_flop / total_flop)
        # measure_flops.measure_model(net.net)
        # learning_rate_decay_epoch = [2 * i for i in [30, 60,90]]
        # num_epochs = 100 * 2
        #
        # net = nn.DataParallel(net)
        # net=net.module.net
        # net=nn.DataParallel(net)
        #
        #
        # net = net.cuda()
        # train.train(net=net,
        #             net_name='resnet50',
        #             exp_name=exp_name,
        #             description=description,
        #             dataset_name='imagenet',
        #             optimizer=optim.SGD,
        #             weight_decay=weight_decay,
        #             momentum=momentum,
        #             # learning_rate=0.01,
        #             # learning_rate_decay_epoch=[30],
        #             # num_epochs=60,
        #             # batch_size=256,
        #             criterion=train.CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1).cuda(),
        #             # learning_rate=0.01,
        #             # learning_rate_decay_epoch=[17],
        #             # num_epochs=47,
        #             learning_rate=learning_rate,
        #             learning_rate_decay_epoch=learning_rate_decay_epoch,
        #             num_epochs=num_epochs,
        #             batch_size=batch_size,
        #             evaluate_step=2000,
        #             resume=True,
        #             test_net=True,
        #             num_workers=4,
        #             learning_rate_decay=True,
        #             learning_rate_decay_factor=0.1,
        #             scheduler_name='MultiStepLR',
        #             top_acc=1,
        #             data_parallel=True,
        #             paint_loss=False,
        #             save_at_each_step=False,
        #             gradient_clip_value=gradient_clip_value,
        #             use_tensorboard=True
        #             )
    if net_type == 'mobilenet_v1':
        # mobilenet_v1
        optimizer_net = optim.SGD
        optimizer_extractor = optim.SGD
        learning_rate = {'default': 0.1, 'extractor': 0.0001}
        weight_decay = {'default': 1e-4, 'extractor': 1e-4}
        momentum = {'default': 0.9, 'extractor': 0.9}
        batch_size = 256
        # 网络参数
        add_shortcut_ratio = 0.9  # 不是这儿！！！
        mask_update_freq = 1000
        mask_update_epochs = 900
        mask_training_start_epoch = 1
        mask_training_stop_epoch = 3

        exp_name = 'gat_mobilenet_v1_predicted_mask_and_variable_shortcut_net_mask_newinner_bn_revised_oldreg_1'
        description = exp_name + '  ' + '专门训练mask,没有warmup，训练20epoch'

        total_flop = 578826728
        prune_ratio = 0.5
        flop_expected = total_flop * (1 - prune_ratio)  # 0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
        gradient_clip_value = None
        learning_rate_decay_epoch = [mask_training_stop_epoch + 1 * i for i in [30, 60]]
        num_epochs = 90 * 1 + mask_training_stop_epoch

        net = mobilenet.mobilenet_v1(num_class=1000).cuda()
        # batch_size=128
        net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                               net_name='mobilenet_v1',
                                                                               dataset_name='imagenet',
                                                                               mask_update_epochs=mask_update_epochs,
                                                                               mask_update_freq=mask_update_freq,
                                                                               flop_expected=flop_expected,
                                                                               mask_training_start_epoch=mask_training_start_epoch,
                                                                               mask_training_stop_epoch=mask_training_stop_epoch,
                                                                               batch_size=batch_size,
                                                                               add_shortcut_ratio=add_shortcut_ratio,
                                                                               gcn_layer_num=2,
                                                                               feature_len=9
                                                                               )

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
        #
        # train.train_extractor_network(net=net,
        #                               net_name='mobilenet_v1',
        #                               exp_name=exp_name,
        #                               description=description,
        #                               dataset_name='imagenet',
        #                               optim_method_net=optimizer_net,
        #                               optim_method_extractor=optimizer_extractor,
        #                               weight_decay=weight_decay,
        #                               momentum=momentum,
        #                               learning_rate=learning_rate,
        #
        #                               num_epochs=num_epochs,
        #                               batch_size=batch_size,
        #                               evaluate_step=2000,
        #                               load_net=False,
        #                               test_net=False,
        #                               num_workers=8,
        #                               learning_rate_decay=True,
        #                               learning_rate_decay_epoch=learning_rate_decay_epoch,
        #                               learning_rate_decay_factor=0.1,
        #                               scheduler_name='MultiStepLR',
        #                               top_acc=1,
        #                               data_distributed=False,
        #                               paint_loss=True,
        #                               save_at_each_step=False,
        #                               gradient_clip_value=gradient_clip_value
        #                               )

        i = 4
        exp_name = 'gat_mobilenet_v1_predicted_mask_and_variable_shortcut_net_newinner_newtrain_' + str(
            int(prune_ratio * 100)) + '_' + str(i)
        description = exp_name + '  ' + ''

        checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
        # save the output to log
        print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
        sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'),
                                   sys.stderr)  # redirect std err, if necessary
        learning_rate = 0.05
        momentum = 0.9
        weight_decay = 4e-5
        num_epochs = 300
        print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value, i)
        checkpoint = torch.load(os.path.join(conf.root_path, 'masked_net', 'mobilenet_v1', str(i) + '.pth'),map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])

        net.mask_net()
        net.print_mask()
        net.prune_net()
        net.print_mask()
        net.current_epoch = net.mask_training_stop_epoch + 1
        pruned_flop = net.measure_self_flops()
        print('prune_ratio:', 1 - pruned_flop / total_flop)
        measure_flops.measure_model(net.net)

        net = nn.DataParallel(net)
        net = net.module.net
        net = nn.DataParallel(net)
        net = net.cuda()

        train.train(net=net,
                    net_name='mobilenet_v1',
                    exp_name=exp_name,
                    description=description,
                    dataset_name='imagenet',
                    optimizer=optim.SGD,

                    criterion=train.CrossEntropyLabelSmooth(num_classes=1000, epsilon=0.1).cuda(),

                    learning_rate=learning_rate,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    scheduler_name='CosineAnnealingLR',

                    num_epochs=num_epochs,
                    batch_size=256,
                    evaluate_step=2000,
                    resume=True,
                    test_net=True,
                    num_workers=4,
                    learning_rate_decay=True,
                    top_acc=1,
                    data_parallel=True,
                    paint_loss=False,
                    save_at_each_step=False,
                    gradient_clip_value=gradient_clip_value,
                    use_tensorboard=True
                    )

    else:
        raise AttributeError






















































# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet50_extractor_static_imagenet/checkpoint/flop=1662949668,accuracy=0.91526.tar'),pretrained=False)
# net=nn.DataParallel(net)
# success=train.train(net=net,
#
#             net_name='resnet50',
#             exp_name='resnet50_imagenet_prunednet_retrain_from_scratch',
#
#             num_epochs=200,
#             learning_rate=0.1,
#             learning_rate_decay=True,
#             learning_rate_decay_factor=0.1,
#             weight_decay=1e-4,
#             # learning_rate_decay_epoch=[10,30],
#             learning_rate_decay_epoch=[30,60,130,160],
#             dataset_name='imagenet',
#             batch_size=512,
#             evaluate_step=1000,
#             load_net=True,
#             test_net=True,
#             num_workers=8,
#
#             # target_accuracy=0.9264796074467897,
#             top_acc=5,
#                     data_parallel=True)

# checkpoint=dict()
# checkpoint['structure']=[64,26,52,64,128,128,128,256,52,52,52,52,52]
# checkpoint['dataset_name']='cifar10'
# checkpoint['net_name']='vgg16_bn'
# net=storage.restore_net(checkpoint,pretrained=False)
# measure_flops.measure_model(net,dataset_name='cifar10')
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16_extractor_static_cifar10/checkpoint/flop=48525158,accuracy=0.93140.tar'),pretrained=False)
#
# # net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/vgg16bn_cifar10_prunednet_retrain_from_scratch/checkpoint/flop=48525158,accuracy=0.91900.tar'),pretrained=True)
# # i=0
# # while i<3:
# success = train.train(net=net,
#
#                       net_name='vgg16_bn',
#                       exp_name='vgg16bn_cifar10_afp_retrain_from_scratch',
#
#                       num_epochs=400,
#                       learning_rate=0.1,
#                       learning_rate_decay=True,
#                       learning_rate_decay_factor=0.1,
#                       weight_decay=1e-4,
#                       momentum=0.9,
#                       # learning_rate_decay_epoch=[10,30],
#                       learning_rate_decay_epoch=[2*i for i in [100,130]],
#                       dataset_name='cifar10',
#                       batch_size=128,
#                       evaluate_step=1000,
#                       load_net=True,
#                       test_net=True,
#                       num_workers=2,
#
#                       # target_accuracy=0.9264796074467897,
#                       top_acc=1,
#                       data_parallel=False)
#
#     i+=1


#
# net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_extractor_static_cifar10/checkpoint/flop=71074442,accuracy=0.93330.tar'),pretrained=False)
#
# a=[7,9,9,9,11,7,7,10,10,12,16,11,12,16,9,6,6,15,13,7   ,18,21,15,18,26, 17,25]
# a=[3+i for i in a]
# p_f_s=[16,     a[0],16,a[1],16,a[2],16,a[3],16,a[4],16,a[5],16,a[6],16,a[7],16,a[8],16,
#    a[9],32,a[10],32,a[11],32,a[12],32,a[13],32,a[14],32,a[15],32,a[16],32,a[17],32,
#    a[18],64,a[19],64,a[20],64,a[21],64,a[22],64,a[23],64,a[24],64,a[25],64,a[26],64]
#
# checkpoint=dict()
# checkpoint['structure']=p_f_s
# checkpoint['net_name']='resnet56'
# checkpoint['dataset_name']='cifar10'
# net=storage.restore_net(checkpoint,pretrained=False)
#
# # net=storage.restore_net(torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet56_cifar10_prunednet_retrain_from_scratch/checkpoint/flop=71074442,accuracy=0.91930.tar'),pretrained=True)
# i=0
# # while i<3:
# success = train.train(net=net,
#
#                       net_name='resnet56',
#                       exp_name='tmp',
#
#
#                       learning_rate=0.1,
#                       learning_rate_decay=True,
#                       weight_decay=1e-4,
#                       momentum=0.9,
#
#                        num_epochs=320,
#                       learning_rate_decay_factor=0.1,
#                       # learning_rate_decay_epoch=[10,30],
#                       learning_rate_decay_epoch=[2*i for i in [80,120]],
#
#                       dataset_name='cifar10',
#                       batch_size=128,
#                       evaluate_step=1000,
#                       load_net=True,
#                       test_net=True,
#                       num_workers=4,
#
#                       # target_accuracy=0.9264796074467897,
#                       top_acc=1,
#                       data_parallel=False,
#                       paint_loss=True
#                       )
#     i+=1

# success=False
# while not success and i<1:
    # success=train.train(net=net,
    #
    #             net_name='vgg16_bn',
    #             exp_name='vgg16_extractor_static_imagenet_train',
    #
    #             num_epochs=100,
    #             learning_rate=0.001,
    #             learning_rate_decay=True,
    #             learning_rate_decay_factor=0.1,
    #             weight_decay=1e-4,
    #             # learning_rate_decay_epoch=[10,30],
    #             learning_rate_decay_epoch=[60],
    #             dataset_name='imagenet',
    #             batch_size=512,
    #             evaluate_step=1000,
    #             load_net=True,
    #             test_net=True,
    #             num_workers=8,
    #
    #             target_accuracy=0.9013827754961294,
    #             top_acc=5,)

    # success=train.train(net=net,
    #
    #             net_name='resnet50',
    #             exp_name='resnet50_extractor_static_imagenet_train2',
    #
    #             num_epochs=100,
    #             learning_rate=0.001,
    #             learning_rate_decay=True,
    #             learning_rate_decay_factor=0.1,
    #             weight_decay=1e-4,
    #             # learning_rate_decay_epoch=[10,30],
    #             learning_rate_decay_epoch=[30,60],
    #             dataset_name='imagenet',
    #             batch_size=512,
    #             evaluate_step=1000,
    #             load_net=True,
    #             test_net=True,
    #             num_workers=8,
    #
    #             target_accuracy=0.9264796074467897,
    #             top_acc=5,
    #                     data_parallel=True)
#flop=2495193892,accuracy=0.92332.tar
    # success=train.train(net=net,
    #
    #             net_name='resnet50',
    #             exp_name='resnet50_extractor_static_imagenet_train2',
    #
    #             num_epochs=60,
    #             learning_rate=0.01,
    #             learning_rate_decay=True,
    #             learning_rate_decay_factor=0.1,
    #             weight_decay=1e-4,
    #             # learning_rate_decay_epoch=[10,30],
    #             learning_rate_decay_epoch=[15,30,45],
    #             dataset_name='imagenet',
    #             batch_size=512,
    #             evaluate_step=1000,
    #             load_net=True,
    #             test_net=True,
    #             num_workers=8,
    #
    #             target_accuracy=0.9266966598185884,
    #             top_acc=5,
    #                     data_parallel=True)


    # success=train.train(net=net,
    #             net_name='resnet56',
    #             exp_name='resnet56_extractor_static_cifar100_2'+'_train',
    #
    #                     num_epochs=200,
    #                     learning_rate=0.001,
    #                     learning_rate_decay=True,
    #                     learning_rate_decay_epoch=[60, 120, 160],  # , 150, 250, 300, 350, 400],
    #                     learning_rate_decay_factor=0.1,
    #                     weight_decay=1e-4,
    #
    #
    #             dataset_name='cifar100',
    #             # num_epochs=450,
    #             batch_size=512,
    #             evaluate_step=3000,
    #             load_net=False,
    #             test_net=True,
    #             num_workers=2,
    #
    #             target_accuracy=0.7108089533597423,
    #             top_acc=1)

    # no_grad = set_modules_no_grad(net, ['layer3.block8', 'fc'])
    # success=train.train(net=net,
    #
    #             net_name='resnet56',
    #             exp_name='resnet56_extractor_static_cifar10_only_gcn_1'+'_train',
    #
    #          optimizer=optim.SGD,
    #          learning_rate=0.00001,  # 标准baseline
    #          learning_rate_decay=True,
    #          learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #          learning_rate_decay_factor=0.5,
    #          weight_decay=5e-4,
    #
    #             dataset_name='cifar10',
    #             num_epochs=450,
    #             batch_size=512,
    #             evaluate_step=3000,
    #             load_net=False,
    #             test_net=True,
    #             num_workers=2,
    #
    #             target_accuracy=0.9349782293095876,
    #             top_acc=1,
    #                     no_grad=no_grad)

    # success = train.train(net=net,
    #
    #                       net_name='resnet56',
    #                       exp_name='resnet56_extractor_static_cifar10_only_gcn_3' + '_train',
    #
    #                       optimizer=optim.SGD,
    #                       learning_rate=0.0001,  # 标准baseline
    #                       learning_rate_decay=True,
    #                       learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #                       learning_rate_decay_factor=0.5,
    #                       weight_decay=5e-4,
    #
    #                       dataset_name='cifar10',
    #                       num_epochs=450,
    #                       batch_size=512,
    #                       evaluate_step=3000,
    #                       load_net=False,
    #                       test_net=True,
    #                       num_workers=2,
    #
    #                       target_accuracy=0.9349782293095876,
    #                       top_acc=1)


    # i+=1
