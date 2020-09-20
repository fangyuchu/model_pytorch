import os,sys
sys.path.append('../')
import torch
from torch import nn
import torch.optim as optim
from framework import evaluate,data_loader,measure_flops,train
from network import vgg,storage,net_with_predicted_mask,resnet_cifar,resnet_cifar,resnet
from framework import config as conf
import logger
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# total_flop=125485706
# prune_ratio=0.90
# flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
# gradient_clip_value=None
# learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
# num_epochs=160*1+mask_training_stop_epoch
#
# l=[0.0001,0.5]
# l+=list(range(1,10))
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
#                                                                            gcn_rounds=2,
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

#gcn_round
optimizer_net = optim.SGD
optimizer_extractor = optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.001}
weight_decay = {'default': 1e-4, 'extractor': 5e-4}
momentum = {'default': 0.9, 'extractor': 0.9}
batch_size=128
#网络参数
mask_update_freq = 1000
mask_update_epochs = 900
mask_training_start_epoch=1
mask_training_stop_epoch=20

total_flop=125485706
prune_ratio=0.90
flop_expected=total_flop*(1 - prune_ratio)#0.627e7#1.25e7#1.88e7#2.5e7#3.6e7#
gradient_clip_value=None
learning_rate_decay_epoch = [mask_training_stop_epoch+1*i for i in [80,120]]
num_epochs=160*1+mask_training_stop_epoch

r=[1,3,4,5]
for num_round in r:
    print("gcn_round: ",num_round)
    net=resnet_cifar.resnet56(num_classes=10).to(device)
    net = net_with_predicted_mask.predicted_mask_and_variable_shortcut_net(net,
                                                                           net_name='resnet56',
                                                                           dataset_name='cifar10',
                                                                           mask_update_epochs=mask_update_epochs,
                                                                           mask_update_freq=mask_update_freq,
                                                                           flop_expected=flop_expected,
                                                                           gcn_rounds=num_round,
                                                                           mask_training_start_epoch=mask_training_start_epoch,
                                                                           mask_training_stop_epoch=mask_training_stop_epoch,
                                                                           batch_size=batch_size,
                                                                           add_shortcut_ratio=0.9
                                                                           )
    net=net.to(device)
    exp_name = 'ablation_gcn_round_predicted_mask_and_variable_shortcut_net/resnet56_gcnround_'+str(num_round)
    description = exp_name + '  ' + ''

    checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
    # save the output to log
    print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
    sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
    print(weight_decay, momentum, learning_rate, flop_expected, gradient_clip_value,num_round)

    if num_round!=1:
        train.train_extractor_network(net=net,
                                      net_name='resnet56',
                                      exp_name=exp_name,
                                      description=description,
                                      dataset_name='cifar10',

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
                                      paint_loss=True,
                                      save_at_each_step=False,
                                      gradient_clip_value=gradient_clip_value
                                      )
    if num_round==1:
        c=torch.load('/home/disk_new/model_saved/ablation_gcn_round_predicted_mask_and_variable_shortcut_net/resnet56_gcnround_1/checkpoint/masked_net.tar')
        net.load_state_dict(c['state_dict'])

    net.mask_net()
    net.print_mask()
    net.prune_net()
    net.current_epoch = net.mask_training_stop_epoch + 1
    learning_rate_decay_epoch = [1*i for i in [80,120]]
    num_epochs = 160*1
    exp_name+='_train'
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
                load_net=False,
                test_net=True,
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