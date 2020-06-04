import torch
from torch import nn
from framework import data_loader, train, evaluate,measure_flops
from network import vgg_channel_weight, vgg,storage,resnet,net_with_predicted_mask,resnet_cifar,modules,resnet_cifar
from framework import config as conf
import os,sys
from filter_characteristic import filter_feature_extractor,predict_dead_filter
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import cgd
import logger
#ssh -L 16006:127.0.0.1:6006 -p 20029 victorfang@210.28.133.13
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = optim.SGD
learning_rate = {'default': 0.1, 'extractor': 0.1}
weight_decay = {'default':5e-4,'extractor':5e-4}
momentum = {'default':0.9,'extractor':0.9}
# optimizer = optim.Adam
# learning_rate = {'default': 0.01, 'extractor': 0.01}
exp_name='resnet56_predicted_mask_shortcut_with_weight_pruneFirstConv_wd5_sgd_blpenaltyNo_smTrain_4'

mask_update_freq = 10
mask_update_epochs = 1
batch_size=128
mask_training_start_epoch=10
mask_training_stop_epoch=80


flop_expected=6.75e7
gradient_clip_value=1



# learning_rate_decay_epoch = [mask_training_stop_epoch+2*i for i in [80,120]]
# num_epochs=160*2+mask_training_stop_epoch


net=resnet_cifar.resnet56(num_classes=10).to(device)

net = net_with_predicted_mask.predicted_mask_shortcut_with_weight_net(net,
                                                                      net_name='resnet56',
                                                                      dataset_name='cifar10',
                                                                      mask_update_epochs=mask_update_epochs,
                                                                      mask_update_freq=mask_update_freq,
                                                                      flop_expected=flop_expected,
                                                                      gcn_rounds=2,
                                                                      mask_training_start_epoch=mask_training_start_epoch,
                                                                      mask_training_stop_epoch=mask_training_stop_epoch,
                                                                      batch_size=batch_size
                                                                      )




net=net.to(device)
checkpoint_path = os.path.join(conf.root_path, 'model_saved', exp_name)
# save the output to log
print('save log in:' + os.path.join(checkpoint_path, 'log.txt'))
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary

print(optimizer, weight_decay, momentum, learning_rate, mask_update_freq, mask_update_epochs, flop_expected, gradient_clip_value)

# train.add_forward_hook(net,module_name='net.features.1')

train.train(net=net,
            net_name='resnet56',
            exp_name=exp_name,
            dataset_name='cifar10',

            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum,
            learning_rate=learning_rate,

            num_epochs=81,
            batch_size=batch_size,
            evaluate_step=5000,
            load_net=False,
            test_net=False,
            num_workers=8,
            # weight_decay=5e-4,
            learning_rate_decay=False,
            learning_rate_decay_factor=0.1,
            scheduler_name='MultiStepLR',
            top_acc=1,
            data_parallel=False,
            paint_loss=True,
            save_at_each_step=False,
            gradient_clip_value=gradient_clip_value
            )

for name,mod in net.named_modules():
    if isinstance(mod,modules.block_with_mask_shortcut):
        mod.shortcut_mask=nn.Parameter(mod.shortcut_mask,requires_grad=True)

train.train(net=net,
            net_name='resnet56',
            exp_name=exp_name,
            dataset_name='cifar10',

            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum,
            learning_rate=learning_rate,

            num_epochs=320,
            batch_size=batch_size,
            evaluate_step=5000,
            load_net=False,
            test_net=False,
            num_workers=8,
            # weight_decay=5e-4,
            learning_rate_decay=True,
            learning_rate_decay_epoch=[160,240],
            learning_rate_decay_factor=0.1,
            scheduler_name='MultiStepLR',
            top_acc=1,
            data_parallel=False,
            paint_loss=True,
            save_at_each_step=False,
            gradient_clip_value=gradient_clip_value
            )
