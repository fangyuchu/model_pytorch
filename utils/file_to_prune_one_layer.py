import torch
from torch import nn
import torch.optim as optim
from prune import prune_and_train,prune_module
from framework import evaluate,data_loader,measure_flops,train
from network import vgg,storage,net_with_predicted_mask,resnet_cifar,resnet_cifar
from framework import config as conf
from framework.train import set_modules_no_grad
import os,sys,logger,copy
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_net=vgg.vgg11_bn(dataset_name='cifar10')
layer=7

checkpoint_path = os.path.join(conf.root_path, 'model_saved', 'log_for_prune_one_layer')
# save the output to log
print('save log in:' + os.path.join(checkpoint_path, 'log'+str(layer)+'.txt'))
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log'+str(layer)+'.txt'), sys.stdout)
sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log'+str(layer)+'.txt'), sys.stderr)  # redirect std err, if necessary

print('prune layer:',layer)
for i in range(0,100,10):
    net=copy.deepcopy(original_net)
    net.to(device)
    l=0
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            if l==layer:
                conv_num=mod.out_channels
            l+=1
    num_conv_to_prune=int(conv_num*(float(i)/100))
    prune_module.prune_conv(net,'vgg11_bn',layer,[j for j in range(num_conv_to_prune)])
    print('ratio set to ',i,'%')
    print('prune',num_conv_to_prune,' filters')
    train.train(net=net,
                net_name='resnet56',
                exp_name='vgg11_bn_layer'+str(layer)+'_'+str(i)+'%pruned',
                description='random prune some filters in designated layer',
                dataset_name='cifar10',
                optimizer=optim.SGD,
                weight_decay=1e-4,
                momentum=0.9,
                learning_rate=0.1,
                num_epochs=160,
                batch_size=128,
                evaluate_step=5000,
                load_net=False,
                test_net=False,
                num_workers=4,
                learning_rate_decay=True,
                learning_rate_decay_epoch=[80,120],
                learning_rate_decay_factor=0.1,
                scheduler_name='MultiStepLR',
                top_acc=1,
                data_parallel=False,
                paint_loss=False,
                save_at_each_step=False,
                gradient_clip_value=None
                )