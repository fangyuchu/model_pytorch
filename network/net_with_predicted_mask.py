from filter_characteristic import filter_feature_extractor
from prune.prune_module import get_module
from network.modules import conv2d_with_mask
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import datetime
import math
import matplotlib.pyplot as plt
from framework import data_loader, measure_flops, evaluate, train, config as conf
from math import ceil
from network import storage
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"





class predicted_mask_net(nn.Module):
    def __init__(self,net,net_name,dataset_name,feature_len=15,gcn_rounds=2,only_gcn=False,only_inner_features=False,mask_update_freq=40,mask_update_steps=10):
        '''
        Use filter feature extractor to extract features from a cnn and predict mask for it. The mask will guide the
        cnn to skip filters when forwarding. Both extractor and cnn are updated through back-propagation.
        :param net:
        :param net_name:
        :param dataset_name:
        :param feature_len: expected length of features to extract
        :param gcn_rounds:
        :param only_gcn:
        :param only_inner_features:
        :param mask_update_freq: how often does the extractor being updated. The extractor will be updated every mask_update_freq STEPs!
        :param mask_update_steps: update mask for mask_update_steps STEPs
        '''
        super(predicted_mask_net, self).__init__()
        self.net=self.transform(net)#.to(device)
        self.extractor=filter_feature_extractor.extractor(feature_len=feature_len,gcn_rounds=gcn_rounds,only_gcn=only_gcn,only_inner_features=only_inner_features)
        self.net_name=net_name
        self.dataset_name=dataset_name
        self.feature_len=feature_len
        self.gcn_rounds=gcn_rounds
        # self.data_parallel=True
        self.mask_update_freq=mask_update_freq
        self.mask_update_steps=mask_update_steps
        self.step_tracked=0

    def train(self, mode=True):
        super().train(mode)
        if mode is False:
            self.update_mask()  # update the mask when being switched to eval mode
        return self

    def eval(self):
        return self.train(False)

    def copy(self):
        '''
        self-made deepcopy
        :return: a cloned network
        '''
        checkpoint = storage.get_net_information(self, self.dataset_name, self.net_name)
        copied_net = storage.restore_net(checkpoint, pretrained=True)
        copied_net.to(self.extractor.network[0].weight.device)
        return copied_net

    def transform(self,net):
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                device=mod.weight.device
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = conv2d_with_mask(mod).to(device)  # replace conv
            if isinstance(mod,nn.BatchNorm2d):
                mod.track_running_stats=False                                     #track running stats for bn will cause huge shake when a filter is pruned
        return net

    def update_mask(self):
        mask = self.extractor(self, self.net_name, self.dataset_name)           #predict mask using extractor
        # print(mask.reshape(-1))
        lo = hi = 0
        for name, mod in self.net.named_modules():
            if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                hi += mod.out_channels
                _modules = get_module(model=self.net, name=name)
                # mod.mask[:]=mask[lo:hi].view(-1)                   #update mask for each conv
                mod.mask=mask[lo:hi].view(-1)                   #update mask for each conv

                channel_num=torch.sum(mod.mask!=0)                #code for debug
                if self.training is True:
                    if self.step_tracked % self.mask_update_freq ==1 or  self.step_tracked % self.mask_update_freq==self.mask_update_steps:
                        print(channel_num)                                #print the number of channels without being pruned

                if torch.sum(mod.mask==0)==mod.out_channels:
                    raise Exception('all filters are pruned')
                lo = hi

    def forward(self, input):
        # if self.training is True or self.data_parallel is True:
        if self.training is True:
            self.step_tracked += 1
            if self.step_tracked % self.mask_update_freq <= self.mask_update_steps:
                self.update_mask()  # mask only need to be updated when training.
            else:
                for name, mod in self.net.named_modules():
                    if isinstance(mod, conv2d_with_mask) and 'downsample' not in name:
                        mod.mask = mod.mask.detach()  # detach masks from computation graph so the extractor will not be updated
        return self.net(input)


# def train(
#         net,
#         net_name,
#         cnn_learning_rate=conf.learning_rate,
#         cnn_momentum=conf.momentum,
#         cnn_weight_decay=conf.weight_decay,
#         cnn_optimizer=optim.SGD,
#
#         extractor_learning_rate=conf.learning_rate,
#         extractor_momentum=conf.momentum,
#         extractor_weight_decay=conf.weight_decay,
#         extractor_optimizer=optim.SGD,
#
#         exp_name='',
#         dataset_name='imagenet',
#         train_loader=None,
#         validation_loader=None,
#         num_epochs=conf.num_epochs,
#         batch_size=conf.batch_size,
#         evaluate_step=conf.evaluate_step,
#         load_net=True,
#         test_net=False,
#         root_path=conf.root_path,
#         checkpoint_path=None,
#         num_workers=conf.num_workers,
#         learning_rate_decay=False,
#         learning_rate_decay_factor=conf.learning_rate_decay_factor,
#         learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
#         target_accuracy=1.0,
#         top_acc=1,
#         criterion=nn.CrossEntropyLoss(),  # 损失函数默为交叉熵，多用于多分类问题
#         no_grad=[],
#         scheduler_name='MultiStepLR',
#         eta_min=0,
#         paint_loss=False,
#         # todo:tmp!!!
#         data_parallel=False,
#
# ):
#     '''
#
#     :param net: net to be trained
#     :param net_name: name of the net
#     :param exp_name: name of the experiment
#     :param dataset_name: name of the dataset
#     :param train_loader: data_loader for training. If not provided, a data_loader will be created based on dataset_name
#     :param validation_loader: data_loader for validation. If not provided, a data_loader will be created based on dataset_name
#     :param learning_rate: initial learning rate
#     :param learning_rate_decay: boolean, if true, the learning rate will decay based on the params provided.
#     :param learning_rate_decay_factor: float. learning_rate*=learning_rate_decay_factor, every time it decay.
#     :param learning_rate_decay_epoch: list[int], the specific epoch that the learning rate will decay.
#     :param num_epochs: max number of epochs for training
#     :param batch_size:
#     :param evaluate_step: how often will the net be tested on validation set. At least one test every epoch is guaranteed
#     :param load_net: boolean, whether loading net from previous checkpoint. The newest checkpoint will be selected.
#     :param test_net:boolean, if true, the net will be tested before training.
#     :param root_path:
#     :param checkpoint_path:
#     :param momentum:
#     :param num_workers:
#     :param weight_decay:
#     :param target_accuracy:float, the training will stop once the net reached target accuracy
#     :param optimizer:
#     :param top_acc: can be 1 or 5
#     :param criterion： loss function
#     :param no_grad: list containing names of the modules that do not need to be trained
#     :param scheduler_name
#     :param eta_min: for CosineAnnealingLR
#     :return:
#     '''
#     success = True  # if the trained net reaches target accuracy
#     # gpu or not
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using: ', end='')
#     if torch.cuda.is_available():
#         print(torch.cuda.device_count(), ' * ', end='')
#         print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print(device)
#
#     # prepare the data
#     if dataset_name is 'imagenet':
#         mean = conf.imagenet['mean']
#         std = conf.imagenet['std']
#         train_set_path = conf.imagenet['train_set_path']
#         train_set_size = conf.imagenet['train_set_size']
#         validation_set_path = conf.imagenet['validation_set_path']
#         default_image_size = conf.imagenet['default_image_size']
#     elif dataset_name is 'cifar10':
#         train_set_size = conf.cifar10['train_set_size']
#         mean = conf.cifar10['mean']
#         std = conf.cifar10['std']
#         train_set_path = conf.cifar10['dataset_path']
#         validation_set_path = conf.cifar10['dataset_path']
#         default_image_size = conf.cifar10['default_image_size']
#     elif dataset_name is 'tiny_imagenet':
#         train_set_size = conf.tiny_imagenet['train_set_size']
#         mean = conf.tiny_imagenet['mean']
#         std = conf.tiny_imagenet['std']
#         train_set_path = conf.tiny_imagenet['train_set_path']
#         validation_set_path = conf.tiny_imagenet['validation_set_path']
#         default_image_size = conf.tiny_imagenet['default_image_size']
#     elif dataset_name is 'cifar100':
#         train_set_size = conf.cifar100['train_set_size']
#         mean = conf.cifar100['mean']
#         std = conf.cifar100['std']
#         train_set_path = conf.cifar100['dataset_path']
#         validation_set_path = conf.cifar100['dataset_path']
#         default_image_size = conf.cifar100['default_image_size']
#     if train_loader is None:
#         train_loader = data_loader.create_train_loader(dataset_path=train_set_path,
#                                                        default_image_size=default_image_size,
#                                                        mean=mean,
#                                                        std=std,
#                                                        batch_size=batch_size,
#                                                        num_workers=num_workers,
#                                                        dataset_name=dataset_name)
#     if validation_loader is None:
#         validation_loader = data_loader.create_validation_loader(dataset_path=validation_set_path,
#                                                                  default_image_size=default_image_size,
#                                                                  mean=mean,
#                                                                  std=std,
#                                                                  batch_size=batch_size,
#                                                                  num_workers=num_workers,
#                                                                  dataset_name=dataset_name)
#
#     if checkpoint_path is None:
#         checkpoint_path = os.path.join(root_path, 'model_saved', exp_name, 'checkpoint')
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path, exist_ok=True)
#
#     # get the latest checkpoint
#     lists = os.listdir(checkpoint_path)
#     file_new = checkpoint_path
#     if len(lists) > 0:
#         lists.sort(key=lambda fn: os.path.getmtime(checkpoint_path + "/" + fn))  # 按时间排序
#         file_new = os.path.join(checkpoint_path, lists[-1])  # 获取最新的文件保存到file_new
#
#     sample_num = 0
#     if os.path.isfile(file_new):
#         if load_net:
#             checkpoint = torch.load(file_new)
#             print('{} load net from previous checkpoint:{}'.format(datetime.now(), file_new))
#             net = storage.restore_net(checkpoint, pretrained=True, data_parallel=data_parallel)
#             sample_num = checkpoint['sample_num']
#
#     if test_net:
#         print('{} test the net'.format(datetime.now()))  # no previous checkpoint
#         accuracy = evaluate.evaluate_net(net, validation_loader,
#                                          save_net=True,
#                                          checkpoint_path=checkpoint_path,
#                                          sample_num=sample_num,
#                                          target_accuracy=target_accuracy,
#                                          dataset_name=dataset_name,
#                                          top_acc=top_acc,
#                                          net_name=net_name,
#                                          exp_name=exp_name
#                                          )
#
#         if accuracy >= target_accuracy:
#             print('{} net reached target accuracy.'.format(datetime.now()))
#             return success
#
#     # ensure the net will be evaluated despite the inappropriate evaluate_step
#     if evaluate_step > math.ceil(train_set_size / batch_size) - 1:
#         evaluate_step = math.ceil(train_set_size / batch_size) - 1
#
#     optimizer = train.prepare_optimizer(net, optimizer, no_grad, momentum, learning_rate, weight_decay)
#     if learning_rate_decay:
#         if scheduler_name == 'MultiStepLR':
#             scheduler = lr_scheduler.MultiStepLR(optimizer,
#                                                  milestones=learning_rate_decay_epoch,
#                                                  gamma=learning_rate_decay_factor,
#                                                  last_epoch=ceil(sample_num / train_set_size))
#         elif scheduler_name == 'CosineAnnealingLR':
#             scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
#                                                        num_epochs,
#                                                        eta_min=eta_min,
#                                                        last_epoch=ceil(sample_num / train_set_size))
#     loss_list = []
#     acc_list = []
#     xaxis_loss = []
#     xaxis_acc = []
#     xaxis = 0
#     print("{} Start training ".format(datetime.now()) + net_name + "...")
#     for epoch in range(math.floor(sample_num / train_set_size), num_epochs):
#         print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
#         net.train()
#         # one epoch for one loop
#         for step, data in enumerate(train_loader, 0):
#             # if step==0 and epoch==0:      # debug code
#             #     old_data=data             #use the same batch of data over and over again
#             # data=old_data                 #the loss should decrease if the net is defined properly
#
#             xaxis += 1
#             if sample_num / train_set_size == epoch + 1:  # one epoch of training finished
#                 accuracy = evaluate.evaluate_net(net, validation_loader,
#                                                  save_net=True,
#                                                  checkpoint_path=checkpoint_path,
#                                                  sample_num=sample_num,
#                                                  target_accuracy=target_accuracy,
#                                                  dataset_name=dataset_name,
#                                                  top_acc=top_acc,
#                                                  net_name=net_name,
#                                                  exp_name=exp_name)
#                 if accuracy >= target_accuracy:
#                     print('{} net reached target accuracy.'.format(datetime.now()))
#                     return success
#                 break
#
#             # 准备数据
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             sample_num += int(images.shape[0])
#
#             optimizer.zero_grad()
#             # forward + backward
#             outputs = net(images)
#             loss = criterion(outputs, labels)
#             # loss2=torch.zeros(1).to(images.device)
#             # for name, mod in net.named_modules():
#             #     if isinstance(mod, net_with_predicted_mask.conv2d_with_mask) and 'downsample' not in name:
#             #         loss2-=torch.sum(mod.mask)
#             #
#             # loss=loss+0.001*loss2
#
#             loss.backward()
#             optimizer.step()
#
#             loss_list += [float(loss.detach())]
#             xaxis_loss += [xaxis]
#
#             if step % 20 == 0:
#                 print('{} loss is {}'.format(datetime.now(), float(loss.data)))
#
#             if step % evaluate_step == 0 and step != 0:
#                 accuracy = evaluate.evaluate_net(net, validation_loader,
#                                                  save_net=True,
#                                                  checkpoint_path=checkpoint_path,
#                                                  sample_num=sample_num,
#                                                  target_accuracy=target_accuracy,
#                                                  dataset_name=dataset_name,
#                                                  top_acc=top_acc,
#                                                  net_name=net_name,
#                                                  exp_name=exp_name)
#                 if accuracy >= target_accuracy:
#                     print('{} net reached target accuracy.'.format(datetime.now()))
#                     return success
#                 accuracy = float(accuracy)
#
#                 acc_list += [accuracy]
#                 xaxis_acc += [xaxis]
#
#                 if paint_loss:
#                     fig, ax1 = plt.subplots()
#                     ax2 = ax1.twinx()
#                     ax1.plot(xaxis_loss, loss_list, 'g')
#                     ax2.plot(xaxis_acc, acc_list, 'b')
#                     ax1.set_xlabel('step')
#                     ax1.set_ylabel('loss')
#                     ax2.set_ylabel('accuracy')
#                     plt.title(exp_name)
#                     plt.savefig(os.path.join(root_path, 'model_saved', exp_name, 'train.png'))
#                     plt.show()
#
#                 print('{} continue training'.format(datetime.now()))
#         if learning_rate_decay:
#             scheduler.step()
#             print(optimizer.state_dict()['param_groups'][0]['lr'])
#
#     print("{} Training finished. Saving net...".format(datetime.now()))
#     flop_num = measure_flops.measure_model(net=net, dataset_name=dataset_name, print_flop=False)
#     accuracy = evaluate.evaluate_net(net, validation_loader,
#                                      save_net=True,
#                                      checkpoint_path=checkpoint_path,
#                                      sample_num=sample_num,
#                                      target_accuracy=target_accuracy,
#                                      dataset_name=dataset_name,
#                                      top_acc=top_acc,
#                                      net_name=net_name,
#                                      exp_name=exp_name)
#     accuracy = float(accuracy)
#     checkpoint = {
#         'highest_accuracy': accuracy,
#         'state_dict': net.state_dict(),
#         'sample_num': sample_num,
#         'flop_num': flop_num}
#     checkpoint.update(storage.get_net_information(net, dataset_name, net_name))
#     torch.save(checkpoint, '%s/flop=%d,accuracy=%.5f.tar' % (checkpoint_path, flop_num, accuracy))
#     print("{} net saved at sample num = {}".format(datetime.now(), sample_num))
#     return not success
