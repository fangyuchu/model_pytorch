import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from framework import data_loader, measure_flops, train, evaluate, config as conf
from prune import prune_module
import logger
import sys
import random
import copy
from filter_characteristic import predict_dead_filter
from filter_characteristic.filter_feature_extractor import train_extractor,load_extractor
from network import resnet_cifar, resnet_tinyimagenet,storage,resnet,vgg,resnet
from network.storage import  restore_net
import math


def prune_inactive_neural_with_regressor(net,
                                         net_name,
                                         target_accuracy,
                                         prune_rate,
                                         exp_name,
                                         load_regressor=False,
                                         predictor_name='random_forest',
                                         round_for_train=2,
                                         tar_acc_gradual_decent=False,
                                         flop_expected=None,
                                         dataset_name='imagenet',
                                         use_random_data=False,
                                         validation_loader=None,
                                         batch_size=conf.batch_size,
                                         num_workers=conf.num_workers,
                                         optimizer=optim.Adam,
                                         learning_rate=0.01,
                                         evaluate_step=1000,
                                         num_epoch=450,
                                         filter_preserve_ratio=0.3,
                                         max_filters_pruned_for_one_time=0.5,
                                         learning_rate_decay=False,
                                         learning_rate_decay_factor=conf.learning_rate_decay_factor,
                                         weight_decay=conf.weight_decay,
                                         learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
                                         top_acc=1,
                                         max_training_round=99999,
                                         round=1,
                                         **kwargs
                                     ):
    '''

    :param net:
    :param net_name:
    :param exp_name:
    :param target_accuracy:
    :param prune_rate:
    :param load_regressor:
    :param predictor_name:
    :param round_for_train:
    :param tar_acc_gradual_decent:
    :param flop_expected:
    :param dataset_name:
    :param use_random_data:
    :param validation_loader:
    :param batch_size:
    :param num_workers:
    :param optimizer:
    :param learning_rate:
    :param evaluate_step:
    :param num_epoch:
    :param filter_preserve_ratio:
    :param max_filters_pruned_for_one_time: If a float provided, it should be slightly larger than prune_rate. If a list object provided, it
    should contain max_prune_ratio_for_one_time for every layer
    :param learning_rate_decay:
    :param learning_rate_decay_factor:
    :param weight_decay:
    :param learning_rate_decay_epoch:
    :param round: the start number of round
    :param kwargs:
    :return:
    '''
    # save the output to log
    print('save log in:' + os.path.join(conf.root_path,'model_saved',exp_name,'log.txt'))
    if not os.path.exists(os.path.join(conf.root_path,'model_saved',exp_name)):
        os.makedirs(os.path.join(conf.root_path,'model_saved',exp_name), exist_ok=True)
    sys.stdout = logger.Logger(os.path.join(conf.root_path,'model_saved',exp_name,'log.txt'), sys.stdout)
    sys.stderr = logger.Logger(os.path.join(conf.root_path,'model_saved',exp_name,'log.txt'), sys.stderr)  # redirect std err, if necessary

    print(
        'net:{}\n' 
        'net_name:{}\n' 
        'exp_name:{}\n'
        'target_accuracy:{}\n' 
        'prune_rate:{}\n' 
        'predictor_name:{}\n' 
        'load_regressor:{}\n'
        'round_for_train:{}\n' 
        'tar_acc_gradual_decent:{}\n' 
        'flop_expected:{}\n' 
        'dataset_name:{}\n' 
        'use_random_data:{}\n' 
        'validation_loader:{}\n' 
        'batch_size:{}\n' 
        'num_workers:{}\n' 
        'optimizer:{}\n' 
        'learning_rate:{}\n' 
        'evaluate_step:{}\n' 
        'num_epoch:{}\n' 
        'filter_preserve_ratio:{}\n' 
        'max_filters_pruned_for_one_time:{}\n' 
        'learning_rate_decay:{}\n' 
        'learning_rate_decay_factor:{}\n' 
        'weight_decay:{}\n' 
        'learning_rate_decay_epoch:{}\n'
        'top_acc:{}\n'
        'max_training_round:{}\n'
        'round:{}\n'
          .format(net, net_name,exp_name, target_accuracy, prune_rate,predictor_name,load_regressor,round_for_train,tar_acc_gradual_decent,
                  flop_expected,dataset_name,use_random_data,validation_loader,batch_size,num_workers,optimizer,learning_rate,
                  evaluate_step,num_epoch,filter_preserve_ratio,max_filters_pruned_for_one_time,learning_rate_decay,learning_rate_decay_factor,
                  weight_decay,learning_rate_decay_epoch,top_acc,max_training_round,round))
    print(kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.device_count(),' * ',end='')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)

    if validation_loader is None:
        validation_loader = data_loader.create_validation_loader(
                                                                 batch_size=batch_size,
                                                                 num_workers=num_workers,
                                                                 dataset_name=dataset_name,
                                                                 )

    flop_original_net = measure_flops.measure_model(net, dataset_name)
    original_accuracy = evaluate.evaluate_net(net=net,
                                              data_loader=validation_loader,
                                              save_net=False,
                                              top_acc=top_acc,
                                              dataset_name=dataset_name,
                                              )
    if tar_acc_gradual_decent is True:
        flop_drop_expected = flop_original_net - flop_expected
        acc_drop_tolerance = original_accuracy - target_accuracy

    num_conv = 0  # num of conv layers in the net
    filter_num_lower_bound = []
    filter_num = []
    for mod in net.modules():
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            num_conv += 1
            filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))
            filter_num.append(mod.out_channels)

    filter=[]
    filter_layer=[]
    FIRE=[]
    predictor = predict_dead_filter.predictor(name=predictor_name)
    if load_regressor is True:
        regressor_exists=predictor.load(path=os.path.join(conf.root_path,'model_saved' , exp_name))
        if regressor_exists is True:
            round_for_train = -1
        else:                                                   #load data from previous rounds of pruning
            print('Can\' find saved regressor. Load data from previous round.')
            filter_tmp, FIRE_tmp, filter_layer_tmp= predict_dead_filter.read_data(path=os.path.join(conf.root_path,'model_saved' , exp_name , 'dead_neural/'),
                                                                                        balance=False,
                                                                                        regression_or_classification='regression',
                                                                                        batch_size=batch_size,
                                                                                        )
            FIRE+=FIRE_tmp
            filter+=filter_tmp
            filter_layer+=filter_layer_tmp

    #using data to prune the net for (round_for_train)rounds

    while True:
        print('{} start round {} of filter pruning.'.format(datetime.now(), round))

        if round<=round_for_train:
            dead_filter_index, module_list, neural_list, FIRE_tmp = evaluate.find_useless_filters_data_version(net=net,
                                                                                                                     batch_size=batch_size,
                                                                                                                     use_random_data=use_random_data,
                                                                                                                     percent_of_inactive_filter=prune_rate,
                                                                                                                     dead_or_inactive='inactive',
                                                                                                                     dataset_name=dataset_name)
            if not os.path.exists(os.path.join(conf.root_path,'model_saved' , exp_name , 'dead_neural')):
                os.makedirs(os.path.join(conf.root_path,'model_saved' , exp_name , 'dead_neural'), exist_ok=True)
            checkpoint={'prune_rate': prune_rate,'module_list': module_list,
                        'neural_list': neural_list, 'state_dict': net.state_dict(),
                        'batch_size': batch_size}
            checkpoint.update(storage.get_net_information(net,dataset_name,net_name))
            torch.save(checkpoint,
                       os.path.join(conf.root_path,'model_saved',exp_name,'dead_neural/round %d.tar' % round)
                       )

            FIRE+=FIRE_tmp
            # save filters for training the regressor
            i = 0
            for mod in net.modules():
                if isinstance(mod, torch.nn.modules.conv.Conv2d):
                    conv_weight = mod.weight.cpu().detach().numpy()
                    for weight in conv_weight:
                        filter.append(weight)
                    filter_layer += [i for j in range(conv_weight.shape[0])]


                    i += 1
            if round == round_for_train:
                # use filters from (round_for_train)rounds to train the regressor
                ##train the predictor######################################################################################################
                predictor.fit(filter=filter, filter_layer=filter_layer, filter_label=FIRE)
                predictor.save(path=os.path.join(conf.root_path, 'model_saved', exp_name))
        else:
            dead_filter_index= evaluate.find_useless_filters_regressor_version(net=net,
                                                                               predictor=predictor,
                                                                               percent_of_inactive_filter=prune_rate,
                                                                               max_filters_pruned_for_one_time=max_filters_pruned_for_one_time
                                                                               )

        net_compressed = False
        #prune the net according to dead_filter_index
        for i in range(num_conv):
            # ensure the number of filters pruned will not be too large for one time
            if type(max_filters_pruned_for_one_time) is list:
                num_filters_to_prune_max=filter_num[i] * max_filters_pruned_for_one_time[i]
            else:
                num_filters_to_prune_max=filter_num[i] * max_filters_pruned_for_one_time
            if num_filters_to_prune_max < len(dead_filter_index[i]):
                dead_filter_index[i] = dead_filter_index[i][
                                       :int(num_filters_to_prune_max)]
            # ensure the lower bound of filter number
            if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
                dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]

            filter_num[i] = filter_num[i] - len(dead_filter_index[i])
            if len(dead_filter_index[i]) > 0:
                net_compressed = True
            print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i],
                                                                          len(dead_filter_index[i])))

            net = prune_module.prune_conv_layer_vgg(model=net, layer_index=i,
                                                filter_index=dead_filter_index[i])  # prune the dead filter

        if net_compressed is False:
            round -= 1
            print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
            continue
        round+=1
        flop_pruned_net = measure_flops.measure_model(net, dataset_name)

        if tar_acc_gradual_decent is True:  # decent the target_accuracy
            flop_reduced = flop_original_net - flop_pruned_net
            target_accuracy =original_accuracy - acc_drop_tolerance * (flop_reduced / flop_drop_expected)
            print('{} current target accuracy:{}'.format(datetime.now(), target_accuracy))

        success = False
        while not success:
            old_net = copy.deepcopy(net).to(torch.device('cpu'))                                    #save cuda memory
            success = train.train(net=net,
                                  net_name=net_name,
                                  exp_name=exp_name,
                                  num_epochs=num_epoch,
                                  target_accuracy=target_accuracy,
                                  learning_rate=learning_rate,
                                  load_net=False,
                                  evaluate_step=evaluate_step,
                                  dataset_name=dataset_name,
                                  optimizer=optimizer,
                                  batch_size=batch_size,
                                  learning_rate_decay=learning_rate_decay,
                                  learning_rate_decay_factor=learning_rate_decay_factor,
                                  weight_decay=weight_decay,
                                  learning_rate_decay_epoch=learning_rate_decay_epoch,
                                  test_net=True,
                                  top_acc=top_acc,
                                  **kwargs
                                  )
            if not success:
                net = old_net.to(device)
                max_training_round -= 1
                if max_training_round == 0:
                    print('{} net can\'t reach target accuracy, pruning stop.'.format(datetime.now()))
                    return


def prune_inactive_neural_with_regressor_resnet(net,
                                                net_name,
                                                exp_name,
                                                target_accuracy,
                                                prune_rate,
                                                load_regressor=False,
                                                predictor_name='random_forest',
                                                round_for_train=2,
                                                tar_acc_gradual_decent=False,
                                                flop_expected=None,
                                                dataset_name='imagenet',
                                                use_random_data=False,
                                                validation_loader=None,
                                                batch_size=conf.batch_size,
                                                num_workers=conf.num_workers,
                                                optimizer=optim.Adam,
                                                learning_rate=0.01,
                                                evaluate_step=1200,
                                                num_epoch=450,
                                                filter_preserve_ratio=0.3,
                                                max_filters_pruned_for_one_time=0.5,
                                                learning_rate_decay=False,
                                                learning_rate_decay_factor=conf.learning_rate_decay_factor,
                                                weight_decay=conf.weight_decay,
                                                learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
                                                max_training_round=9999,
                                                round=1,
                                                top_acc=1,
                                                **kwargs
                                 ):
    '''

    :param net:
    :param net_name:
    :param exp_name:
    :param target_accuracy:
    :param prune_rate:
    :param load_regressor:
    :param predictor_name:
    :param round_for_train:
    :param tar_acc_gradual_decent:
    :param flop_expected:
    :param dataset_name:
    :param use_random_data:
    :param validation_loader:
    :param batch_size:
    :param num_workers:
    :param optimizer:
    :param learning_rate:
    :param evaluate_step:
    :param num_epoch:
    :param filter_preserve_ratio:
    :param max_filters_pruned_for_one_time:
    :param learning_rate_decay:
    :param learning_rate_decay_factor:
    :param weight_decay:
    :param learning_rate_decay_epoch:
    :param max_training_round:if the net can't reach target accuracy in max_training_round , the program stop.
    :param top_acc:
    :param kwargs:
    :return:
    '''

    # save the output to log
    print('save log in:' + os.path.join(conf.root_path,'model_saved' , exp_name , 'log.txt'))
    if not os.path.exists(os.path.join(conf.root_path,'model_saved' , exp_name)):
        os.makedirs(os.path.join(conf.root_path,'model_saved' , exp_name), exist_ok=True)
    sys.stdout = logger.Logger(os.path.join(conf.root_path,'model_saved' , exp_name , 'log.txt'), sys.stdout)
    sys.stderr = logger.Logger(os.path.join(conf.root_path,'model_saved' , exp_name , 'log.txt'), sys.stderr)  # redirect std err, if necessary

    print('net:',net)
    print('net_name:',net_name)
    print('exp_name:',exp_name)
    print('target_accuracy:',target_accuracy)
    print('prune_rate:',prune_rate)
    print('load_regressor:',load_regressor)
    print('predictor_name:',predictor_name)
    print('round_for_train:',round_for_train)
    print('tar_acc_gradual_decent:',tar_acc_gradual_decent)
    print('flop_expected:',flop_expected)
    print('dataset_name:',dataset_name)
    print('use_random_data:',use_random_data)
    print('validation_loader:',validation_loader)
    print('batch_size:',batch_size)
    print('num_workers:',num_workers)
    print('optimizer:',optimizer)
    print('learning_rate:',learning_rate)
    print('evaluate_step:',evaluate_step)
    print('num_epoch:',num_epoch)
    print('filter_preserve_ratio:',filter_preserve_ratio)
    print('max_filters_pruned_for_one_time:',max_filters_pruned_for_one_time)
    print('learning_rate_decay:',learning_rate_decay)
    print('learning_rate_decay_factor:',learning_rate_decay_factor)
    print('weight_decay:',weight_decay)
    print('learning_rate_decay_epoch:',learning_rate_decay_epoch)
    print('max_training_round:',max_training_round)
    print('top_acc:',top_acc)
    print('round:',round)

    print(kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.device_count(),' * ',end='')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)
    net.to(device)
    '''加载数据集'''
    if validation_loader is None:
        validation_loader = data_loader.create_validation_loader(batch_size=batch_size,
                                                                 num_workers=num_workers,
                                                                 dataset_name=dataset_name)

    flop_original_net = measure_flops.measure_model(net, dataset_name)
    original_accuracy = evaluate.evaluate_net(net=net,
                                              data_loader=validation_loader,
                                              save_net=False,
                                              dataset_name=dataset_name,
                                              top_acc=top_acc
                                              )
    if tar_acc_gradual_decent is True:
        flop_drop_expected = flop_original_net - flop_expected
        acc_drop_tolerance = original_accuracy - target_accuracy

    '''计算Conv的层数'''
    conv_list = []  # List，保存要剪枝的Conv层索引，下标从0开始
    i = -1  # Conv总数
    filter_num_lower_bound = []  # 最低filter数量
    filter_num = []
    for name,mod in net.named_modules():
        if isinstance(mod,torch.nn.Conv2d)  and 'downsample' not in name:               #1*1 conv in shortcut will not be counted
            i+=1
            filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
            filter_num.append(mod.out_channels)
            if 'layer'in name and 'conv3' not in name:
                conv_list.append(i)                                                         #layer that need to be pruned


    filter_layer = []
    filter=[]
    FIRE=[]
    predictor = predict_dead_filter.predictor(name=predictor_name)
    if load_regressor is True:
        regressor_exists=predictor.load(path=os.path.join(conf.root_path,'model_saved' , exp_name))
        if regressor_exists is True:
            round_for_train = -1
        else:                                                   #load data from previous rounds of pruning
            print('Can\'t find saved regressor. Load data from previous round.')
            filter_tmp, FIRE_tmp, filter_layer_tmp= predict_dead_filter.read_data(path=os.path.join(conf.root_path,'model_saved' , exp_name , 'dead_neural'),
                                                                                        balance=False,
                                                                                        regression_or_classification='regression',
                                                                                        batch_size=batch_size,
                                                                                        )
            FIRE+=FIRE_tmp
            filter+=filter_tmp
            filter_layer+=filter_layer_tmp
    # using data to prune the net for (round_for_train)rounds
    round-=1                        #懒得改了，先这样吧
    while True:
        round += 1
        print('{} start round {} of filter pruning.'.format(datetime.now(), round))

        if round == round_for_train + 1:
            # use filters from (round_for_train)rounds to train the regressor
            ##train the predictor######################################################################################################
            predictor.fit(filter=filter, filter_layer=filter_layer, filter_label=FIRE)
            predictor.save(path=os.path.join(conf.root_path,'model_saved' , exp_name))
        if round <= round_for_train:
            dead_filter_index, module_list, neural_list, FIRE_tmp = evaluate.find_useless_filters_data_version(
                net=net,
                batch_size=batch_size,
                use_random_data=use_random_data,
                percent_of_inactive_filter=prune_rate,
                dead_or_inactive='inactive',
                dataset_name=dataset_name,
                max_data_to_test=1000
                )
            if not os.path.exists(os.path.join(conf.root_path,'model_saved',exp_name,'dead_neural')):
                os.makedirs(os.path.join(conf.root_path,'model_saved',exp_name,'dead_neural'), exist_ok=True)

            checkpoint={'prune_rate': prune_rate,'module_list': module_list,
                        'neural_list': neural_list, 'state_dict': net.state_dict(),
                        'batch_size': batch_size}
            checkpoint.update(storage.get_net_information(net,dataset_name,net_name))
            torch.save(checkpoint,
                       os.path.join(conf.root_path,'model_saved',exp_name,'dead_neural/round %d.tar' % round)
                        )


            FIRE += FIRE_tmp
            # save filters for training the regressor
            i = 0
            for name,mod in net.named_modules():
                if isinstance(mod, torch.nn.Conv2d) and 'dowmsample' not in name:
                    conv_weight = mod.weight.cpu().detach().numpy()
                    for weight in conv_weight:
                        filter.append(weight)
                    filter_layer += [i for j in range(conv_weight.shape[0])]
                    i += 1
        else:
            dead_filter_index = evaluate.find_useless_filters_regressor_version(net=net,
                                                                                predictor=predictor,
                                                                                percent_of_inactive_filter=prune_rate,
                                                                                max_filters_pruned_for_one_time=max_filters_pruned_for_one_time
                                                                                )

        net_compressed = False
        '''卷积核剪枝'''
        for i in conv_list:

            # ensure the number of filters pruned will not be too large for one time
            if type(max_filters_pruned_for_one_time) is list:
                num_filters_to_prune_max = filter_num[i] * max_filters_pruned_for_one_time[i]
            else:
                num_filters_to_prune_max = filter_num[i] * max_filters_pruned_for_one_time
            if num_filters_to_prune_max < len(dead_filter_index[i]):
                dead_filter_index[i] = dead_filter_index[i][
                                       :int(num_filters_to_prune_max)]
            # ensure the lower bound of filter number
            if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
                dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]

            filter_num[i] = filter_num[i] - len(dead_filter_index[i])
            if len(dead_filter_index[i]) > 0:
                net_compressed = True
            print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i], len(dead_filter_index[i])))
            net = prune_module.prune_conv_layer_resnet(net=net,
                                                       layer_index=i ,
                                                       filter_index=dead_filter_index[i],
                                                       )

        if net_compressed is False:
            round -= 1
            print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
            continue

        flop_pruned_net = measure_flops.measure_model(net, dataset_name)

        if tar_acc_gradual_decent is True:  # decent the target_accuracy
            flop_reduced = flop_original_net - flop_pruned_net
            target_accuracy = original_accuracy - acc_drop_tolerance * (flop_reduced / flop_drop_expected)
            print('{} current target accuracy:{}'.format(datetime.now(), target_accuracy))

        success = False
        while not success:
            old_net = copy.deepcopy(net)
            success = train.train(net=net,
                                  net_name=net_name,
                                  exp_name=exp_name,
                                  num_epochs=num_epoch,
                                  target_accuracy=target_accuracy,
                                  learning_rate=learning_rate,
                                  load_net=False,
                                  evaluate_step=evaluate_step,
                                  dataset_name=dataset_name,
                                  optimizer=optimizer,
                                  batch_size=batch_size,
                                  learning_rate_decay=learning_rate_decay,
                                  learning_rate_decay_factor=learning_rate_decay_factor,
                                  weight_decay=weight_decay,
                                  learning_rate_decay_epoch=learning_rate_decay_epoch,
                                  test_net=True,
                                  top_acc=top_acc
                                  **kwargs
                                  )
            if not success:
                net = old_net
                max_training_round-=1
                if max_training_round==0:
                    print('{} net can\'t reach target accuracy, pruning stop.'.format(datetime.now()))
                    return

def prune_inactive_neural_with_extractor(net,
                                         net_name,
                                         exp_name,
                                         target_accuracy,
                                         prune_rate,
                                         round_for_train=2,
                                         round_to_train_freq=5,
                                         tar_acc_gradual_decent=False,
                                         flop_expected=None,
                                         dataset_name='imagenet',
                                         validation_loader=None,
                                         batch_size=conf.batch_size,
                                         num_workers=conf.num_workers,
                                         optimizer=optim.Adam,
                                         learning_rate=0.01,
                                         learning_rate_decay=False,
                                         learning_rate_decay_factor=conf.learning_rate_decay_factor,
                                         weight_decay=conf.weight_decay,
                                         learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
                                         evaluate_step=1200,
                                         num_epoch=450,
                                         filter_preserve_ratio=0.3,
                                         max_filters_pruned_for_one_time=0.5,
                                         max_training_round=9999,
                                         round=1,
                                         top_acc=1,
                                         max_data_to_test=10000,
                                         extractor_epoch=300,
                                         extractor_feature_len=27,
                                         gcn_rounds=2,
                                         only_gcn=False,
                                         no_grad=[],
                                         scheduler_name='MultiStepLR',
                                 ):
    '''

    :param net:
    :param net_name:
    :param exp_name:
    :param target_accuracy:
    :param prune_rate:
    :param load_regressor:
    :param predictor_name:
    :param round_for_train:
    :param tar_acc_gradual_decent:
    :param flop_expected:
    :param dataset_name:
    :param use_random_data:
    :param validation_loader:
    :param batch_size:
    :param num_workers:
    :param optimizer:
    :param learning_rate:
    :param evaluate_step:
    :param num_epoch:
    :param filter_preserve_ratio:
    :param max_filters_pruned_for_one_time:
    :param learning_rate_decay:
    :param learning_rate_decay_factor:
    :param weight_decay:
    :param learning_rate_decay_epoch:
    :param max_training_round:if the net can't reach target accuracy in max_training_round , the program stop.
    :param top_acc:
    :param only_gcn: only use gcn for prediction
    :param scheduler_name:
    :return:
    '''
    checkpoint_path=os.path.join(conf.root_path, 'model_saved', exp_name)
    # save the output to log
    print('save log in:' + os.path.join(checkpoint_path , 'log.txt'))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
    sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary

    print('net:',net)
    print('net_name:',net_name)
    print('exp_name:',exp_name)
    print('target_accuracy:',target_accuracy)
    print('prune_rate:',prune_rate)
    print('round_for_train:',round_for_train)
    print('roung_to_train_freq:',round_to_train_freq)
    print('tar_acc_gradual_decent:',tar_acc_gradual_decent)
    print('flop_expected:',flop_expected)
    print('dataset_name:',dataset_name)
    print('validation_loader:',validation_loader)
    print('batch_size:',batch_size)
    print('num_workers:',num_workers)
    print('optimizer:',optimizer)
    print('learning_rate:',learning_rate)
    print('evaluate_step:',evaluate_step)
    print('num_epoch:',num_epoch)
    print('filter_preserve_ratio:',filter_preserve_ratio)
    print('max_filters_pruned_for_one_time:',max_filters_pruned_for_one_time)
    print('learning_rate_decay:',learning_rate_decay)
    print('learning_rate_decay_factor:',learning_rate_decay_factor)
    print('weight_decay:',weight_decay)
    print('learning_rate_decay_epoch:',learning_rate_decay_epoch)
    print('max_training_round:',max_training_round)
    print('top_acc:',top_acc)
    print('round:',round)
    print('max_data_to_test',max_data_to_test )
    print('extractor_epoch',extractor_epoch )
    print('extractor_feature_len', extractor_feature_len)
    print('gcn_rounds',gcn_rounds )
    print('only_gcn:',only_gcn)
    print('scheduler_name:',scheduler_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.device_count(),' * ',end='')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)
    net.to(device)
    '''加载数据集'''
    if validation_loader is None:
        validation_loader = data_loader.create_validation_loader(batch_size=batch_size,
                                                                 num_workers=num_workers,
                                                                 dataset_name=dataset_name)


    if 'vgg' in net_name:
        if 'imagenet'==dataset_name:
            original_net=getattr(globals()['vgg'], net_name)(pretrained=True).to(device)
        elif 'cifar10'==dataset_name:
            original_net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/vgg16_bn_cifar10,accuracy=0.941.tar')))
        elif 'tiny_imagenet'==dataset_name:
            original_net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path, 'baseline/vgg16bn_tinyimagenet_0.73150.tar')))
        elif 'cifar100'==dataset_name:
            original_net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path, 'baseline/vgg16bn_cifar100_0.73020.tar')))
        else:
            raise Exception('Please input right dataset_name.')
    elif 'resnet' in net_name:
        if 'imagenet' == dataset_name:
            original_net = getattr(globals()['resnet'], net_name)(pretrained=True).to(device)
        elif 'tiny_imagenet'==dataset_name:
            original_net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet18_tinyimagenet_v2_0.72990.tar')))
        elif 'cifar10' == dataset_name:
            original_net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet56_cifar10,accuracy=0.94230.tar')))
        elif 'cifar100' == dataset_name:
            # original_net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path, 'baseline/resnet56_cifar100_0.70370.tar')))
            original_net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path, 'baseline/resnet32_cifar100_0.70580.tar')))

        else:
            raise Exception('Please input right dataset_name.')
    else:
        raise Exception('Unsupported net type:'+net_name)
    if isinstance(net, torch.nn.DataParallel):
        original_net=nn.DataParallel(original_net)
    flop_original_net = measure_flops.measure_model(original_net, dataset_name,print_flop=True)
    original_accuracy = evaluate.evaluate_net(net=original_net,
                                              data_loader=validation_loader,
                                              save_net=False,
                                              dataset_name=dataset_name,
                                              top_acc=top_acc
                                              )
    if tar_acc_gradual_decent is True:
        if flop_expected<=1:
            flop_expected=int(flop_original_net*flop_expected)
        flop_drop_expected = flop_original_net - flop_expected
        acc_drop_tolerance = original_accuracy - target_accuracy

    filter_num_lower_bound = []  # 最低filter数量
    i=0
    for name, mod in original_net.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and 'downsample' not in name:  # 1*1 conv in shortcut will not be counted
            if type(filter_preserve_ratio) is list:
                ratio=filter_preserve_ratio[i]
            else:
                ratio=filter_preserve_ratio
            filter_num_lower_bound+=[max(extractor_feature_len,int(mod.out_channels * ratio))]  # 输出通道数 * filter的保存比例
            i += 1
    del original_net

    if isinstance(net, nn.DataParallel):
        net_entity = net.module
    else:
        net_entity = net

    conv_list = []  # List，保存要剪枝的Conv层索引，下标从0开始
    conv_name_list=[]
    i = 0  # Conv总数
    filter_num = []
    num_filters_to_prune_at_most=[]
    for name,mod in net.named_modules():
        if isinstance(mod,torch.nn.Conv2d)  and 'downsample' not in name:               #1*1 conv in shortcut will not be counted
            # filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
            filter_num+=[mod.out_channels]
            conv_name_list+=[name]
            num_filters_to_prune_at_most+=[0]
            # layer that need to be pruned
            if isinstance(net_entity,resnet.ResNet) and 'conv3' not in name and 'layer' in name:                        #resnet for imagenet
                if net_name=='resnet18' and 'conv2' in name:
                    #todo:为了resnet18 on tinyimagenet临时改的，暂时赶时间，之后改过来！！！
                    i+=1
                    continue
                conv_list.append(i)
            elif isinstance(net_entity,resnet_cifar.ResNet) and 'conv2' not in name and 'layer' in name:                #resnet for cifar
                conv_list.append(i)
            elif 'features' in name:                                                                                    #vgg
                conv_list.append(i)
            i += 1

    filter_layer = []
    FIRE=[]
    extractor = None

    # using data to prune the net for (round_for_train)rounds
    while True:
        print('{} start round {} of filter pruning.'.format(datetime.now(), round))
        if isinstance(net,nn.DataParallel):
            net_entity = net.module
        else:
            net_entity= net

        for i in range(len(filter_num)):
            if type(max_filters_pruned_for_one_time) is list:
                num_filters_to_prune_at_most[i] = min(int(filter_num[i] * max_filters_pruned_for_one_time[i]),filter_num[i]-filter_num_lower_bound[i])
            else:
                num_filters_to_prune_at_most[i] = min(int(filter_num[i] * max_filters_pruned_for_one_time),filter_num[i]-filter_num_lower_bound[i])

        if round%round_to_train_freq<=round_for_train:                                                                          #save data to train extractor
            dead_filter_index, module_list, neural_list, _ = evaluate.find_useless_filters_data_version(
                net=net_entity,
                batch_size=16,                                                                                                  #this function need to run on sigle gpu
                percent_of_inactive_filter=prune_rate,
                dead_or_inactive='inactive',
                dataset_name=dataset_name,
                max_data_to_test=max_data_to_test,
                num_filters_to_prune_at_most=num_filters_to_prune_at_most,
            )
            num_test_images = math.ceil(max_data_to_test / 16) * 16                            #Warning, this goes wrong if dataset_size is smaller
            if not os.path.exists(os.path.join(checkpoint_path, 'dead_neural')):
                os.makedirs(os.path.join(checkpoint_path, 'dead_neural'), exist_ok=True)

            checkpoint = {'prune_rate': prune_rate, 'module_list': module_list,
                          'neural_list': neural_list, 'state_dict': net_entity.state_dict(),
                          'num_test_images':num_test_images}
            checkpoint.update(storage.get_net_information(net_entity, dataset_name, net_name))
            torch.save(checkpoint,
                       os.path.join(checkpoint_path, 'dead_neural/round %d.tar' % round))
            if round%round_to_train_freq == round_for_train:
                print('{} train extractor at round {}.'.format(datetime.now(),round))
                extractor=None
                train_extractor(train_data_dir=os.path.join(checkpoint_path, 'dead_neural'),
                                net_name=net_name,
                                dataset_name=dataset_name,
                                num_images=num_test_images,
                                epoch=extractor_epoch,
                                feature_len=extractor_feature_len,
                                gcn_rounds=gcn_rounds,
                                checkpoint_path=os.path.join(checkpoint_path, 'extractor'),
                                only_gcn=only_gcn)


        else:
            if extractor is None:
                extractor = load_extractor(os.path.join(checkpoint_path, 'extractor', str(extractor_epoch)+'.tar'))
            extractor.eval()
            FIRE=extractor.forward(net_entity,net_name,dataset_name).view(-1).tolist()
            dead_filter_index=evaluate.sort_inactive_filters(net_entity,prune_rate,FIRE,num_filters_to_prune_at_most)

        net_compressed = False
        '''卷积核剪枝'''
        for i in conv_list:
            filter_num[i] = filter_num[i] - len(dead_filter_index[i])
            if len(dead_filter_index[i]) > 0:
                net_compressed = True
            print('layer {}: remain {} filters, prune {} filters. ({})'.format(i, filter_num[i], len(dead_filter_index[i]),conv_name_list[i]))
            net = prune_module.prune_conv(net=net,
                                          net_name=net_name,
                                          layer_index=i ,
                                          filter_index=dead_filter_index[i])

        if net_compressed is False:
            print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
            continue
        else:
            round+=1
        flop_pruned_net = measure_flops.measure_model(net_entity, dataset_name)

        if tar_acc_gradual_decent is True:  # decent the target_accuracy
            flop_reduced = flop_original_net - flop_pruned_net
            target_accuracy = original_accuracy - acc_drop_tolerance * (flop_reduced / flop_drop_expected)
            print('{} current target accuracy:{}'.format(datetime.now(), target_accuracy))

        success = False
        training_round=0
        while not success:
            old_net = copy.deepcopy(net)
            success = train.train(net=net,
                                  net_name=net_name,
                                  exp_name=exp_name,
                                  num_epochs=num_epoch,
                                  target_accuracy=target_accuracy,
                                  learning_rate=learning_rate,
                                  load_net=False,
                                  evaluate_step=evaluate_step,
                                  dataset_name=dataset_name,
                                  optimizer=optimizer,
                                  batch_size=batch_size,
                                  learning_rate_decay=learning_rate_decay,
                                  learning_rate_decay_factor=learning_rate_decay_factor,
                                  weight_decay=weight_decay,
                                  learning_rate_decay_epoch=learning_rate_decay_epoch,
                                  test_net=True,
                                  top_acc=top_acc,
                                  no_grad=no_grad,
                                  scheduler_name=scheduler_name
                                  )
            training_round += 1

            if not success:
                net = old_net
                if max_training_round == training_round:
                    print('{} net can\'t reach target accuracy. Continue to prune anyway.'.format(datetime.now()))

                    file_path, file_name = find_best_net(flop_pruned_net, os.path.join(checkpoint_path, 'checkpoint'))
                    print('{} load net:{}'.format(datetime.now(), file_name))
                    data_parallel=isinstance(net,nn.DataParallel)
                    net = restore_net(torch.load(file_path), True)
                    if data_parallel is True:
                        net = nn.DataParallel(net)
                    break







def find_best_net(flop,dir):
    file_list=os.listdir(dir)
    flop = str(flop)
    target_net_list=[]
    for file in file_list:
        if flop in file:
            target_net_list+=[file]
    target_net_list.sort()                                                                  #nets are sorted by their accuracy in ascending order
    return os.path.join(dir,target_net_list[-1]),target_net_list[-1]

# def prune_inactive_neural_with_extractor_consistent_training(net,
#                                          net_name,
#                                          exp_name,
#                                          prune_rate,
#                                          round_for_train=2,
#                                          round_to_train_freq=5,
#                                          flop_expected=None,
#                                          dataset_name='imagenet',
#                                          validation_loader=None,
#                                          batch_size=conf.batch_size,
#                                          num_workers=conf.num_workers,
#                                          optimizer=optim.Adam,
#                                          learning_rate=0.01,
#                                          learning_rate_decay=False,
#                                          learning_rate_decay_factor=conf.learning_rate_decay_factor,
#                                          weight_decay=conf.weight_decay,
#                                          learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
#                                          evaluate_step=1200,
#                                          num_epoch=450,
#                                          filter_preserve_ratio=0.3,
#                                          max_filters_pruned_for_one_time=0.5,
#                                          max_training_round=9999,
#                                          round=1,
#                                          top_acc=1,
#                                          max_data_to_test=10000,
#                                          extractor_epoch=300,
#                                          extractor_feature_len=27,
#                                          gcn_rounds=2,
#                                  ):
#     '''
#
#     :param net:
#     :param net_name:
#     :param exp_name:
#     :param prune_rate:
#     :param load_regressor:
#     :param predictor_name:
#     :param round_for_train:
#     :param flop_expected:
#     :param dataset_name:
#     :param use_random_data:
#     :param validation_loader:
#     :param batch_size:
#     :param num_workers:
#     :param optimizer:
#     :param learning_rate:
#     :param evaluate_step:
#     :param num_epoch:
#     :param filter_preserve_ratio:
#     :param max_filters_pruned_for_one_time:
#     :param learning_rate_decay:
#     :param learning_rate_decay_factor:
#     :param weight_decay:
#     :param learning_rate_decay_epoch:
#     :param max_training_round:if the net can't reach target accuracy in max_training_round , the program stop.
#     :param top_acc:
#     :param kwargs:
#     :return:
#     '''
#     checkpoint_path=os.path.join(conf.root_path, 'model_saved', exp_name)
#     # save the output to log
#     print('save log in:' + os.path.join(checkpoint_path , 'log.txt'))
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path, exist_ok=True)
#     sys.stdout = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stdout)
#     sys.stderr = logger.Logger(os.path.join(checkpoint_path, 'log.txt'), sys.stderr)  # redirect std err, if necessary
#
#     print('net:',net)
#     print('net_name:',net_name)
#     print('exp_name:',exp_name)
#     print('prune_rate:',prune_rate)
#     print('round_for_train:',round_for_train)
#     print('roung_to_train_freq:',round_to_train_freq)
#     print('flop_expected:',flop_expected)
#     print('dataset_name:',dataset_name)
#     print('validation_loader:',validation_loader)
#     print('batch_size:',batch_size)
#     print('num_workers:',num_workers)
#     print('optimizer:',optimizer)
#     print('learning_rate:',learning_rate)
#     print('evaluate_step:',evaluate_step)
#     print('num_epoch:',num_epoch)
#     print('filter_preserve_ratio:',filter_preserve_ratio)
#     print('max_filters_pruned_for_one_time:',max_filters_pruned_for_one_time)
#     print('learning_rate_decay:',learning_rate_decay)
#     print('learning_rate_decay_factor:',learning_rate_decay_factor)
#     print('weight_decay:',weight_decay)
#     print('learning_rate_decay_epoch:',learning_rate_decay_epoch)
#     print('max_training_round:',max_training_round)
#     print('top_acc:',top_acc)
#     print('round:',round)
#     print('max_data_to_test',max_data_to_test )
#     print('extractor_epoch',extractor_epoch )
#     print('extractor_feature_len', extractor_feature_len)
#     print('gcn_rounds',gcn_rounds )
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using: ', end='')
#     if torch.cuda.is_available():
#         print(torch.cuda.device_count(),' * ',end='')
#         print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print(device)
#     net.to(device)
#     data_parallel=isinstance(net,nn.DataParallel)
#     '''加载数据集'''
#     if validation_loader is None:
#         validation_loader = data_loader.create_validation_loader(batch_size=batch_size,
#                                                                  num_workers=num_workers,
#                                                                  dataset_name=dataset_name)
#
#
#     if 'vgg' in net_name:
#         original_net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/vgg16_bn_cifar10,accuracy=0.941.tar')))
#     elif 'resnet' in net_name:
#         if 'imagenet' == dataset_name:
#             original_net = getattr(globals()['resnet'], net_name)(pretrained=True).to(device)
#         elif 'tiny_imagenet'==dataset_name:
#             original_net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet18_tinyimagenet_v2_0.72990.tar')))
#         elif 'cifar10' == dataset_name:
#             original_net = storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/resnet56_cifar10,accuracy=0.94230.tar')))
#         else:
#             raise Exception('Please input right dataset_name.')
#     else:
#         raise Exception('Unsupported net type:'+net_name)
#
#
#     filter_num_lower_bound = []  # 最低filter数量
#     i=0
#     for name, mod in original_net.named_modules():
#         if isinstance(mod, torch.nn.Conv2d) and 'downsample' not in name:  # 1*1 conv in shortcut will not be counted
#             if type(filter_preserve_ratio) is list:
#                 ratio=filter_preserve_ratio[i]
#             else:
#                 ratio=filter_preserve_ratio
#             filter_num_lower_bound+=[max(extractor_feature_len,int(mod.out_channels * ratio))]  # 输出通道数 * filter的保存比例
#             i+=1
#     del original_net
#
#     if isinstance(net, nn.DataParallel):
#         net_entity = net.module
#     else:
#         net_entity = net
#
#     conv_list = []  # List，保存要剪枝的Conv层索引，下标从0开始
#     i = 0  # Conv总数
#     filter_num = []
#     num_filters_to_prune_at_most=[]
#     for name,mod in net.named_modules():
#         if isinstance(mod,torch.nn.Conv2d)  and 'downsample' not in name:               #1*1 conv in shortcut will not be counted
#             # filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
#             filter_num.append(mod.out_channels)
#             num_filters_to_prune_at_most+=[0]
#             # layer that need to be pruned
#             if isinstance(net_entity,resnet.ResNet) and 'conv3' not in name and 'layer' in name:                        #resnet for imagenet
#                 conv_list.append(i)
#             elif isinstance(net_entity,resnet_cifar.ResNet) and 'conv2' not in name and 'layer' in name:                #resnet for cifar
#                 conv_list.append(i)
#             elif 'features' in name:                                                                                    #vgg
#                 conv_list.append(i)
#             i += 1
#
#
#     filter_layer = []
#     FIRE=[]
#     extractor = None
#
#     # using data to prune the net for (round_for_train)rounds
#     while True:
#         print('{} start round {} of filter pruning.'.format(datetime.now(), round))
#         if data_parallel is True:
#             net_entity = net.module
#         else:
#             net_entity= net
#
#         for i in range(len(filter_num)):
#             if type(max_filters_pruned_for_one_time) is list:
#                 num_filters_to_prune_at_most[i] = min(int(filter_num[i] * max_filters_pruned_for_one_time[i]),filter_num[i]-filter_num_lower_bound[i])
#             else:
#                 num_filters_to_prune_at_most[i] = min(int(filter_num[i] * max_filters_pruned_for_one_time),filter_num[i]-filter_num_lower_bound[i])
#
#         if round%round_to_train_freq<=round_for_train:                                                                          #save data to train extractor
#             dead_filter_index, module_list, neural_list, _ = evaluate.find_useless_filters_data_version(
#                 net=net_entity,
#                 batch_size=16,                                                                                                  #this function need to run on sigle gpu
#                 percent_of_inactive_filter=prune_rate,
#                 dead_or_inactive='inactive',
#                 dataset_name=dataset_name,
#                 max_data_to_test=max_data_to_test,
#                 num_filters_to_prune_at_most=num_filters_to_prune_at_most,
#             )
#             num_test_images = math.ceil(max_data_to_test / 16) * 16                            #Warning, this goes wrong if dataset_size is smaller
#             if not os.path.exists(os.path.join(checkpoint_path, 'dead_neural')):
#                 os.makedirs(os.path.join(checkpoint_path, 'dead_neural'), exist_ok=True)
#
#             checkpoint = {'prune_rate': prune_rate, 'module_list': module_list,
#                           'neural_list': neural_list, 'state_dict': net_entity.state_dict(),
#                           'num_test_images':num_test_images}
#             checkpoint.update(storage.get_net_information(net_entity, dataset_name, net_name))
#             torch.save(checkpoint,
#                        os.path.join(checkpoint_path, 'dead_neural/round %d.tar' % round))
#             if round%round_to_train_freq == round_for_train:
#                 print('{} train extractor at round {}.'.format(datetime.now(),round))
#                 extractor=None
#                 train_extractor(train_data_dir=os.path.join(checkpoint_path, 'dead_neural'),
#                                 net_name=net_name,
#                                 dataset_name=dataset_name,
#                                 num_images=num_test_images,
#                                 epoch=extractor_epoch,
#                                 feature_len=extractor_feature_len,
#                                 gcn_rounds=gcn_rounds,
#                                 checkpoint_path=os.path.join(checkpoint_path, 'extractor'))
#
#
#         else:
#             if extractor is None:
#                 extractor = load_extractor(os.path.join(checkpoint_path, 'extractor', str(extractor_epoch)+'.tar'))
#             extractor.eval()
#             FIRE=extractor.forward(net_entity,net_name,dataset_name).view(-1).tolist()
#             dead_filter_index=evaluate.sort_inactive_filters(net_entity,prune_rate,FIRE,num_filters_to_prune_at_most)
#
#         net_compressed = False
#         '''卷积核剪枝'''
#         for i in conv_list:
#             filter_num[i] = filter_num[i] - len(dead_filter_index[i])
#             if len(dead_filter_index[i]) > 0:
#                 net_compressed = True
#             print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i], len(dead_filter_index[i])))
#             net = prune_module.prune_conv(net=net,
#                                           net_name=net_name,
#                                           layer_index=i ,
#                                           filter_index=dead_filter_index[i])
#
#         if net_compressed is False:
#             print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
#             continue
#         else:
#             round+=1
#         flop_pruned_net = measure_flops.measure_model(net_entity, dataset_name)
#
#         training_round=0
#
#         success = train.train(net=net,
#                               net_name=net_name,
#                               exp_name=exp_name,
#                               num_epochs=num_epoch,
#                               learning_rate=learning_rate,
#                               load_net=False,
#                               evaluate_step=evaluate_step,
#                               dataset_name=dataset_name,
#                               optimizer=optimizer,
#                               batch_size=batch_size,
#                               learning_rate_decay=learning_rate_decay,
#                               learning_rate_decay_factor=learning_rate_decay_factor,
#                               weight_decay=weight_decay,
#                               learning_rate_decay_epoch=learning_rate_decay_epoch,
#                               test_net=True,
#                               top_acc=top_acc
#                               )
#         file_path,file_name=find_best_net(flop_pruned_net,os.path.join(checkpoint_path,'checkpoint'))
#         print('{} load net:{}'.format(datetime.now(),file_name))
#         net=restore_net(torch.load(file_path),True)
#         if data_parallel is True:
#             net=nn.DataParallel(net)
#         training_round += 1


# #old version, for resnet on cifar10/100 and tiny imagenet
# def prune_inactive_neural_with_regressor_resnet(net,
#                                                 net_name,
#                                                 exp_name,
#                                                 target_accuracy,
#                                                 prune_rate,
#                                                 load_regressor=False,
#                                                 predictor_name='random_forest',
#                                                 round_for_train=2,
#                                                 tar_acc_gradual_decent=False,
#                                                 flop_expected=None,
#                                                 dataset_name='imagenet',
#                                                 use_random_data=False,
#                                                 validation_loader=None,
#                                                 batch_size=conf.batch_size,
#                                                 num_workers=conf.num_workers,
#                                                 optimizer=optim.Adam,
#                                                 learning_rate=0.01,
#                                                 evaluate_step=1000,
#                                                 num_epoch=450,
#                                                 filter_preserve_ratio=0.3,
#                                                 max_filters_pruned_for_one_time=0.5,
#                                                 learning_rate_decay=False,
#                                                 learning_rate_decay_factor=conf.learning_rate_decay_factor,
#                                                 weight_decay=conf.weight_decay,
#                                                 learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
#                                                 max_training_round=9999,
#                                                 round=1,
#                                                 **kwargs
#                                  ):
#     '''
#
#     :param net:
#     :param net_name:
#     :param exp_name:
#     :param target_accuracy:
#     :param prune_rate:
#     :param load_regressor:
#     :param predictor_name:
#     :param round_for_train:
#     :param tar_acc_gradual_decent:
#     :param flop_expected:
#     :param dataset_name:
#     :param use_random_data:
#     :param validation_loader:
#     :param batch_size:
#     :param num_workers:
#     :param optimizer:
#     :param learning_rate:
#     :param evaluate_step:
#     :param num_epoch:
#     :param filter_preserve_ratio:
#     :param max_filters_pruned_for_one_time:
#     :param learning_rate_decay:
#     :param learning_rate_decay_factor:
#     :param weight_decay:
#     :param learning_rate_decay_epoch:
#     :param max_training_round:if the net can't reach target accuracy in max_training_round , the program stop.
#     :param kwargs:
#     :return:
#     '''
#
#     # save the output to log
#     print('save log in:' + conf.root_path + exp_name + '/log.txt')
#     if not os.path.exists(conf.root_path + exp_name):
#         os.makedirs(conf.root_path + exp_name, exist_ok=True)
#     sys.stdout = logger.Logger(conf.root_path + exp_name + '/log.txt', sys.stdout)
#     sys.stderr = logger.Logger(conf.root_path + exp_name + '/log.txt', sys.stderr)  # redirect std err, if necessary
#
#     print('net:',net)
#     print('net_name:',net_name)
#     print('exp_name:',exp_name)
#     print('target_accuracy:',target_accuracy)
#     print('prune_rate:',prune_rate)
#     print('load_regressor:',load_regressor)
#     print('predictor_name:',predictor_name)
#     print('round_for_train:',round_for_train)
#     print('tar_acc_gradual_decent:',tar_acc_gradual_decent)
#     print('flop_expected:',flop_expected)
#     print('dataset_name:',dataset_name)
#     print('use_random_data:',use_random_data)
#     print('validation_loader:',validation_loader)
#     print('batch_size:',batch_size)
#     print('num_workers:',num_workers)
#     print('optimizer:',optimizer)
#     print('learning_rate:',learning_rate)
#     print('evaluate_step:',evaluate_step)
#     print('num_epoch:',num_epoch)
#     print('filter_preserve_ratio:',filter_preserve_ratio)
#     print('max_filters_pruned_for_one_time:',max_filters_pruned_for_one_time)
#     print('learning_rate_decay:',learning_rate_decay)
#     print('learning_rate_decay_factor:',learning_rate_decay_factor)
#     print('weight_decay:',weight_decay)
#     print('learning_rate_decay_epoch:',learning_rate_decay_epoch)
#     print('max_training_round:',max_training_round)
#     print('round:',round)
#
#     print(kwargs)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using: ', end='')
#     if torch.cuda.is_available():
#         print(torch.cuda.device_count(),' * ',end='')
#         print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print(device)
#     net.to(device)
#     '''加载数据集'''
#     if validation_loader is None:
#         validation_loader = data_loader.create_validation_loader(batch_size=batch_size,
#                                                                  num_workers=num_workers,
#                                                                  dataset_name=dataset_name)
#
#     flop_original_net = measure_flops.measure_model(net, dataset_name)
#     original_accuracy = evaluate.evaluate_net(net=net,
#                                               data_loader=validation_loader,
#                                               save_net=False,
#                                               dataset_name=dataset_name,
#                                               )
#     if tar_acc_gradual_decent is True:
#         flop_drop_expected = flop_original_net - flop_expected
#         acc_drop_tolerance = original_accuracy - target_accuracy
#
#     '''计算Conv的层数'''
#     conv_list = []  # List，保存要剪枝的Conv层索引，下标从0开始
#     i = 0  # Conv总数
#     index_in_block = -1
#     filter_num_lower_bound = []  # 最低filter数量
#     filter_num = []
#     for mod in net.modules():
#         if isinstance(mod, resnet_cifar.BasicBlock) or isinstance(mod, resnet_tinyimagenet.BasicBlock):
#             index_in_block = 1
#         elif isinstance(mod, torch.nn.modules.conv.Conv2d):
#             if index_in_block == 1:  # 在block里面
#                 index_in_block = 2
#                 conv_list.append(i)
#                 filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
#                 filter_num.append(mod.out_channels)
#             elif index_in_block == 2:  # 不需要剪枝的Conv层
#                 index_in_block = -1
#                 filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
#                 filter_num.append(mod.out_channels)
#             elif index_in_block == -1:  # 不在block里面
#                 # conv_list.append(i)
#                 filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
#                 filter_num.append(mod.out_channels)
#             i += 1
#
#     modules_list=prune.create_module_name_list(net)  # 创建一个list保存每一个module的名字
#
#     filter_layer = []
#     filter=[]
#     FIRE=[]
#     predictor = predict_dead_filter.predictor(name=predictor_name)
#     if load_regressor is True:
#         regressor_exists=predictor.load(path=conf.root_path + exp_name)
#         if regressor_exists is True:
#             round_for_train = -1
#         else:                                                   #load data from previous rounds of pruning
#             print('Can\'t find saved regressor. Load data from previous round.')
#             filter_tmp, FIRE_tmp, filter_layer_tmp= predict_dead_filter.read_data(path=conf.root_path + exp_name + '/dead_neural/',
#                                                                                         balance=False,
#                                                                                         regression_or_classification='regression',
#                                                                                         batch_size=batch_size,
#                                                                                         )
#             FIRE+=FIRE_tmp
#             filter+=filter_tmp
#             filter_layer+=filter_layer_tmp
#     # using data to prune the net for (round_for_train)rounds
#     round-=1                        #懒得改了，先这样吧
#     while True:
#         round += 1
#         print('{} start round {} of filter pruning.'.format(datetime.now(), round))
#
#         if round == round_for_train + 1:
#             # use filters from (round_for_train)rounds to train the regressor
#
#             ##train the predictor######################################################################################################
#             predictor.fit(filter=filter, filter_layer=filter_layer, filter_label=FIRE)
#             predictor.save(path=conf.root_path + exp_name)
#         if round <= round_for_train:
#
#             dead_filter_index, module_list, neural_list, FIRE_tmp = evaluate.find_useless_filters_data_version(
#                 net=net,
#                 batch_size=batch_size,
#                 use_random_data=use_random_data,
#                 percent_of_inactive_filter=prune_rate,
#                 dead_or_inactive='inactive',
#                 dataset_name=dataset_name
#                 )
#             if not os.path.exists(conf.root_path + exp_name + '/dead_neural'):
#                 os.makedirs(conf.root_path + exp_name + '/dead_neural', exist_ok=True)
#
#             checkpoint={'prune_rate': prune_rate,'module_list': module_list,
#                         'neural_list': neural_list, 'state_dict': net.state_dict(),
#                         'batch_size': batch_size}
#             checkpoint.update(storage.get_net_information(net,dataset_name,net_name))
#             torch.save(checkpoint,
#                        conf.root_path + exp_name + '/dead_neural/round %d.tar' % round, )
#
#
#             FIRE += FIRE_tmp
#             # save filters for training the regressor
#             i = 0
#             for mod in net.modules():
#                 if isinstance(mod, torch.nn.modules.conv.Conv2d):
#                     conv_weight = mod.weight.cpu().detach().numpy()
#                     for weight in conv_weight:
#                         filter.append(weight)
#                     filter_layer += [i for j in range(conv_weight.shape[0])]
#                     i += 1
#         else:
#             dead_filter_index = evaluate.find_useless_filters_regressor_version(net=net,
#                                                                                 predictor=predictor,
#                                                                                 percent_of_inactive_filter=prune_rate,
#                                                                                 max_filters_pruned_for_one_time=max_filters_pruned_for_one_time
#                                                                                 )
#
#         net_compressed = False
#         '''卷积核剪枝'''
#         for i in conv_list:
#
#             # ensure the number of filters pruned will not be too large for one time
#             if type(max_filters_pruned_for_one_time) is list:
#                 num_filters_to_prune_max = filter_num[i] * max_filters_pruned_for_one_time[i]
#             else:
#                 num_filters_to_prune_max = filter_num[i] * max_filters_pruned_for_one_time
#             if num_filters_to_prune_max < len(dead_filter_index[i]):
#                 dead_filter_index[i] = dead_filter_index[i][
#                                        :int(num_filters_to_prune_max)]
#             # ensure the lower bound of filter number
#             if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
#                 dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]
#
#             filter_num[i] = filter_num[i] - len(dead_filter_index[i])
#             if len(dead_filter_index[i]) > 0:
#                 net_compressed = True
#             print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i], len(dead_filter_index[i])))
#             net = prune.prune_conv_layer_resnet(net=net,
#                                                 layer_index=i + 1,
#                                                 filter_index=dead_filter_index[i],
#                                                 modules_list=modules_list)
#
#         if net_compressed is False:
#             round -= 1
#             print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
#             continue
#
#         flop_pruned_net = measure_flops.measure_model(net, dataset_name)
#
#         if tar_acc_gradual_decent is True:  # decent the target_accuracy
#             flop_reduced = flop_original_net - flop_pruned_net
#             target_accuracy = original_accuracy - acc_drop_tolerance * (flop_reduced / flop_drop_expected)
#             print('{} current target accuracy:{}'.format(datetime.now(), target_accuracy))
#
#         success = False
#         while not success:
#             old_net = copy.deepcopy(net)
#             success = train.train(net=net,
#                                   net_name=net_name,
#                                   exp_name=exp_name,
#                                   num_epochs=num_epoch,
#                                   target_accuracy=target_accuracy,
#                                   learning_rate=learning_rate,
#                                   load_net=False,
#                                   evaluate_step=evaluate_step,
#                                   dataset_name=dataset_name,
#                                   optimizer=optimizer,
#                                   batch_size=batch_size,
#                                   learning_rate_decay=learning_rate_decay,
#                                   learning_rate_decay_factor=learning_rate_decay_factor,
#                                   weight_decay=weight_decay,
#                                   learning_rate_decay_epoch=learning_rate_decay_epoch,
#                                   test_net=True,
#                                   **kwargs
#                                   )
#             if not success:
#                 net = old_net
#                 max_training_round-=1
#                 if max_training_round==0:
#                     print('{} net can\'t reach target accuracy, pruning stop.'.format(datetime.now()))
#                     return


                
                
                
                

# def prune_dead_neural_with_classifier(net,
#                                      net_name,
#                                      neural_dead_times,
#                                      filter_FIRE,
#                                      target_accuracy,
#                                      predictor_name='logistic_regression',
#                                      round_for_train=2,
#                                      tar_acc_gradual_decent=False,
#                                      flop_expected=None,
#                                      filter_FIRE_decay=0.95,
#                                      neural_dead_times_decay=0.95,
#                                      dataset_name='imagenet',
#                                      use_random_data=False,
#                                      validation_loader=None,
#                                      batch_size=conf.batch_size,
#                                      num_workers=conf.num_workers,
#                                      optimizer=optim.Adam,
#                                      learning_rate=0.01,
#                                      evaluate_step=1000,
#                                      num_epoch=350,
#                                      filter_preserve_ratio=0.3,
#                                      min_filters_pruned_for_one_time=0.05,
#                                      max_filters_pruned_for_one_time=0.5,
#                                      learning_rate_decay=False,
#                                      learning_rate_decay_factor=conf.learning_rate_decay_factor,
#                                      weight_decay=conf.weight_decay,
#                                      learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
#                                      **kwargs
#                                      ):
#     '''
# 
#     :param net:
#     :param net_name:
#     :param neural_dead_times:
#     :param filter_FIRE:
#     :param target_accuracy:
#     :param predictor_name: name of the predictor used
#     :param round_for_train:
#     :param tar_acc_gradual_decent:
#     :param flop_expected:
#     :param filter_FIRE_decay:
#     :param neural_dead_times_decay:
#     :param dataset_name:
#     :param use_random_data:
#     :param validation_loader:
#     :param batch_size:
#     :param num_workers:
#     :param optimizer:
#     :param learning_rate:
#     :param evaluate_step:
#     :param num_epoch:
#     :param filter_preserve_ratio:
#     :param min_filters_pruned_for_one_time:
#     :param max_filters_pruned_for_one_time:
#     :param learning_rate_decay:
#     :param learning_rate_decay_factor:
#     :param weight_decay:
#     :param learning_rate_decay_epoch:
#     :return:
#     '''
#     # save the output to log
#     print('save log in:' + conf.root_path + net_name + '/log.txt')
#     if not os.path.exists(conf.root_path + net_name):
#         os.makedirs(conf.root_path + net_name, exist_ok=True)
#     sys.stdout = logger.Logger(conf.root_path + net_name + '/log.txt', sys.stdout)
#     sys.stderr = logger.Logger(conf.root_path + net_name + '/log.txt', sys.stderr)  # redirect std err, if necessary
# 
#     print('net:{}\n'
#           'net_name:{}\n'
#           'use_random_data:{}\n'
#           'neural_dead_times:{}\n'
#           'filter_FIRE:{}\n'
#           'target_accuracy:{}\n'
#           'predictor_name:{}\n'
#           'round_for_train:{}\n'
#           'tar_acc_gradual_decent:{}\n'
#           'flop_expected:{}\n'
#           'filter_FIRE_decay:{}\n'
#           'neural_dead_times_decay:{}\n'
#           'dataset_name:{}\n'
#           'validation_loader:{}\n'
#           'batch_size:{}\n'
#           'num_workers:{}\n'
#           'optimizer:{}\n'
#           'learning_rate:{}\n'
#           'evaluate_step:{}\n'
#           'num_epoch:{}\n'
#           'filter_preserve_ratio:{}\n'
#           'max_filters_pruned_for_one_time:{}\n'
#           'min_filters_pruned_for_one_time:{}\n'
#           'learning_rate_decay:{}\n'
#           'learning_rate_decay_factor:{}\n'
#           'weight_decay:{}\n'
#           'learning_rate_decay_epoch:{}'
#           .format(net, net_name, use_random_data, neural_dead_times, filter_FIRE, target_accuracy,
#                   predictor_name,round_for_train,
#                   tar_acc_gradual_decent,
#                   flop_expected, filter_FIRE_decay,
#                   neural_dead_times_decay, dataset_name, validation_loader, batch_size, num_workers, optimizer,
#                   learning_rate, evaluate_step,
#                   num_epoch, filter_preserve_ratio, max_filters_pruned_for_one_time,min_filters_pruned_for_one_time, learning_rate_decay,
#                   learning_rate_decay_factor,
#                   weight_decay, learning_rate_decay_epoch))
#     print(kwargs)
# 
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using: ', end='')
#     if torch.cuda.is_available():
#         print(torch.cuda.device_count(),' * ',end='')
#         print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print(device)
# 
#     if validation_loader is None:
#         validation_loader = data_loader.create_validation_loader(
#                                                                  batch_size=batch_size,
#                                                                  num_workers=num_workers,
#                                                                  dataset_name=dataset_name,
#                                                                  )
# 
#     flop_original_net = measure_flops.measure_model(net, dataset_name)
#     original_accuracy = evaluate.evaluate_net(net=net,
#                                               data_loader=validation_loader,
#                                               save_net=False,
#                                               dataset_name=dataset_name,
#                                               )
#     if tar_acc_gradual_decent is True:
#         flop_drop_expected = flop_original_net - flop_expected
#         acc_drop_tolerance = original_accuracy - target_accuracy
# 
#     num_conv = 0  # num of conv layers in the net
#     filter_num_lower_bound = []
#     filter_num = []
#     for mod in net.features:
#         if isinstance(mod, torch.nn.modules.conv.Conv2d):
#             num_conv += 1
#             filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))
#             filter_num.append(mod.out_channels)
# 
#     lived_filter=[]
#     dead_filter=[]
# 
#     #using data to prune the net for (round_for_train)rounds
#     round = 0
#     while True:
#         round += 1
#         print('{} start round {} of filter pruning.'.format(datetime.now(), round))
# 
#         if round==round_for_train+1:
#             # use filters from (round_for_train)rounds to train the classifier
# 
#             ##train the predictor######################################################################################################
#             predictor= predict_dead_filter.predictor(name=predictor_name)
#             predictor.fit(lived_filter=lived_filter, dead_filter=dead_filter)
# 
#         if round<=round_for_train:
#             print('{} current filter_FIRE:{},neural_dead_times:{}'.format(datetime.now(), filter_FIRE,
#                                                                                 neural_dead_times))
#             #find dead filters
#             dead_filter_index,module_list,neural_list = evaluate.find_useless_filters_data_version(net=net, filter_FIRE=filter_FIRE,
#                                                                                                    neural_dead_times=neural_dead_times, batch_size=batch_size, use_random_data=use_random_data)
#             # save dead and lived filters for training the classifier
#             i = 0
#             for mod in net.features:
#                 if isinstance(mod, torch.nn.modules.conv.Conv2d):
#                     conv_weight = copy.deepcopy(mod).weight.cpu().detach().numpy()
#                     dead_filter = dead_filter + list(conv_weight[dead_filter_index[i]])
#                     lived_filter = lived_filter + list(
#                         conv_weight[[j for j in range(conv_weight.shape[0]) if j not in dead_filter_index[i]]])
# 
#                     # ensure the number of filters pruned will not be too large for one time
#                     if filter_num[i] * max_filters_pruned_for_one_time < len(dead_filter_index[i]):
#                         dead_filter_index[i] = dead_filter_index[i][
#                                                :int(filter_num[i] * max_filters_pruned_for_one_time)]
#                     # ensure the lower bound of filter number
#                     if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
#                         dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]
#                     i += 1
#         else:
#             dead_filter_index= evaluate.predict_dead_filters_classifier_version(net=net,
#                                                                                 predictor=predictor,
#                                                                                 min_ratio_dead_filters=min_filters_pruned_for_one_time,
#                                                                                 max_ratio_dead_filters=max_filters_pruned_for_one_time,
#                                                                                 filter_num_lower_bound=filter_num_lower_bound)
# 
#         net_compressed = False
#         #prune the net according to dead_filter_index
#         for i in range(num_conv):
#             filter_num[i] = filter_num[i] - len(dead_filter_index[i])
#             if len(dead_filter_index[i]) > 0:
#                 net_compressed = True
#             print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i],
#                                                                           len(dead_filter_index[i])))
# 
#             net = prune.prune_conv_layer_vgg(model=net, layer_index=i + 1,
#                                          filter_index=dead_filter_index[i])  # prune the dead filter
# 
#         if net_compressed is False:
#             round -= 1
#             filter_FIRE *= filter_FIRE_decay
#             neural_dead_times *= neural_dead_times_decay
#             print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
#             continue
# 
#         flop_pruned_net = measure_flops.measure_model(net, dataset_name)
# 
#         if tar_acc_gradual_decent is True:  # decent the target_accuracy
#             flop_reduced = flop_original_net - flop_pruned_net
#             target_accuracy = original_accuracy - acc_drop_tolerance * (flop_reduced / flop_drop_expected)
#             print('{} current target accuracy:{}'.format(datetime.now(), target_accuracy))
# 
#         success = False
#         # while not success:
#         #     old_net = copy.deepcopy(net)
#         #     success = train.train(net=net,
#         #                           net_name=net_name,
#         #                           num_epochs=num_epoch,
#         #                           target_accuracy=target_accuracy,
#         #                           learning_rate=learning_rate,
#         #                           load_net=False,
#         #                           evaluate_step=evaluate_step,
#         #                           dataset_name=dataset_name,
#         #                           optimizer=optimizer,
#         #                           batch_size=batch_size,
#         #                           learning_rate_decay=learning_rate_decay,
#         #                           learning_rate_decay_factor=learning_rate_decay_factor,
#         #                           weight_decay=weight_decay,
#         #                           learning_rate_decay_epoch=learning_rate_decay_epoch,
#         #                           test_net=True,
#         #                           )
#         #     if not success:
#         #         net = old_net
#         # filter_FIRE *= filter_FIRE_decay
#         # neural_dead_times *= neural_dead_times_decay




# def prune_dead_neural(net,
#                       net_name,
#                       neural_dead_times,
#                       filter_FIRE,
#                       target_accuracy,
#                       tar_acc_gradual_decent=False,
#                       flop_expected=None,
#                       filter_FIRE_decay=0.95,
#                       neural_dead_times_decay=0.95,
#                       dataset_name='imagenet',
#                       use_random_data=False,
#                       validation_loader=None,
#                       batch_size=conf.batch_size,
#                       num_workers=conf.num_workers,
#                       optimizer=optim.Adam,
#                       learning_rate=0.01,
#                       evaluate_step=1000,
#                       num_epoch=350,
#                       filter_preserve_ratio=0.3,
#                       max_filters_pruned_for_one_time=0.5,
#                       learning_rate_decay=False,
#                       learning_rate_decay_factor=conf.learning_rate_decay_factor,
#                       weight_decay=conf.weight_decay,
#                       learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
#                       **kwargs
#                      ):
#     '''
# 
#     :param net:
#     :param net_name:
#     :param neural_dead_times(ndt):int, threshold for judging a dead neural node
#     :param filter_FIRE(fdr):float, threshold for judging a dead filter
#     :param target_accuracy: float,
#     :param tar_acc_gradual_decent:bool, if true, the target accuracy will decent from original acc. to target acc. during every round of pruning
#     :param flop_expected: int: expected flop after net pruned. will only work when tar_acc_gradual_decent is true
#     :param filter_FIRE_decay:float, decay rate for fdr in each round of pruning
#     :param neural_dead_times_decay:float, decay rate for ndt in each round of pruning
#     :param dataset_name:
#     :param use_random_data:bool, if true, generated data which fits normal distribution will be used to calculate dead filters.
#     :param validation_loader:
#     :param batch_size:
#     :param num_workers:
#     :param optimizer:
#     :param learning_rate:
#     :param evaluate_step:
#     :param num_epoch:
#     :param filter_preserve_ratio:
#     :param max_filters_pruned_for_one_time:
#     :param learning_rate_decay:
#     :param learning_rate_decay_factor:
#     :param weight_decay:
#     :param learning_rate_decay_epoch:
#     :return:
#     '''
#     #save the output to log
#     print('save log in:' + conf.root_path + net_name + '/log.txt')
#     if not os.path.exists(conf.root_path + net_name ):
#         os.makedirs(conf.root_path + net_name , exist_ok=True)
#     sys.stdout = logger.Logger(conf.root_path+net_name+'/log.txt', sys.stdout)
#     sys.stderr = logger.Logger(conf.root_path+net_name+'/log.txt', sys.stderr)  # redirect std err, if necessary
# 
#     print('net:{}\n'
#           'net_name:{}\n'
#           'use_random_data:{}\n'
#           'neural_dead_times:{}\n'
#           'filter_FIRE:{}\n'
#           'target_accuracy:{}\n'
#           'tar_acc_gradual_decent:{}\n'
#           'flop_expected:{}\n'
#           'filter_FIRE_decay:{}\n'
#           'neural_dead_times_decay:{}\n'
#           'dataset_name:{}\n'
#           'validation_loader:{}\n'
#           'batch_size:{}\n'
#           'num_workers:{}\n'
#           'optimizer:{}\n'
#           'learning_rate:{}\n'
#           'evaluate_step:{}\n'
#           'num_epoch:{}\n'
#           'filter_preserve_ratio:{}\n'
#           'max_filters_pruned_for_one_time:{}\n'
#           'learning_rate_decay:{}\n'
#           'learning_rate_decay_factor:{}\n'
#           'weight_decay:{}\n'
#           'learning_rate_decay_epoch:{}'
#           .format(net,net_name,use_random_data,neural_dead_times,filter_FIRE,target_accuracy,tar_acc_gradual_decent,
#                   flop_expected,filter_FIRE_decay,
#                   neural_dead_times_decay,dataset_name,validation_loader,batch_size,num_workers,optimizer,learning_rate,evaluate_step,
#                   num_epoch,filter_preserve_ratio,max_filters_pruned_for_one_time,learning_rate_decay,learning_rate_decay_factor,
#                   weight_decay,learning_rate_decay_epoch))
#     print(kwargs)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using: ', end='')
#     if torch.cuda.is_available():
#         print(torch.cuda.device_count(),' * ',end='')
#         print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print(device)
# 
#     if validation_loader is None :
#         validation_loader = data_loader.create_validation_loader(batch_size=batch_size,
#                                                                  num_workers=num_workers,
#                                                                  dataset_name=dataset_name,
#                                                                  )
# 
#     flop_original_net= measure_flops.measure_model(net, dataset_name)
#     original_accuracy= evaluate.evaluate_net(net=net,
#                                              data_loader=validation_loader,
#                                              save_net=False,
#                                              dataset_name=dataset_name,
#                                              )
#     if tar_acc_gradual_decent is True:
#         flop_drop_expected = flop_original_net - flop_expected
#         acc_drop_tolerance = original_accuracy - target_accuracy
# 
#     num_conv = 0  # num of conv layers in the net
#     filter_num_lower_bound=[]
#     filter_num=[]
#     for mod in net.features:
#         if isinstance(mod, torch.nn.modules.conv.Conv2d):
#             num_conv += 1
#             filter_num_lower_bound.append(int(mod.out_channels*filter_preserve_ratio))
#             filter_num.append(mod.out_channels)
# 
#     round=0
#     while True:
#         round+=1
#         print('{} start round {} of filter pruning.'.format(datetime.now(), round))
#         print('{} current filter_FIRE:{},neural_dead_times:{}'.format(datetime.now(), filter_FIRE,
#                                                                             neural_dead_times))
# 
#         # find dead filters
#         dead_filter_index, module_list, neural_list = evaluate.find_useless_filters_data_version(net=net,
#                                                                                                  filter_FIRE=filter_FIRE,
#                                                                                                  neural_dead_times=neural_dead_times,
#                                                                                                  batch_size=batch_size,
#                                                                                                  use_random_data=use_random_data)
# 
#         if not os.path.exists(conf.root_path + net_name + '/dead_neural'):
#             os.makedirs(conf.root_path + net_name + '/dead_neural', exist_ok=True)
# 
#         torch.save({'neural_dead_times': neural_dead_times, 'filter_FIRE': filter_FIRE,
#                     'net': net, 'module_list': module_list,
#                     'neural_list': neural_list, 'state_dict': net.state_dict(),'batch_size':batch_size},
#                    conf.root_path + net_name + '/dead_neural/round %d.tar' % round, )
# 
#         net_compressed = False
#         for i in range(num_conv):
#             # ensure the number of filters pruned will not be too large for one time
#             if filter_num[i] * max_filters_pruned_for_one_time < len(dead_filter_index[i]):
#                 dead_filter_index[i] = dead_filter_index[i][:int(filter_num[i] * max_filters_pruned_for_one_time)]
#             # ensure the lower bound of filter number
#             if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
#                 dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]
#             filter_num[i] = filter_num[i] - len(dead_filter_index[i])
#             if len(dead_filter_index[i]) > 0:
#                 net_compressed = True
# 
#             print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i],
#                                                                           len(dead_filter_index[i])))
# 
#             net= prune.prune_conv_layer_vgg(model=net, layer_index=i + 1, filter_index=dead_filter_index[i])    #prune the dead filter
# 
#         if net_compressed is False:
#             os.remove(conf.root_path+net_name+'/dead_neural/round %d.tar'%round)
#             round-=1
#             filter_FIRE *= filter_FIRE_decay
#             neural_dead_times *= neural_dead_times_decay
#             print('{} round {} did not prune any filters. Restart.'.format(datetime.now(),round+1))
#             continue
# 
#         flop_pruned_net= measure_flops.measure_model(net, dataset_name)
# 
#         #todo:现在根据剪了多少浮点量来线性降低准确率，也可考虑根据剪的轮数来降低准确率
#         #todo:也可考虑改为非线性下降，一开始下降的少点
#         if tar_acc_gradual_decent is True:                                      #decent the target_accuracy
#             flop_reduced=flop_original_net-flop_pruned_net
#             target_accuracy=original_accuracy-acc_drop_tolerance*(flop_reduced/flop_drop_expected)
#             print('{} current target accuracy:{}'.format(datetime.now(),target_accuracy))
# 
#         success=False
#         while not success:
#             old_net=copy.deepcopy(net)
#             success= train.train(net=net,
#                                  net_name=net_name,
#                                  num_epochs=num_epoch,
#                                  target_accuracy=target_accuracy,
#                                  learning_rate=learning_rate,
#                                  load_net=False,
#                                  evaluate_step=evaluate_step,
#                                  dataset_name=dataset_name,
#                                  optimizer=optimizer,
#                                  batch_size=batch_size,
#                                  learning_rate_decay=learning_rate_decay,
#                                  learning_rate_decay_factor=learning_rate_decay_factor,
#                                  weight_decay=weight_decay,
#                                  learning_rate_decay_epoch=learning_rate_decay_epoch,
#                                  test_net=True,
#                                  **kwargs
#                                  )
#             if not success:
#                 net=old_net
#         filter_FIRE*=filter_FIRE_decay
#         neural_dead_times*=neural_dead_times_decay





# def prune_layer_gradually():
#     net = train.create_net('vgg16_bn', True)
#
#     num_conv = 0  # num of conv layers in the net
#     for mod in net.features:
#         if isinstance(mod, torch.nn.modules.conv.Conv2d):
#             num_conv += 1
#
#     # for i in range(1, 7):
#     #     net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
#     #                                   ord=2)  # prune the model
#
#     file_new = '/home/victorfang/Desktop/pytorch_model/vgg16_bn,gradual_pruned/checkpoint/sample_num=64064.tar'
#     if os.path.isfile(file_new):
#         checkpoint = torch.load(file_new)
#         net = checkpoint['net']
#         net.load_state_dict(checkpoint['state_dict'])
#
#     iteration = 1
#     while (True):
#         print('{} start iteration:{}'.format(datetime.now(), iteration))
#         for i in range(10, num_conv + 1):
#             net = select_and_prune_filter(net, layer_index=i, percent_of_pruning=0.1,
#                                           ord=2)  # prune the model
#             print('{} layer {} pruned'.format(datetime.now(), i))
#
#             validation_loader = data_loader.create_validation_loader(dataset_path=conf.imagenet['validation_set_path'],
#                                                                      default_image_size=224,
#                                                                      mean=conf.imagenet['mean'],
#                                                                      std=conf.imagenet['std'],
#                                                                      batch_size=conf.batch_size,
#                                                                      num_workers=conf.num_workers,
#                                                                      dataset_name='imagenet')
#             net_name = 'vgg16_bn,gradual_pruned'
#             checkpoint_path = conf.root_path + net_name + '/checkpoint'
#             accuracy = evaluate.evaluate_net(net, validation_loader,
#                                              save_net=True,
#                                              checkpoint_path=checkpoint_path,
#                                              sample_num=0,
#                                              target_accuracy=0.7)
#             if accuracy < 0.7:
#                 train.train(net=net,
#                             net_name=net_name,
#                             num_epochs=1,
#                             target_accuracy=0.7,
#                             learning_rate=2e-5,
#                             load_net=False,
#                             evaluate_step=1000
#                             )
#         break
#         iteration += 1




# def prune_dead_neural_resnet(net,
#                  net_name,
#                  neural_dead_times,
#                  filter_FIRE,  # no use
#                  target_accuracy,
#                  predictor_name='logistic_regression',  # no use
#                  round_for_train=11,  # no use
#                  tar_acc_gradual_decent=False,
#                  flop_expected=None,
#                  filter_FIRE_decay=0.95,  # no use
#                  neural_dead_times_decay=0.95,  # no use
#                  dataset_name='imagenet',
#                  use_random_data=False,
#                  validation_loader=None,
#                  batch_size=conf.batch_size,
#                  num_workers=conf.num_workers,
#                  optimizer=optim.Adam,
#                  learning_rate=0.01,
#                  evaluate_step=1000,
#                  num_epoch=350,
#                  filter_preserve_ratio=0.3,
#                  min_filters_pruned_for_one_time=0.05,
#                  max_filters_pruned_for_one_time=0.5,
#                  learning_rate_decay=False,
#                  learning_rate_decay_factor=conf.learning_rate_decay_factor,
#                  weight_decay=conf.weight_decay,
#                  learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
#                  **kwargs):
#     """
#     对ResNet剪枝
#     """
#
#     # save the output to log
#     print('save log in:' + conf.root_path + net_name + '/log.txt')
#     if not os.path.exists(conf.root_path + net_name):
#         os.makedirs(conf.root_path + net_name, exist_ok=True)
#     sys.stdout = logger.Logger(conf.root_path + net_name + '/log.txt', sys.stdout)
#     sys.stderr = logger.Logger(conf.root_path + net_name + '/log.txt', sys.stderr)  # redirect std err, if necessary
#
#     print('net:{}\n'
#           'net_name:{}\n'
#           'use_random_data:{}\n'
#           'neural_dead_times:{}\n'
#           'filter_FIRE:{}\n'
#           'target_accuracy:{}\n'
#           'predictor_name:{}\n'
#           'round_for_train:{}\n'
#           'tar_acc_gradual_decent:{}\n'
#           'flop_expected:{}\n'
#           'filter_FIRE_decay:{}\n'
#           'neural_dead_times_decay:{}\n'
#           'dataset_name:{}\n'
#           'validation_loader:{}\n'
#           'batch_size:{}\n'
#           'num_workers:{}\n'
#           'optimizer:{}\n'
#           'learning_rate:{}\n'
#           'evaluate_step:{}\n'
#           'num_epoch:{}\n'
#           'filter_preserve_ratio:{}\n'
#           'max_filters_pruned_for_one_time:{}\n'
#           'min_filters_pruned_for_one_time:{}\n'
#           'learning_rate_decay:{}\n'
#           'learning_rate_decay_factor:{}\n'
#           'weight_decay:{}\n'
#           'learning_rate_decay_epoch:{}'.format(net, net_name, use_random_data, neural_dead_times, filter_FIRE,
#                                                 target_accuracy,
#                                                 predictor_name, round_for_train,
#                                                 tar_acc_gradual_decent,
#                                                 flop_expected, filter_FIRE_decay,
#                                                 neural_dead_times_decay, dataset_name, validation_loader, batch_size,
#                                                 num_workers, optimizer,
#                                                 learning_rate, evaluate_step,
#                                                 num_epoch, filter_preserve_ratio, max_filters_pruned_for_one_time,
#                                                 min_filters_pruned_for_one_time, learning_rate_decay,
#                                                 learning_rate_decay_factor,
#                                                 weight_decay, learning_rate_decay_epoch))
#     print(kwargs)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using: ', end='')
#     if torch.cuda.is_available():
#         print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print(device)
#     net.to(device)
#     '''加载数据集'''
#     if validation_loader is None:
#         validation_loader = data_loader.create_validation_loader(batch_size=batch_size,
#                                                                  num_workers=num_workers,
#                                                                  dataset_name=dataset_name)
#
#     flop_original_net = measure_flops.measure_model(net, dataset_name)
#     original_accuracy = evaluate.evaluate_net(net=net,
#                                               data_loader=validation_loader,
#                                               save_net=False,
#                                               dataset_name=dataset_name,
#                                               )
#     if tar_acc_gradual_decent is True:
#         flop_drop_expected = flop_original_net - flop_expected
#         acc_drop_tolerance = original_accuracy - target_accuracy
#
#     '''计算Conv的层数'''
#     conv_list = []  # List，保存要剪枝的Conv层索引，下标从0开始
#     i = 0  # Conv总数
#     index_in_block = -1
#     filter_num_lower_bound = []  # 最低filter数量
#     filter_num = []
#     for mod in net.modules():
#         if isinstance(mod, resnet_copied.BasicBlock):
#             index_in_block = 1
#         elif isinstance(mod, torch.nn.modules.conv.Conv2d):
#             if index_in_block == 1:  # 在block里面
#                 index_in_block = 2
#                 conv_list.append(i)
#                 filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
#                 filter_num.append(mod.out_channels)
#             elif index_in_block == 2:  # 不需要剪枝的Conv层
#                 index_in_block = -1
#                 filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
#                 filter_num.append(mod.out_channels)
#             elif index_in_block == -1:  # 不在block里面
#                 # conv_list.append(i)
#                 filter_num_lower_bound.append(int(mod.out_channels * filter_preserve_ratio))  # 输出通道数 * filter的保存比例
#                 filter_num.append(mod.out_channels)
#             i += 1
#
#     modules_list=prune.create_modulesList(net)  # 创建一个list保存每一个module的名字
#
#     lived_filter = []
#     dead_filter = []
#
#     # using data to prune the net for (round_for_train)rounds
#     round = 0
#     while True:
#         round += 1
#         print('{} start round {} of filter pruning.'.format(datetime.now(), round))
#
#         if round == round_for_train + 1:
#             # use filters from (round_for_train)rounds to train the classifier
#             '''训练分类器'''
#             ##train the predictor######################################################################################################
#             predictor = predict_dead_filter.predictor(name=predictor_name, **kwargs)
#             predictor.fit(lived_filter=lived_filter, dead_filter=dead_filter)
#
#         if round <= round_for_train:
#             print('{} current filter_FIRE:{},neural_dead_times:{}'.format(datetime.now(), filter_FIRE,
#                                                                                 neural_dead_times))
#             '''找到死亡卷积核'''
#             dead_filter_index, module_list, neural_list \
#                 = evaluate.find_useless_filters_data_version(net=net,
#                                                           filter_FIRE=filter_FIRE,
#                                                           neural_dead_times=neural_dead_times,
#                                                           batch_size=batch_size,
#                                                           use_random_data=use_random_data)
#         else:
#             dead_filter_index \
#                 = evaluate.predict_dead_filters_classifier_version(net=net,
#                                                                    predictor=predictor,
#                                                                    min_ratio_dead_filters=min_filters_pruned_for_one_time,
#                                                                    max_ratio_dead_filters=max_filters_pruned_for_one_time,
#                                                                    filter_num_lower_bound=filter_num_lower_bound)
#
#         net_compressed = False
#         '''卷积核剪枝'''
#         for i in conv_list:
#
#             # ensure the number of filters pruned will not be too large for one time
#             if filter_num[i] * max_filters_pruned_for_one_time < len(dead_filter_index[i]):
#                 dead_filter_index[i] = dead_filter_index[i][:int(filter_num[i] * max_filters_pruned_for_one_time)]
#             # ensure the lower bound of filter number
#             if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
#                 dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]
#
#             filter_num[i] = filter_num[i] - len(dead_filter_index[i])
#             if len(dead_filter_index[i]) > 0:
#                 net_compressed = True
#             print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i], len(dead_filter_index[i])))
#             net = prune.prune_conv_layer_resnet(net=net,
#                                                 layer_index=i + 1,
#                                                 filter_index=dead_filter_index[i],
#                                                 modules_list=modules_list)
#
#         if net_compressed is False:
#             round -= 1
#             filter_FIRE *= filter_FIRE_decay
#             neural_dead_times *= neural_dead_times_decay
#             print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
#             continue
#
#         flop_pruned_net = measure_flops.measure_model(net, dataset_name)
#
#         if tar_acc_gradual_decent is True:  # decent the target_accuracy
#             flop_reduced = flop_original_net - flop_pruned_net
#             target_accuracy = original_accuracy - acc_drop_tolerance * (flop_reduced / flop_drop_expected)
#             print('{} current target accuracy:{}'.format(datetime.now(), target_accuracy))
#
#         success = False
#         # while not success:
#         #     old_net = copy.deepcopy(net)
#         #     success = train.train(net=net,
#         #                           net_name=net_name,
#         #                           num_epochs=num_epoch,
#         #                           target_accuracy=target_accuracy,
#         #                           learning_rate=learning_rate,
#         #                           load_net=False,
#         #                           evaluate_step=evaluate_step,
#         #                           dataset_name=dataset_name,
#         #                           optimizer=optimizer,
#         #                           batch_size=batch_size,
#         #                           learning_rate_decay=learning_rate_decay,
#         #                           learning_rate_decay_factor=learning_rate_decay_factor,
#         #                           weight_decay=weight_decay,
#         #                           learning_rate_decay_epoch=learning_rate_decay_epoch,
#         #                           test_net=True,
#         #                           )
#         #     if not success:
#         #         net = old_net
#         # filter_FIRE *= filter_FIRE_decay
#         # neural_dead_times *= neural_dead_times_decay




# def prune.create_modulesList(net):
#     """
#     创建一个list保存每一个module的名字
#     :param net:
#     :return: modules_list
#     """
#     modules_list = []
#     num_conv = 0  # 统计Conv总数
#     block_i = -1  # block从0开始
#     index_in_block = -1
#     layer = 1  # layer从1开始
#     tag=0
#     for mod in net.modules():
#         if isinstance(mod, torch.nn.modules.container.Sequential):  # layer
#             if tag==0:# 上一层不是Sequential
#                 temp_string = "layer" + str(layer) + "."
#                 tag=1
#             elif tag==1:#上一层是Sequential
#                 layer+=1
#                 temp_string = "layer" + str(layer) + "."
#                 block_i=-1
#                 tag=1
#         elif isinstance(mod,resnet_copied.LambdaLayer):
#             temp_string = "layer" + str(layer) + "."
#         elif isinstance(mod, resnet_copied.BasicBlock):
#             block_i += 1  # block索引
#             index_in_block = 1
#             temp_string += "block" + str(block_i) + "."
#             tag=0
#         elif isinstance(mod, torch.nn.modules.conv.Conv2d):
#             num_conv += 1
#             if index_in_block != -1:  # 在block里面
#                 tmp = temp_string + "conv" + str(index_in_block)
#                 modules_list.append(tmp)  # layer_i.block_j.conv_k
#             elif index_in_block == -1:  # 在block外面
#                 modules_list.append("conv" + str(num_conv))
#
#             tag=0
#         elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
#             if index_in_block != -1:  # 在block里面
#                 tmp = temp_string + "bn" + str(index_in_block)
#                 modules_list.append(tmp)  # layer_i.block_j.bn_k
#             elif index_in_block == -1:  # 在block外面
#                 modules_list.append("bn" + str(num_conv))
#             tag=0
#         elif isinstance(mod, torch.nn.modules.activation.ReLU):
#             if index_in_block != -1:  # 在block里面
#                 tmp = temp_string + "relu" + str(index_in_block)
#                 modules_list.append(tmp)  # layer_i.block_j.relu_k
#                 index_in_block+=1
#             elif index_in_block == -1:  # 在block外面
#                 modules_list.append("relu" + str(num_conv))
#             tag=0
#     return modules_list

#def prune_inactive_neural(net,
#                       net_name,
#                           prune_rate,
#                       target_accuracy,
#                       tar_acc_gradual_decent=False,
#                       flop_expected=None,
#                       dataset_name='imagenet',
#                       use_random_data=False,
#                       validation_loader=None,
#                       batch_size=conf.batch_size,
#                       num_workers=conf.num_workers,
#                       optimizer=optim.Adam,
#                       learning_rate=0.01,
#                       evaluate_step=1000,
#                       num_epoch=350,
#                       filter_preserve_ratio=0.3,
#                       max_filters_pruned_for_one_time=0.5,
#                       learning_rate_decay=False,
#                       learning_rate_decay_factor=conf.learning_rate_decay_factor,
#                       weight_decay=conf.weight_decay,
#                       learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
#                           **kwargs
#                      ):
#     '''
#
#     :param net:
#     :param net_name:
#     :param prune_rate
#     :param target_accuracy: float,
#     :param tar_acc_gradual_decent:bool, if true, the target accuracy will decent from original acc. to target acc. during every round of pruning
#     :param flop_expected: int: expected flop after net pruned. will only work when tar_acc_gradual_decent is true
#     :param dataset_name:
#     :param use_random_data:bool, if true, generated data which fits normal distribution will be used to calculate dead filters.
#     :param validation_loader:
#     :param batch_size:
#     :param num_workers:
#     :param optimizer:
#     :param learning_rate:
#     :param evaluate_step:
#     :param num_epoch:
#     :param filter_preserve_ratio:
#     :param max_filters_pruned_for_one_time:
#     :param learning_rate_decay:
#     :param learning_rate_decay_factor:
#     :param weight_decay:
#     :param learning_rate_decay_epoch:
#     :return:
#     '''
#     #save the output to log
#     print('save log in:' + conf.root_path + net_name + '/log.txt')
#     if not os.path.exists(conf.root_path + net_name ):
#         os.makedirs(conf.root_path + net_name , exist_ok=True)
#     sys.stdout = logger.Logger(conf.root_path+net_name+'/log.txt', sys.stdout)
#     sys.stderr = logger.Logger(conf.root_path+net_name+'/log.txt', sys.stderr)  # redirect std err, if necessary
#
#     print('net:{}\n'
#           'net_name:{}\n'
#           'use_random_data:{}\n'
#           'prune_rate:{}\n'
#           'target_accuracy:{}\n'
#           'tar_acc_gradual_decent:{}\n'
#           'flop_expected:{}\n'
#           'dataset_name:{}\n'
#           'validation_loader:{}\n'
#           'batch_size:{}\n'
#           'num_workers:{}\n'
#           'optimizer:{}\n'
#           'learning_rate:{}\n'
#           'evaluate_step:{}\n'
#           'num_epoch:{}\n'
#           'filter_preserve_ratio:{}\n'
#           'max_filters_pruned_for_one_time:{}\n'
#           'learning_rate_decay:{}\n'
#           'learning_rate_decay_factor:{}\n'
#           'weight_decay:{}\n'
#           'learning_rate_decay_epoch:{}'
#           .format(net,net_name,use_random_data,prune_rate,target_accuracy,tar_acc_gradual_decent,
#                   flop_expected,
#                   dataset_name,validation_loader,batch_size,num_workers,optimizer,learning_rate,evaluate_step,
#                   num_epoch,filter_preserve_ratio,max_filters_pruned_for_one_time,learning_rate_decay,learning_rate_decay_factor,
#                   weight_decay,learning_rate_decay_epoch))
#     print(kwargs)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using: ', end='')
#     if torch.cuda.is_available():
#         print(torch.cuda.device_count(),' * ',end='')
#         print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print(device)
#
#     if validation_loader is None :
#         validation_loader = data_loader.create_validation_loader(batch_size=batch_size,
#                                                                  num_workers=num_workers,
#                                                                  dataset_name=dataset_name,
#                                                                  )
#
#     flop_original_net= measure_flops.measure_model(net, dataset_name)
#     original_accuracy= evaluate.evaluate_net(net=net,
#                                              data_loader=validation_loader,
#                                              save_net=False,
#                                              dataset_name=dataset_name,
#                                              )
#     if tar_acc_gradual_decent is True:
#         flop_drop_expected = flop_original_net - flop_expected
#         acc_drop_tolerance = original_accuracy - target_accuracy
#
#     num_conv = 0  # num of conv layers in the net
#     filter_num_lower_bound=[]
#     filter_num=[]
#     for mod in net.features:
#         if isinstance(mod, torch.nn.modules.conv.Conv2d):
#             num_conv += 1
#             filter_num_lower_bound.append(int(mod.out_channels*filter_preserve_ratio))
#             filter_num.append(mod.out_channels)
#
#     round=0
#     while True:
#         round+=1
#         print('{} start round {} of filter pruning.'.format(datetime.now(), round))
#
#
#         # find dead filters
#         dead_filter_index, module_list, neural_list, FIRE = evaluate.find_useless_filters_data_version(net=net,
#                                                                                                              batch_size=batch_size,
#                                                                                                              use_random_data=use_random_data,
#                                                                                                              percent_of_inactive_filter=prune_rate,
#                                                                                                              dead_or_inactive='inactive'
#                                                                                                              )
#
#         if not os.path.exists(conf.root_path + net_name + '/dead_neural'):
#             os.makedirs(conf.root_path + net_name + '/dead_neural', exist_ok=True)
#
#         checkpoint = {'prune_rate': prune_rate, 'module_list': module_list,
#                       'neural_list': neural_list, 'state_dict': net.state_dict(),
#                       'batch_size': batch_size}
#         checkpoint.update(storage.get_net_information(net, dataset_name, net_name))
#         torch.save(checkpoint,
#                    conf.root_path + net_name + '/dead_neural/round %d.tar' % round, )
#
#
#         net_compressed = False
#         for i in range(num_conv):
#             # ensure the number of filters pruned will not be too large for one time
#             if filter_num[i] * max_filters_pruned_for_one_time < len(dead_filter_index[i]):
#                 dead_filter_index[i] = dead_filter_index[i][:int(filter_num[i] * max_filters_pruned_for_one_time)]
#             # ensure the lower bound of filter number
#             if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
#                 dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]
#             filter_num[i] = filter_num[i] - len(dead_filter_index[i])
#             if len(dead_filter_index[i]) > 0:
#                 net_compressed = True
#
#             print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i],
#                                                                           len(dead_filter_index[i])))
#
#             net= prune.prune_conv_layer_vgg(model=net, layer_index=i + 1, filter_index=dead_filter_index[i])    #prune the dead filter
#
#         if net_compressed is False:
#             os.remove(conf.root_path+net_name+'/dead_neural/round %d.tar'%round)
#             round-=1
#             print('{} round {} did not prune any filters. Restart.'.format(datetime.now(),round+1))
#             continue
#
#         flop_pruned_net= measure_flops.measure_model(net, dataset_name)
#
#
#         if tar_acc_gradual_decent is True:                                      #decent the target_accuracy
#             flop_reduced=flop_original_net-flop_pruned_net
#             target_accuracy=original_accuracy-acc_drop_tolerance*(flop_reduced/flop_drop_expected)
#             print('{} current target accuracy:{}'.format(datetime.now(),target_accuracy))
#
#         success=False
#         while not success:
#             old_net=copy.deepcopy(net)
#             success= train.train(net=net,
#                                  net_name=net_name,
#                                  num_epochs=num_epoch,
#                                  target_accuracy=target_accuracy,
#                                  learning_rate=learning_rate,
#                                  load_net=False,
#                                  evaluate_step=evaluate_step,
#                                  dataset_name=dataset_name,
#                                  optimizer=optimizer,
#                                  batch_size=batch_size,
#                                  learning_rate_decay=learning_rate_decay,
#                                  learning_rate_decay_factor=learning_rate_decay_factor,
#                                  weight_decay=weight_decay,
#                                  learning_rate_decay_epoch=learning_rate_decay_epoch,
#                                  test_net=True,
#                                  **kwargs
#                                  )
#             if not success:
#                 net=old_net




# def prune_filters_randomly(net,
#                            net_name,
#                            target_accuracy,
#                            round_of_prune,
#                            final_filter_num=[15,34,54,68,131,140,127,166,75,69,44,52,52],
#                            tar_acc_gradual_decent=False,
#                            flop_expected=None,
#                            validation_loader=None,
#                            dataset_name='cifar10',
#                            batch_size=conf.batch_size,
#                            num_workers=conf.num_workers,
#                            optimizer=optim.Adam,
#                            learning_rate=0.01,
#                            evaluate_step=1000,
#                            num_epoch=450,
#                            learning_rate_decay=False,
#                            learning_rate_decay_factor=conf.learning_rate_decay_factor,
#                            weight_decay=conf.weight_decay,
#                            learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
#                            **kwargs
#                            ):
#     '''
#
#     :param net:
#     :param net_name:
#     :param target_accuracy:
#     :param round_of_prune:
#     :param final_filter_num:
#     :param tar_acc_gradual_decent:
#     :param flop_expected:
#     :param validation_loader:
#     :param dataset_name:
#     :param batch_size:
#     :param num_workers:
#     :param optimizer:
#     :param learning_rate:
#     :param evaluate_step:
#     :param num_epoch:
#     :param learning_rate_decay:
#     :param learning_rate_decay_factor:
#     :param weight_decay:
#     :param learning_rate_decay_epoch:
#     :return:
#     '''
#     # save the output to log
#     print('save log in:' + conf.root_path + net_name + '/log.txt')
#     if not os.path.exists(conf.root_path + net_name):
#         os.makedirs(conf.root_path + net_name, exist_ok=True)
#     sys.stdout = logger.Logger(conf.root_path + net_name + '/log.txt', sys.stdout)
#     sys.stderr = logger.Logger(conf.root_path + net_name + '/log.txt', sys.stderr)  # redirect std err, if necessary
#
#     print('net:{}\n'
#           'net_name:{}\n'
#           'target_accuracy:{}\n'
#           'round of prune:{}\n'
#           'final_filter_num:{}\n'
#           'tar_acc_gradual_decent:{}\n'
#           'flop_expected:{}\n'
#           'validation_loader:{}\n'
#           'dataset_name:{}\n'
#           'batch_size:{}\n'
#           'num_workers:{}\n'
#           'optimizer:{}\n'
#           'learning_rate:{}\n'
#           'evaluate_step:{}\n'
#           'num_epoch:{}\n'
#           'learning_rate_decay:{}\n'
#           'learning_rate_decay_factor:{}\n'
#           'weight_decay:{}\n'
#           'learning_rate_decay_epoch:{}'
#           .format(net, net_name,   target_accuracy,
#                   round_of_prune,final_filter_num,tar_acc_gradual_decent,flop_expected,validation_loader,
#                   dataset_name, batch_size, num_workers, optimizer,
#                   learning_rate, evaluate_step,
#                   num_epoch, learning_rate_decay,
#                   learning_rate_decay_factor,
#                   weight_decay, learning_rate_decay_epoch))
#     print(kwargs)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print('using: ', end='')
#     if torch.cuda.is_available():
#         print(torch.cuda.device_count(),' * ',end='')
#         print(torch.cuda.get_device_name(torch.cuda.current_device()))
#     else:
#         print(device)
#
#     if validation_loader is None :
#         validation_loader = data_loader.create_validation_loader(batch_size=batch_size,
#                                                                  num_workers=num_workers,
#                                                                  dataset_name=dataset_name,
#                                                                  )
#
#     flop_original_net = measure_flops.measure_model(net, dataset_name)
#     original_accuracy = evaluate.evaluate_net(net=net,
#                                               data_loader=validation_loader,
#                                               save_net=False,
#                                               dataset_name=dataset_name,
#                                               )
#     if tar_acc_gradual_decent is True:
#         flop_drop_expected = flop_original_net - flop_expected
#         acc_drop_tolerance = original_accuracy - target_accuracy
#
#     num_conv = 0  # num of conv layers in the net
#     filter_num=[]
#     for mod in net.features:
#         if isinstance(mod, torch.nn.modules.conv.Conv2d):
#             num_conv += 1
#             filter_num.append(mod.out_channels)
#
#     round=0
#     while round<round_of_prune:
#         round+=1
#         print('{} start round {} of random filter pruning.'.format(datetime.now(),round))
#         for i in range(num_conv):
#             num_to_prune=int((filter_num[i]-final_filter_num[i])/(round_of_prune-round+1))           #number of filters to prune in this round
#             filter_num[i]=filter_num[i]-num_to_prune                                            #update the number of filters in layer i
#             pruning_filter_index=random.sample(range(0,filter_num[i]),num_to_prune)
#             print('layer {}: remain {} filters, prune {} filters.'.format(i, filter_num[i],
#                                                                           num_to_prune))
#             net = prune.prune_conv_layer_vgg(model=net, layer_index=i + 1,
#                                          filter_index=pruning_filter_index)  # prune the dead filter
#
#
#
#         flop_pruned_net = measure_flops.measure_model(net, dataset_name)
#
#         if tar_acc_gradual_decent is True:                                      #decent the target_accuracy
#             flop_reduced=flop_original_net-flop_pruned_net
#             target_accuracy=min(0.935,original_accuracy-acc_drop_tolerance*(flop_reduced/flop_drop_expected))
#             print('{} current target accuracy:{}'.format(datetime.now(),target_accuracy))
#
#         success=False
#         while not success:
#             old_net = copy.deepcopy(net)
#             success = train.train(net=net,
#                                   net_name=net_name,
#                                   num_epochs=num_epoch,
#                                   target_accuracy=target_accuracy,
#                                   learning_rate=learning_rate,
#                                   load_net=False,
#                                   evaluate_step=evaluate_step,
#                                   dataset_name=dataset_name,
#                                   optimizer=optimizer,
#                                   batch_size=batch_size,
#                                   learning_rate_decay=learning_rate_decay,
#                                   learning_rate_decay_factor=learning_rate_decay_factor,
#                                   weight_decay=weight_decay,
#                                   learning_rate_decay_epoch=learning_rate_decay_epoch,
#                                   test_net=True,
#                                   **kwargs
#                                   )
#             if not success:
#                 net = old_net



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net=storage.restore_net(checkpoint=torch.load(os.path.join(conf.root_path,'baseline/vgg16_bn_cifar10,accuracy=0.941.tar')))
    max_filters_pruned_for_one_time=[0.3 for i in range(13)]
    # max_filters_pruned_for_one_time[5]=0
    max_filters_pruned_for_one_time[5]=0.1
    max_filters_pruned_for_one_time[6]=0
    max_filters_pruned_for_one_time[7]=0.01
    max_filters_pruned_for_one_time[8]=0.01
    max_filters_pruned_for_one_time[9]=0.01
    max_filters_pruned_for_one_time[10]=0.6
    max_filters_pruned_for_one_time[11]=0.6
    max_filters_pruned_for_one_time[12]=0.6

    # max_filters_pruned_for_one_time=[0 for i in range(13)]
    # max_filters_pruned_for_one_time[11]=0.1

    prune_inactive_neural_with_extractor_consistent_training(net=net,
                                                         net_name='vgg16_bn',
                                                         exp_name='test',
                                                         # exp_name='vgg16_extractor_static_cifar10_0.1prunerate',
                                                         prune_rate=0.03,
                                                         round_for_train=2,
                                                         round_to_train_freq=6,
                                                         flop_expected=4e7,
                                                         dataset_name='cifar10',
                                                         batch_size=512,
                                                         num_workers=8,
                                                         optimizer=optim.SGD,
                                                         learning_rate=0.001,
                                                         evaluate_step=3000,
                                                         num_epoch=10,
                                                         filter_preserve_ratio=0.2,
                                                         max_filters_pruned_for_one_time=max_filters_pruned_for_one_time,
                                                         learning_rate_decay=True,
                                                         learning_rate_decay_factor=0.5,
                                                         weight_decay=5e-4,
                                                         learning_rate_decay_epoch=[20,50, 100, 150, 250, 300, 350, 400],
                                                         max_training_round=2,
                                                         round=19,
                                                         top_acc=1,
                                                         max_data_to_test=10000,
                                                         extractor_epoch=100,
                                                         extractor_feature_len=15,
                                                         gcn_rounds=2
                                                         )



    #
    # checkpoint = torch.load('../data/baseline/vgg16bn_cifar100_0.72630_t+v.tar')
    # checkpoint=torch.load('/home/zzj/fang/model_pytorch/model_saved/vgg16bn_cifar10_realdata_regressor5_大幅度/checkpoint/flop=49582154,accuracy=0.93330.tar')
    # net=checkpoint['net']

    # checkpoint = torch.load('../data/baseline/resnet56_cifar10,accuracy=0.93280.tar')
    # checkpoint=torch.load('../data/baseline/resnet56_cifar10,accuracy=0.94230.tar')
    # checkpoint=torch.load('/home/zzj/fang/model_pytorch/model_saved/resnet56_cifar10_regressor_prunedBaseline2/checkpoint/flop=52642442,accuracy=0.93320.tar')
    # net = resnet_copied.resnet56().to(device)
    # net = net=storage.restore_net(checkpoint).to(device)

    # checkpoint=torch.load('/home/disk_new/model_saved/resnet56_cifar10_DeadNeural_realdata_good_baseline_过得去/代表/sample_num=13300000,accuracy=0.93610，flop=65931914.tar')
    # net=checkpoint['net']

    # net.load_state_dict(checkpoint['state_dict'])
    # print(checkpoint['highest_accuracy'])
    #
    # measure_flops.measure_model(net, 'cifar100', print_flop=True)
    #
    # prune_inactive_neural_with_regressor(net=net,
    #                                      exp_name='vgg16bn_base_v+t_cifar100',
    #                                      net_name='vgg16_bn',
    #                                      prune_rate=0.15,
    #                                      load_regressor=False,
    #                                      dataset_name='cifar100',
    #                                      filter_preserve_ratio=0.15,
    #                                      max_filters_pruned_for_one_time=0.2,
    #                                      # [0.11,0.11,0.11,0.11,0.11,0.11,0.08,0.11,0.11,0.11,0.2,0.2,0.2],
    #                                      target_accuracy=0.717,
    #                                      tar_acc_gradual_decent=True,
    #                                      flop_expected=1.7e8,
    #                                      batch_size=1600,
    #                                      num_epoch=450,
    #                                      evaluate_step=3000,
    #                                      use_random_data=False,
    #                                      round_for_train=2,
    #                                      # optimizer=optim.Adam,
    #                                      # learning_rate=1e-3,
    #                                      # weight_decay=0
    #                                      optimizer=optim.SGD,
    #                                      learning_rate=0.0001,
    #                                      weight_decay=5e-4,
    #                                      learning_rate_decay=True,
    #                                      learning_rate_decay_epoch=[20, 100, 150, 250, 300, 350, 400],
    #                                      learning_rate_decay_factor=0.5,
    #                                      max_training_round=2
    #                                      )

    # for original baseline
    # prune_inactive_neural_with_regressor_resnet(net=net,
    #                                             net_name='resnet56_cifar10_regressor2',
    #                                             prune_rate=0.15,
    #                                             load_regressor=False,
    #                                             dataset_name='cifar10',
    #                                             filter_preserve_ratio=0.15,
    #                                             max_filters_pruned_for_one_time=0.2,
    #                                             target_accuracy=0.93,
    #                                             tar_acc_gradual_decent=True,
    #                                             flop_expected=4e7,
    #                                             batch_size=1600,
    #                                             num_epoch=450,
    #                                             evaluate_step=3000,
    #                                             use_random_data=False,
    #                                             round_for_train=2,
    #                                             # optimizer=optim.Adam,
    #                                             # learning_rate=1e-3,
    #                                             # weight_decay=0
    #                                             optimizer=optim.SGD,
    #                                             learning_rate=0.01,
    #                                             learning_rate_decay=True,
    #                                             learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #                                             learning_rate_decay_factor=0.5,
    #                                             )
    # prune_resnet(net=net,
    #              net_name='tmp',
    #              neural_dead_times=9000,
    #              filter_FIRE=0.9,
    #              target_accuracy=0.93,
    #              tar_acc_gradual_decent=True,
    #              dataset_name='cifar10',
    #              flop_expected=4e7,
    #              round_for_train=20,
    #              use_random_data=False,
    #              batch_size=1600,
    #              optimizer=optim.SGD,
    #              learning_rate=0.01,
    #              learning_rate_decay=True,
    #              learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #              learning_rate_decay_factor=0.5,
    #              neural_dead_times_decay=0.95,
    #              filter_FIRE_decay=0.98,
    #              filter_preserve_ratio=0.1,
    #              max_filters_pruned_for_one_time=0.3,
    #              )


    # prune_filters_randomly(net=net,
    #
    #                         net_name='vgg16bn_cifar10_randomly_pruned_acc_not_decent',
    #                        round_of_prune=11,
    #                        final_filter_num=[15,34,54,68,131,140,127,166,75,69,44,52,52],
    #                        dataset_name='cifar10',
    #                         tar_acc_gradual_decent=False,
    #                          target_accuracy=0.933,
    #                          batch_size=1600,
    #                          num_epoch=450,
    #                          evaluate_step=1600,
    #
    #
    #                          optimizer=optim.SGD,
    #                          learning_rate=0.01,
    #                          learning_rate_decay=True,
    #                          learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #                          learning_rate_decay_factor=0.5,
    #
    #                                                  )

    # prune_inactive_neural(net=net,
    #                       net_name='vgg16bn_cifar10_inactiveFilter',
    #                       dataset_name='cifar10',
    #                       prune_rate=0.1,
    #
    #                       filter_preserve_ratio=0.1,
    #                       max_filters_pruned_for_one_time=0.3,
    #                       target_accuracy=0.933,
    #                       tar_acc_gradual_decent=True,
    #                       flop_expected=5e7,
    #                       batch_size=1600,
    #                       num_epoch=300,
    #                       evaluate_step=1600,
    #                       use_random_data=True,
    #                       # optimizer=optim.Adam,
    #                       # learning_rate=1e-3,
    #                       # weight_decay=0
    #                       optimizer=optim.SGD,
    #                       learning_rate=0.01,
    #                       learning_rate_decay=True,
    #                       learning_rate_decay_epoch=[50, 100, 150, 250, 300, 350, 400],
    #                       learning_rate_decay_factor=0.5,
    # )


    net=resnet.resnet50(pretrained=True).to(device)
    checkpoint=torch.load('/home/victorfang/model_pytorch/data/model_saved/resnet50_pureData/checkpoint/flop=3326272945,accuracy=0.73972.tar')
    net = storage.restore_net(checkpoint, pretrained=True)

    net=torch.nn.DataParallel(net)

    prune_inactive_neural_with_regressor_resnet(net=net,
                                                net_name='resnet50',
                                                exp_name='resnet50_pureData',
                                                prune_rate=0.05,
                                                round_for_train=10,
                                                tar_acc_gradual_decent=True,
                                                batch_size=512,
                                                num_workers=8,

                                                # optimizer=optim.Adam,
                                                # learning_rate=0.0001,
                                                # weight_decay=5e-4,

                                                optimizer=optim.SGD,
                                                learning_rate=0.001,
                                                weight_decay=5e-4,
                                                momentum=0.9,
                                                learning_rate_decay=True,
                                                learning_rate_decay_factor=10,
                                                learning_rate_decay_epoch=[5,10,15],


                                                num_epoch=20,
                                                max_filters_pruned_for_one_time=0.2,

                                                max_training_round=2,
                                                flop_expected=2e9,
                                                target_accuracy=0.7,

                                                round=7

                                                )