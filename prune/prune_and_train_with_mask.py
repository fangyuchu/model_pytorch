from network import net_with_mask,storage,vgg
from framework import train,evaluate,data_loader,config as conf
from framework.measure_flops import measure_model
from filter_characteristic.filter_feature_extractor import train_extractor,load_extractor
import logger
from torch import nn
from torch import optim
import torch
import os,sys,math
from datetime import datetime
from copy import deepcopy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_information_for_pruned_conv(net,net_name,filter_preserve_ratio):
    conv_list=[]                    #layer of the conv to prune
    filter_num_lower_bound = []  # 最低filter数量
    filter_num = []
    layer=0
    for name,mod in net.named_modules():
        if isinstance(mod,nn.Conv2d) and 'downsample' not in name:
            filter_num_lower_bound += [int(mod.out_channels * filter_preserve_ratio)]  # 输出通道数 * filter的保存比例
            filter_num += [mod.out_channels]
            if 'vgg' in net_name:
                conv_list+=[layer]
            #todo: might be wrong for resnet on cifar
            elif 'resnet' in net_name:
                if 'layer' in name and 'conv3' not in name:
                    conv_list+=[layer]
            else:
                raise Exception('Unknown net_name:'+net_name)
            layer+=1
    return conv_list,filter_num,filter_num_lower_bound
def dynamic_prune_inactive_neural_with_feature_extractor(net,
                                                         net_name,
                                                         exp_name,
                                                         target_accuracy,
                                                         initial_prune_rate,
                                                         delta_prune_rate=0.02,
                                                         round_for_train=2,
                                                         tar_acc_gradual_decent=False,
                                                         flop_expected=None,
                                                         dataset_name='imagenet',
                                                         batch_size=conf.batch_size,
                                                         num_workers=conf.num_workers,
                                                         learning_rate=0.01,
                                                         evaluate_step=1200,
                                                         num_epoch=450,
                                                         num_epoch_after_finetune=20,
                                                         filter_preserve_ratio=0.3,
                                                         # max_filters_pruned_for_one_time=0.5,
                                                         optimizer=optim.Adam,
                                                         learning_rate_decay=False,
                                                         learning_rate_decay_factor=conf.learning_rate_decay_factor,
                                                         weight_decay=conf.weight_decay,
                                                         learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
                                                         max_training_round=9999,
                                                         round=1,
                                                         top_acc=1,
                                                         max_data_to_test=10000,
                                                         **kwargs
                                                         ):
    '''

       :param net:
       :param net_name:
       :param exp_name:
       :param target_accuracy:
       :param initial_prune_rate:
       :param delta_prune_rate: increase/decrease of prune_rate after each round
       :param round_for_train:
       :param tar_acc_gradual_decent:
       :param flop_expected:
       :param dataset_name:
       :param batch_size:
       :param num_workers:
       :param optimizer:
       :param learning_rate:
       :param evaluate_step:
       :param num_epoch:
       :param num_epoch_after_finetune: epoch to train after fine-tune the pruned net
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
    print('save log in:' + os.path.join(conf.root_path, 'model_saved', exp_name, 'log.txt'))
    if not os.path.exists(os.path.join(conf.root_path, 'model_saved', exp_name)):
        os.makedirs(os.path.join(conf.root_path, 'model_saved', exp_name), exist_ok=True)
    sys.stdout = logger.Logger(os.path.join(conf.root_path, 'model_saved', exp_name, 'log.txt'), sys.stdout)
    sys.stderr = logger.Logger(os.path.join(conf.root_path, 'model_saved', exp_name, 'log.txt'),
                               sys.stderr)  # redirect std err, if necessary

    print('net:', net)
    print('net_name:', net_name)
    print('exp_name:', exp_name)
    print('target_accuracy:', target_accuracy)
    print('initial_prune_rate:', initial_prune_rate)
    print('delta_prune_rate:',delta_prune_rate)
    print('round_for_train:', round_for_train)
    print('tar_acc_gradual_decent:', tar_acc_gradual_decent)
    print('flop_expected:', flop_expected)
    print('dataset_name:', dataset_name)
    print('batch_size:', batch_size)
    print('num_workers:', num_workers)
    print('optimizer:', optimizer)
    print('learning_rate:', learning_rate)
    print('evaluate_step:', evaluate_step)
    print('num_epoch:', num_epoch)
    print('num_epoch_after_finetune:',num_epoch_after_finetune)
    print('filter_preserve_ratio:', filter_preserve_ratio)
    # print('max_filters_pruned_for_one_time:', max_filters_pruned_for_one_time)
    print('learning_rate_decay:', learning_rate_decay)
    print('learning_rate_decay_factor:', learning_rate_decay_factor)
    print('weight_decay:', weight_decay)
    print('learning_rate_decay_epoch:', learning_rate_decay_epoch)
    print('max_training_round:', max_training_round)
    print('top_acc:', top_acc)
    print('round:', round)
    print(kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.device_count(),' * ',end='')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)

    checkpoint_path=os.path.join(conf.root_path, 'model_saved', exp_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    validation_loader = data_loader.create_validation_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_name=dataset_name,
    )

    if isinstance(net,nn.DataParallel):
        net_entity=net.module
    else:
        net_entity=net

    flop_original_net=flop_before_prune = measure_model(net_entity.prune(), dataset_name)

    original_accuracy = evaluate.evaluate_net(net=net,
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

    conv_list, filter_num, filter_num_lower_bound=get_information_for_pruned_conv(net_entity,net_name,filter_preserve_ratio)
    num_filters_to_prune_at_most=[]
    for i in range(len(filter_num)):
        num_filters_to_prune_at_most+=[filter_num[i]-filter_num_lower_bound[i]]

    prune_rate=initial_prune_rate
    while True:
        print('{} start round {} of filter pruning.'.format(datetime.now(), round))
        print('{} current prune_rate:{}'.format(datetime.now(),prune_rate))
        #todo 待考虑
        net_entity.reset_mask()
        if round <= round_for_train:
            dead_filter_index, module_list, neural_list, FIRE_tmp = evaluate.find_useless_filters_data_version(
                net=net_entity,
                batch_size=16,                                                                                                  #this function need to run on sigle gpu
                percent_of_inactive_filter=prune_rate,
                dead_or_inactive='inactive',
                dataset_name=dataset_name,
                max_data_to_test=max_data_to_test,
                num_filters_to_prune_at_most=num_filters_to_prune_at_most,
            )
            num_test_images = math.ceil(max_data_to_test / 16) * 16                                                             #Warning, this goes wrong if dataset_size is smaller
            if not os.path.exists(os.path.join(checkpoint_path, 'dead_neural')):
                os.makedirs(os.path.join(checkpoint_path, 'dead_neural'), exist_ok=True)

            checkpoint = {'prune_rate': prune_rate, 'module_list': module_list,
                          'neural_list': neural_list, 'state_dict': net_entity.state_dict(),
                          'num_test_images':num_test_images}
            checkpoint.update(storage.get_net_information(net_entity, dataset_name, net_name))
            torch.save(checkpoint,
                       os.path.join(checkpoint_path, 'dead_neural/round %d.tar' % round)
                       )
        else :
            print('hahaha')
            return

        #todo:round>round_for_train,use filter feature extractor

        '''卷积核剪枝'''
        for i in conv_list:
            #todo 待考虑
            # # ensure the number of filters pruned will not be too large for one time
            # if type(max_filters_pruned_for_one_time) is list:
            #     num_filters_to_prune_max = filter_num[i] * max_filters_pruned_for_one_time[i]
            # else:
            #     num_filters_to_prune_max = filter_num[i] * max_filters_pruned_for_one_time
            # if num_filters_to_prune_max < len(dead_filter_index[i]):
            #     dead_filter_index[i] = dead_filter_index[i][:int(num_filters_to_prune_max)]
            # ensure the lower bound of filter number
            if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
                raise Exception('i think something is wrong')
                dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]

            print('layer {}: has {} filters, prunes {} filters, remains {} filters.'.
                  format(i, filter_num[i], len(dead_filter_index[i]),filter_num[i]-len(dead_filter_index[i])))
            net_entity.mask_filters(i,dead_filter_index[i])

        flop_after_prune = measure_model(net_entity.prune(), dataset_name)
        net_compressed= (flop_after_prune != flop_before_prune)
        if net_compressed is False:
            round -= 1
            print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
            prune_rate += 0.02
            continue


        if tar_acc_gradual_decent is True:  # decent the target_accuracy
            flop_reduced = flop_original_net - flop_after_prune
            target_accuracy = original_accuracy - acc_drop_tolerance * (flop_reduced / flop_drop_expected)
            print('{} current target accuracy:{}'.format(datetime.now(), target_accuracy))

        success = False
        training_round=0
        while not success:
            old_net = deepcopy(net)
            training_round+=1
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
                                  no_grad=['mask']
                                  )
            if success:
                prune_rate+= delta_prune_rate
                round += 1
                flop_before_prune=flop_after_prune
                net_entity.reset_mask()
                if num_epoch_after_finetune ==0:
                    break
                train.train(net=net,
                            net_name=net_name,
                            exp_name=exp_name,
                            num_epochs=num_epoch_after_finetune,
                            learning_rate=learning_rate,
                            load_net=False,
                            evaluate_step=evaluate_step,
                            dataset_name=dataset_name,
                            optimizer=optimizer,
                            batch_size=batch_size,
                            test_net=True,
                            top_acc=top_acc,
                            no_grad=['mask']
                            )
            else:
                net = old_net
                if isinstance(net, nn.DataParallel):
                    net_entity = net.module
                else:
                    net_entity = net
                net_entity.reset_mask()
                if max_training_round == training_round:
                    prune_rate-=delta_prune_rate/2
                    if prune_rate<=0:
                        print('{} failed to prune the net, pruning stop.'.format(datetime.now()))
                        return
                    break
def static_prune_inactive_neural_with_feature_extractor(net,
                                                         net_name,
                                                         exp_name,
                                                         target_accuracy,
                                                         initial_prune_rate,
                                                         delta_prune_rate=0.02,
                                                         round_for_train=2,
                                                         tar_acc_gradual_decent=False,
                                                         flop_expected=None,
                                                         dataset_name='imagenet',
                                                         batch_size=conf.batch_size,
                                                         num_workers=conf.num_workers,
                                                         learning_rate=0.01,
                                                         evaluate_step=1200,
                                                         num_epoch=450,
                                                         filter_preserve_ratio=0.3,
                                                         optimizer=optim.Adam,
                                                         learning_rate_decay=False,
                                                         learning_rate_decay_factor=conf.learning_rate_decay_factor,
                                                         weight_decay=conf.weight_decay,
                                                         learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
                                                         max_training_round=9999,
                                                         round=1,
                                                         top_acc=1,
                                                         max_data_to_test=10000,
                                                        extractor_epoch=1001,
                                                        extractor_feature_len=27,
                                                        gcn_rounds=2,
                                                         **kwargs
                                                         ):
    '''

       :param net:
       :param net_name:
       :param exp_name:
       :param target_accuracy:
       :param initial_prune_rate:
       :param delta_prune_rate: increase/decrease of prune_rate after each round
       :param round_for_train:
       :param tar_acc_gradual_decent:
       :param flop_expected:
       :param dataset_name:
       :param batch_size:
       :param num_workers:
       :param optimizer:
       :param learning_rate:
       :param evaluate_step:
       :param num_epoch:
       :param filter_preserve_ratio:
       :param learning_rate_decay:
       :param learning_rate_decay_factor:
       :param weight_decay:
       :param learning_rate_decay_epoch:
       :param max_training_round:if the net can't reach target accuracy in max_training_round , the program stop.
       :param top_acc:
       :param extractor_epoch
       :param extractor_feature_len
       :param gcn_rounds
       :param kwargs:
       :return:
       '''

    # save the output to log
    print('save log in:' + os.path.join(conf.root_path, 'model_saved', exp_name, 'log.txt'))
    if not os.path.exists(os.path.join(conf.root_path, 'model_saved', exp_name)):
        os.makedirs(os.path.join(conf.root_path, 'model_saved', exp_name), exist_ok=True)
    sys.stdout = logger.Logger(os.path.join(conf.root_path, 'model_saved', exp_name, 'log.txt'), sys.stdout)
    sys.stderr = logger.Logger(os.path.join(conf.root_path, 'model_saved', exp_name, 'log.txt'),
                               sys.stderr)  # redirect std err, if necessary

    print('net:', net)
    print('net_name:', net_name)
    print('exp_name:', exp_name)
    print('target_accuracy:', target_accuracy)
    print('initial_prune_rate:', initial_prune_rate)
    print('delta_prune_rate:',delta_prune_rate)
    print('round_for_train:', round_for_train)
    print('tar_acc_gradual_decent:', tar_acc_gradual_decent)
    print('flop_expected:', flop_expected)
    print('dataset_name:', dataset_name)
    print('batch_size:', batch_size)
    print('num_workers:', num_workers)
    print('optimizer:', optimizer)
    print('learning_rate:', learning_rate)
    print('evaluate_step:', evaluate_step)
    print('num_epoch:', num_epoch)
    print('filter_preserve_ratio:', filter_preserve_ratio)
    print('learning_rate_decay:', learning_rate_decay)
    print('learning_rate_decay_factor:', learning_rate_decay_factor)
    print('weight_decay:', weight_decay)
    print('learning_rate_decay_epoch:', learning_rate_decay_epoch)
    print('max_training_round:', max_training_round)
    print('top_acc:', top_acc)
    print('round:', round)
    print('extractor_epoch:',extractor_epoch)
    print('extractor_feature_len:',extractor_feature_len)
    print('gcn_rounds:',gcn_rounds)
    print(kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.device_count(),' * ',end='')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)

    checkpoint_path=os.path.join(conf.root_path, 'model_saved', exp_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    validation_loader = data_loader.create_validation_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_name=dataset_name,
    )

    if isinstance(net,nn.DataParallel):
        net_entity=net.module
    else:
        net_entity=net

    flop_original_net=flop_before_prune = measure_model(net_entity.prune(), dataset_name)

    original_accuracy = evaluate.evaluate_net(net=net,
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

    conv_list, filter_num, filter_num_lower_bound=get_information_for_pruned_conv(net_entity,net_name,filter_preserve_ratio)
    num_filters_to_prune_at_most=[]
    for i in range(len(filter_num)):
        num_filters_to_prune_at_most+=[filter_num[i]-filter_num_lower_bound[i]]

    prune_rate=initial_prune_rate
    while True:
        print('{} start round {} of filter pruning.'.format(datetime.now(), round))
        print('{} current prune_rate:{}'.format(datetime.now(),prune_rate))
        if round <= round_for_train:
            dead_filter_index, module_list, neural_list, FIRE_tmp = evaluate.find_useless_filters_data_version(
                net=net_entity,
                batch_size=16,                                                                                                  #this function need to run on sigle gpu
                percent_of_inactive_filter=prune_rate,
                dead_or_inactive='inactive',
                dataset_name=dataset_name,
                max_data_to_test=max_data_to_test,
                num_filters_to_prune_at_most=num_filters_to_prune_at_most,
            )
            num_test_images = math.ceil(max_data_to_test / 16) * 16                                                             #Warning, this goes wrong if dataset_size is smaller
            if not os.path.exists(os.path.join(checkpoint_path, 'dead_neural')):
                os.makedirs(os.path.join(checkpoint_path, 'dead_neural'), exist_ok=True)

            checkpoint = {'prune_rate': prune_rate, 'module_list': module_list,
                          'neural_list': neural_list, 'state_dict': net_entity.state_dict(),
                          'num_test_images':num_test_images}
            checkpoint.update(storage.get_net_information(net_entity, dataset_name, net_name))
            torch.save(checkpoint,
                       os.path.join(checkpoint_path, 'dead_neural/round %d.tar' % round)
                       )
            if round == round_for_train:
                train_extractor(train_data_dir=os.path.join(checkpoint_path,'dead_neural'),
                                net_name=net_name,
                                dataset_name=dataset_name,
                                num_images=num_test_images,
                                epoch=extractor_epoch,
                                feature_len=extractor_feature_len,
                                gcn_rounds=gcn_rounds,
                                checkpoint_path=os.path.join(checkpoint_path,'extractor'))
                extractor=load_extractor(os.path.join(checkpoint_path,'extractor',str(extractor_epoch),'.tar'))
        else :
            print('jajajajja')


        #todo:round>round_for_train,use filter feature extractor

        '''卷积核剪枝'''
        for i in conv_list:
            #todo 待考虑
            # # ensure the number of filters pruned will not be too large for one time
            # if type(max_filters_pruned_for_one_time) is list:
            #     num_filters_to_prune_max = filter_num[i] * max_filters_pruned_for_one_time[i]
            # else:
            #     num_filters_to_prune_max = filter_num[i] * max_filters_pruned_for_one_time
            # if num_filters_to_prune_max < len(dead_filter_index[i]):
            #     dead_filter_index[i] = dead_filter_index[i][:int(num_filters_to_prune_max)]
            # ensure the lower bound of filter number
            if filter_num[i] - len(dead_filter_index[i]) < filter_num_lower_bound[i]:
                raise Exception('i think something is wrong')
                dead_filter_index[i] = dead_filter_index[i][:filter_num[i] - filter_num_lower_bound[i]]

            print('layer {}: has {} filters, prunes {} filters, remains {} filters.'.
                  format(i, filter_num[i], len(dead_filter_index[i]),filter_num[i]-len(dead_filter_index[i])))
            net_entity.mask_filters(i,dead_filter_index[i])

        flop_after_prune = measure_model(net_entity.prune(), dataset_name)
        net_compressed= (flop_after_prune != flop_before_prune)
        if net_compressed is False:
            round -= 1
            print('{} round {} did not prune any filters. Restart.'.format(datetime.now(), round + 1))
            prune_rate += 0.02
            continue


        if tar_acc_gradual_decent is True:  # decent the target_accuracy
            flop_reduced = flop_original_net - flop_after_prune
            target_accuracy = original_accuracy - acc_drop_tolerance * (flop_reduced / flop_drop_expected)
            print('{} current target accuracy:{}'.format(datetime.now(), target_accuracy))

        success = False
        training_round=0
        while not success:
            old_net = deepcopy(net)
            training_round+=1
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
                                  no_grad=['mask']
                                  )
            if success:
                prune_rate+= delta_prune_rate
                round += 1
                flop_before_prune=flop_after_prune

            else:
                net = old_net
                if isinstance(net, nn.DataParallel):
                    net_entity = net.module
                else:
                    net_entity = net
                if max_training_round == training_round:
                    prune_rate-=delta_prune_rate/2
                    if prune_rate<=0:
                        print('{} failed to prune the net, pruning stop.'.format(datetime.now()))
                        return
                    break


if __name__ == "__main__":
    print(torch.cuda.is_available())

    net = net_with_mask.NetWithMask(dataset_name='cifar10', net_name='vgg16_bn')
    net=nn.DataParallel(net)
    # net = nn.DataParallel(net)



    static_prune_inactive_neural_with_feature_extractor(net=net,
                                                         net_name='vgg16_bn',
                                                         exp_name='vgg16_cifar_static_mask_net_big_delta',

                                                         target_accuracy=0.93,
                                                         initial_prune_rate=0.1,
                                                         delta_prune_rate=0.05,
                                                         round_for_train=100,
                                                         tar_acc_gradual_decent=True,
                                                         flop_expected=4e7,
                                                         dataset_name='cifar10',
                                                         batch_size=512,
                                                         num_workers=4,
                                                         evaluate_step=3000,
                                                         num_epoch=450,
                                                         filter_preserve_ratio=0.3,
                                                         optimizer=optim.SGD,
                                                         learning_rate=0.001,
                                                         learning_rate_decay=True,
                                                         learning_rate_decay_factor=0.5,
                                                         weight_decay=5e-4,
                                                         learning_rate_decay_epoch=[50,100,150,200,250,300,350,400],
                                                         max_training_round=1,
                                                         round=1,
                                                         top_acc=1,
                                                         max_data_to_test=10000
                                                         )
