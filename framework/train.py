import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from network import modules
import os
from datetime import datetime
import math
import matplotlib.pyplot as plt
from framework import data_loader, measure_flops, evaluate, config as conf
from math import ceil
from network import storage
from torch.utils.tensorboard import SummaryWriter
from framework.draw import draw_masked_net


def exponential_decay_learning_rate(optimizer, sample_num, train_set_size,learning_rate_decay_epoch,learning_rate_decay_factor,batch_size):
    """Sets the learning rate to the initial LR decayed by learning_rate_decay_factor every decay_steps"""
    current_epoch=ceil(sample_num/train_set_size)
    if learning_rate_decay_factor>1:
        learning_rate_decay_factor=1/learning_rate_decay_factor             #to prevent the mistake
    if current_epoch in learning_rate_decay_epoch and sample_num-(train_set_size*(current_epoch-1))<=batch_size:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*learning_rate_decay_factor
            lr=param_group['lr']
        print('{} learning rate at present is {}'.format(datetime.now(), lr))


def set_learning_rate(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_modules_no_grad(net, module_no_grad):
    '''
    :param net:
    :param module_no_grad: module name which need to be trained
    :return: a dict containing parameter names that don't need grad
    '''
    requires_grad_dict=dict()
    for name, _ in net.named_parameters():
        requires_grad_dict[name] = True
        if type(module_no_grad) is list:
            for mod_name in module_no_grad:
                if mod_name in name:
                    requires_grad_dict[name]=False
                    print(name)
        else:
            if module_no_grad in name:
                requires_grad_dict[name]=False
    requires_grad_dict['default']=True
    return requires_grad_dict

def look_up_hyperparameter(parameter_dict,parameter_name, hyperparameter_type):
    '''
    find the specified hyper-parameter for each parameter
    :param parameter_dict: a dict containing(key:partial name of the parameter, value: hyperparameter)
    :param parameter_name: name of the parameter in model
    :param hyperparameter_type: type of the hyperparameter. e.g 'learning rate', 'momentum'
    :return:
    '''
    if type(parameter_dict) is not dict:
        return parameter_dict
    if 'default' not in parameter_dict.keys():
        raise Exception('Default '+hyperparameter_type+' is not given.')
    for key in parameter_dict.keys():
        if key in parameter_name:
            return parameter_dict[key]
    return parameter_dict['default']

def prepare_optimizer(
        net,
        optimizer,
        momentum=conf.momentum,
        learning_rate=conf.learning_rate,
        weight_decay=conf.weight_decay,
        requires_grad=True,
        # **kwargs
):
    param_list=[]
    for name, value in net.named_parameters():
        value.requires_grad = look_up_hyperparameter(requires_grad,name,'requires_grad')
        if value.requires_grad is True:
            m=look_up_hyperparameter(momentum,name,'momentum')
            lr=look_up_hyperparameter(learning_rate,name,'learning rate')
            wd=look_up_hyperparameter(weight_decay,name,'weight decay')
            param_list+=[{'params':value,
                          'lr':lr,
                          'initial_lr':lr,
                          'weight_decay':wd,
                          'momentum':m}]

    optimizer=optimizer(param_list,lr=look_up_hyperparameter(learning_rate,'default','lr'))



    # if optimizer is optim.Adam:
    #     # optimizer = optimizer([{'params':filter(lambda p: p.requires_grad, net.parameters()),'initial_lr':learning_rate}],
    #     #                       lr=learning_rate,
    #     #                       weight_decay=weight_decay,**kwargs)
    # elif optimizer is optim.SGD:
    #     optimizer=optimizer([{'params':filter(lambda p: p.requires_grad, net.parameters()),'initial_lr':learning_rate}],
    #                         lr=learning_rate,
    #                         weight_decay=weight_decay,
    #                         momentum=momentum,**kwargs)

    return optimizer


    


def train(
        net,
        net_name,
        exp_name='',
        dataset_name='imagenet',
        train_loader=None,
        validation_loader=None,
        learning_rate=conf.learning_rate,
        num_epochs=conf.num_epochs,
        batch_size=conf.batch_size,
        evaluate_step=conf.evaluate_step,
        load_net=True,
        test_net=False,
        root_path=conf.root_path,
        # checkpoint_path=None,
        momentum=conf.momentum,
        num_workers=conf.num_workers,
        learning_rate_decay=False,
        learning_rate_decay_factor=conf.learning_rate_decay_factor,
        learning_rate_decay_epoch=conf.learning_rate_decay_epoch,
        weight_decay=conf.weight_decay,
        target_accuracy=1.0,
        optimizer=optim.SGD,
        top_acc=1,
        criterion=nn.CrossEntropyLoss(),  # 损失函数默为交叉熵，多用于多分类问题
        requires_grad=True,
        scheduler_name='MultiStepLR',
        eta_min=0,
        paint_loss=False,
        #todo:tmp!!!
        data_parallel=False,
        save_at_each_step=False,
        gradient_clip_value=None,

):
    '''

    :param net: net to be trained
    :param net_name: name of the net
    :param exp_name: name of the experiment
    :param dataset_name: name of the dataset
    :param train_loader: data_loader for training. If not provided, a data_loader will be created based on dataset_name
    :param validation_loader: data_loader for validation. If not provided, a data_loader will be created based on dataset_name
    :param learning_rate: initial learning rate
    :param learning_rate_decay: boolean, if true, the learning rate will decay based on the params provided.
    :param learning_rate_decay_factor: float. learning_rate*=learning_rate_decay_factor, every time it decay.
    :param learning_rate_decay_epoch: list[int], the specific epoch that the learning rate will decay.
    :param num_epochs: max number of epochs for training
    :param batch_size:
    :param evaluate_step: how often will the net be tested on validation set. At least one test every epoch is guaranteed
    :param load_net: boolean, whether loading net from previous checkpoint. The newest checkpoint will be selected.
    :param test_net:boolean, if true, the net will be tested before training.
    :param root_path:
    :param checkpoint_path:
    :param momentum:
    :param num_workers:
    :param weight_decay:
    :param target_accuracy:float, the training will stop once the net reached target accuracy
    :param optimizer:
    :param top_acc: can be 1 or 5
    :param criterion： loss function
    :param requires_grad: list containing names of the modules that do not need to be trained
    :param scheduler_name
    :param eta_min: for CosineAnnealingLR
    :param save_at_each_step:save model and input data at each step so the bug can be reproduced
    :return:
    '''
    params = dict(locals())  # aquire all input params
    for k in list(params.keys()):
        params[k]=str(params[k])

    torch.autograd.set_detect_anomaly(True)
    success = True  # if the trained net reaches target accuracy
    # gpu or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using: ', end='')
    if torch.cuda.is_available():
        print(torch.cuda.device_count(), ' * ', end='')
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print(device)

    # prepare the data
    if dataset_name == 'imagenet':
        mean = conf.imagenet['mean']
        std = conf.imagenet['std']
        train_set_path = conf.imagenet['train_set_path']
        train_set_size = conf.imagenet['train_set_size']
        validation_set_path = conf.imagenet['validation_set_path']
        default_image_size = conf.imagenet['default_image_size']
    elif dataset_name == 'cifar10':
        train_set_size = conf.cifar10['train_set_size']
        mean = conf.cifar10['mean']
        std = conf.cifar10['std']
        train_set_path = conf.cifar10['dataset_path']
        validation_set_path = conf.cifar10['dataset_path']
        default_image_size = conf.cifar10['default_image_size']
    elif dataset_name == 'tiny_imagenet':
        train_set_size = conf.tiny_imagenet['train_set_size']
        mean = conf.tiny_imagenet['mean']
        std = conf.tiny_imagenet['std']
        train_set_path = conf.tiny_imagenet['train_set_path']
        validation_set_path = conf.tiny_imagenet['validation_set_path']
        default_image_size = conf.tiny_imagenet['default_image_size']
    elif dataset_name == 'cifar100':
        train_set_size = conf.cifar100['train_set_size']
        mean = conf.cifar100['mean']
        std = conf.cifar100['std']
        train_set_path = conf.cifar100['dataset_path']
        validation_set_path = conf.cifar100['dataset_path']
        default_image_size = conf.cifar100['default_image_size']
    if train_loader is None:
        train_loader= data_loader.create_train_loader(dataset_path=train_set_path,
                                                      default_image_size=default_image_size,
                                                      mean=mean,
                                                      std=std,
                                                      batch_size=batch_size,
                                                      num_workers=num_workers,
                                                      dataset_name=dataset_name)
    if validation_loader is None:
        validation_loader= data_loader.create_validation_loader(dataset_path=validation_set_path,
                                                                default_image_size=default_image_size,
                                                                mean=mean,
                                                                std=std,
                                                                batch_size=batch_size,
                                                                num_workers=num_workers,
                                                                dataset_name=dataset_name)


    # if checkpoint_path is None:
    exp_path=os.path.join(root_path,'model_saved',exp_name)
    checkpoint_path=os.path.join(exp_path,'checkpoint')
    tensorboard_path=os.path.join(exp_path,'tensorboard')
    crash_path=os.path.join(exp_path,'crash')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path,exist_ok=True)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)
    if not os.path.exists(crash_path):
        os.makedirs(crash_path, exist_ok=True)

    #get the latest checkpoint
    lists = os.listdir(checkpoint_path)
    file_new=checkpoint_path
    if len(lists)>0:
        lists.sort(key=lambda fn: os.path.getmtime(checkpoint_path + "/" + fn))  # 按时间排序
        file_new = os.path.join(checkpoint_path, lists[-1])  # 获取最新的文件保存到file_new

    sample_num=0
    if os.path.isfile(file_new):
        if load_net:
            checkpoint = torch.load(file_new)
            print('{} load net from previous checkpoint:{}'.format(datetime.now(),file_new))
            net=storage.restore_net(checkpoint,pretrained=True,data_parallel=data_parallel)
            sample_num = checkpoint['sample_num']

    #set up summary writer for tensorboard
    writer=SummaryWriter(log_dir=tensorboard_path,
                         purge_step=int(sample_num/batch_size))
    if dataset_name == 'imagenet'or dataset_name == 'tiny_imagenet':
        image=torch.zeros(2,3,224,224).to(device)
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        image=torch.zeros(2,3,32,32).to(device)

    # writer.add_graph(net, image)
    for k in params.keys():
        writer.add_text(tag=k,text_string=params[k])

    if test_net:
        print('{} test the net'.format(datetime.now()))                      #no previous checkpoint
        accuracy= evaluate.evaluate_net(net, validation_loader,
                                        save_net=True,
                                        checkpoint_path=checkpoint_path,
                                        sample_num=sample_num,
                                        target_accuracy=target_accuracy,
                                        dataset_name=dataset_name,
                                        top_acc=top_acc,
                                        net_name=net_name,
                                        exp_name=exp_name
                                        )

        if accuracy >= target_accuracy:
            print('{} net reached target accuracy.'.format(datetime.now()))
            return success

    #ensure the net will be evaluated despite the inappropriate evaluate_step
    if evaluate_step>math.ceil(train_set_size / batch_size)-1:
        evaluate_step= math.ceil(train_set_size / batch_size) - 1


    optimizer=prepare_optimizer(net, optimizer, momentum, learning_rate, weight_decay ,requires_grad)
    if learning_rate_decay:
        if scheduler_name =='MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=learning_rate_decay_epoch,
                                                 gamma=learning_rate_decay_factor,
                                                 last_epoch=ceil(sample_num/train_set_size))
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler=lr_scheduler.CosineAnnealingLR(optimizer,
                                                     num_epochs,
                                                     eta_min=eta_min,
                                                     last_epoch=ceil(sample_num/train_set_size))
    loss_list=[]
    acc_list=[]
    xaxis_loss=[]
    xaxis_acc=[]
    xaxis=0
    print("{} Start training ".format(datetime.now())+net_name+"...")
    # from network.modules import conv2d_with_mask_shortcut
    # add_forward_hook(net, module_class=conv2d_with_mask_shortcut)

    for epoch in range(math.floor(sample_num/train_set_size),num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
        net.train()
        # one epoch for one loop
        for step, data in enumerate(train_loader, 0):
            # if step==0 and epoch==0:      # debug code
            #     old_data=data             #use the same batch of data over and over again
            # data=old_data                 #the loss should decrease if the net is defined properly

            xaxis+=1
            if sample_num / train_set_size==epoch+1:               #one epoch of training finished
                accuracy= evaluate.evaluate_net(net, validation_loader,
                                                save_net=True,
                                                checkpoint_path=checkpoint_path,
                                                sample_num=sample_num,
                                                target_accuracy=target_accuracy,
                                                dataset_name=dataset_name,
                                                top_acc=top_acc,
                                                net_name=net_name,
                                                exp_name=exp_name)
                if accuracy>=target_accuracy:
                    print('{} net reached target accuracy.'.format(datetime.now()))
                    return success
                break

            # 准备数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            sample_num += int(images.shape[0])

            optimizer.zero_grad()
            # forward + backward
            net.train()
            outputs = net(images)
            loss = criterion(outputs, labels)
            #torch.sum(torch.argmax(outputs,dim=1) == labels)/float(batch_size) #code for debug in watches to calculate acc

            if net.mask_training_start_epoch<=net.current_epoch<=net.mask_training_stop_epoch:
                if (net.step_tracked==1 or net.step_tracked==0) and net.mask_updating is True:
                    fig=draw_masked_net(net)
                    writer.add_figure(tag='net structure',figure=fig,global_step=int(sample_num / batch_size))

                # target_block_mask=0.5
                # block_penalty=torch.zeros(1).to(net.extractor.network[0].weight.device)
                # last_conv_prune = True  # to push to the direction that two consecutive layers will not be pruned together
                # for name,mod in net.named_modules():
                #     if isinstance(mod,modules.block_with_mask_shortcut):
                #         mask_abs=mod.shortcut_mask.abs()
                #         # if mask_abs<=0.5 and last_conv_prune is False:
                #         #     target_block_mask=0.1
                #         #     last_conv_prune=True
                #         # else:
                #         #     target_block_mask=0.9
                #         #     last_conv_prune=False
                #         block_penalty=block_penalty+(target_block_mask-mask_abs).abs()
                # alpha=7.5e-3
                # if step==0:
                #     writer.add_text(tag='alpha',text_string=str(alpha))
                #     writer.add_text(tag='target_block_mask', text_string=str(target_block_mask))
                # weighted_block_penalty=alpha*block_penalty
                # writer.add_scalar(tag='weighted_block_penalty',
                #                   scalar_value=weighted_block_penalty,
                #                   global_step=int(sample_num / batch_size))
                # loss=loss+weighted_block_penalty

            if save_at_each_step:
                torch.save(images,os.path.join(crash_path,'images.pt'))
                torch.save(labels,os.path.join(crash_path,'labels.pt'))
                torch.save(net.state_dict(),os.path.join(crash_path,'net.pt'))
                torch.save(loss,os.path.join(crash_path,'loss.pt'))
                torch.save(outputs,os.path.join(crash_path,'outputs.pt'))

            loss.backward()



            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_value_(net.parameters(), gradient_clip_value)

            optimizer.step()
            loss_list+=[float(loss.detach())]
            xaxis_loss+=[xaxis]
            writer.add_scalar(tag='status/loss',
                              scalar_value=float(loss.detach()),
                              global_step=int(sample_num / batch_size))


            if step%20==0:
                print('{} loss is {}'.format(datetime.now(),float(loss.data)))

            if step % evaluate_step == 0 and step != 0:
                accuracy= evaluate.evaluate_net(net, validation_loader,
                                                save_net=True,
                                                checkpoint_path=checkpoint_path,
                                                sample_num=sample_num,
                                                target_accuracy=target_accuracy,
                                                dataset_name=dataset_name,
                                                top_acc=top_acc,
                                                net_name=net_name,
                                                exp_name=exp_name)

                if accuracy>=target_accuracy:
                    print('{} net reached target accuracy.'.format(datetime.now()))
                    return success
                accuracy =float(accuracy)

                acc_list+=[accuracy]
                xaxis_acc+=[xaxis]
                writer.add_scalar(tag='status/val_acc',
                                  scalar_value=accuracy,
                                  global_step=epoch)

                if paint_loss:
                    fig , ax1 =plt.subplots()
                    ax2=ax1.twinx()
                    ax1.plot(xaxis_loss,loss_list,'g')
                    ax2.plot(xaxis_acc,acc_list,'b')
                    ax1.set_xlabel('step')
                    ax1.set_ylabel('loss')
                    ax2.set_ylabel('accuracy')
                    plt.title(exp_name)
                    plt.savefig(os.path.join(root_path,'model_saved',exp_name,'train.png'))
                    plt.close()

                print('{} continue training'.format(datetime.now()))
        if learning_rate_decay:
            scheduler.step()
            print(optimizer.state_dict()['param_groups'][0]['lr'])

    print("{} Training finished. Saving net...".format(datetime.now()))
    flop_num= measure_flops.measure_model(net=net, dataset_name=dataset_name, print_flop=False)
    accuracy = evaluate.evaluate_net(net, validation_loader,
                                     save_net=True,
                                     checkpoint_path=checkpoint_path,
                                     sample_num=sample_num,
                                     target_accuracy=target_accuracy,
                                     dataset_name=dataset_name,
                                     top_acc=top_acc,
                                     net_name=net_name,
                                     exp_name=exp_name)
    accuracy=float(accuracy)
    checkpoint = {
                  'highest_accuracy': accuracy,
                  'state_dict': net.state_dict(),
                  'sample_num': sample_num,
                  'flop_num': flop_num}
    checkpoint.update(storage.get_net_information(net,dataset_name,net_name))
    torch.save(checkpoint, '%s/flop=%d,accuracy=%.5f.tar' % (checkpoint_path, flop_num, accuracy))
    print("{} net saved at sample num = {}".format(datetime.now(), sample_num))
    writer.close()
    return not success

def add_forward_hook(net,module_class=None,module_name=None):
    def hook(module, input, output):
        print(name_of_mod[module])
        # print('input:',input[0].shape)
        # print('output:',output.shape)
        # mean=input[0].mean(dim=(0,2,3))
        # var=input[0].var(dim=(0,2,3))
        # mod=copy.deepcopy(module)
        # nn.init.ones_(mod.weight)
        # nn.init.zeros_(mod.bias)
        # o=mod.forward(input[0])

        print('max input:',input[0].detach().cpu().numpy().max())
        print('max output:',output.detach().cpu().numpy().max())
        print()
    name_of_mod={}
    for name,mod in net.named_modules() :
        if module_class is not None and isinstance(mod,module_class) or\
                module_name is not None and module_name == name:
            handle=mod.register_forward_hook(hook)
            name_of_mod[mod]=name

def add_backward_hook(net,module_class=None,module_name=None):
    def hook(module, grad_input, grad_output):
        print(name_of_mod[module])
        # print('input:',input[0].shape)
        # print('output:',output.shape)
        print('max input:',grad_input[0].detach().cpu().numpy().max())
        print('max output:',grad_output[0].detach().cpu().numpy().max())
        raise Exception('got you')
    name_of_mod={}
    for name,mod in net.named_modules() :
        if module_class is not None and isinstance(mod,module_class) or\
                module_name is not None and module_name == name:
            handle=mod.register_backward_hook(hook)
            name_of_mod[mod]=name

def add_hook(net,module_class=None,module_name=None):
    def hook(grad):
        print(module_name,grad.abs().max(),grad.abs().mean())

    name_of_mod={}
    for name,mod in net.named_modules() :
        if module_class is not None and isinstance(mod,module_class) or\
                module_name is not None and module_name == name:
            handle=mod.weight.register_hook(hook)
            name_of_mod[mod]=name

def check_grad(net,step):
    print(step)
    net.print_mask()
    for name,mod in net.named_modules():
        from network.modules import conv2d_with_mask_shortcut
        if isinstance(mod,conv2d_with_mask_shortcut) or isinstance(mod,nn.Linear) or isinstance(mod,nn.Conv2d):
            grad=mod.weight.grad.detach().cpu().numpy()
            grad_no_zero=grad[grad!=0]
            print(name,grad_no_zero.mean())
    print()

# def show_feature_map(
#                     net,
#                     data_loader,
#                     layer_indexes,
#                     num_image_show=64
#                      ):
#     '''
#     show the feature converted feature maps of a cnn
#     :param net: full net net
#     :param data_loader: data_loader to load data
#     :param layer_indexes: list of indexes of conv layer whose feature maps will be extracted and showed
#     :param num_image_show: number of feature maps showed in one conv_layer. Supposed to be a square number
#     :return:
#     '''
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     sub_net=[]
#     conv_index = 0
#     ind_in_features=-1
#     j=0
#     for mod in net.features:
#         ind_in_features+=1
#         if isinstance(mod, torch.nn.modules.conv.Conv2d):
#             conv_index+=1
#             if conv_index in layer_indexes:
#                 sub_net.append(nn.Sequential(*list(net.children())[0][:ind_in_features+1]))
#                 j+=1
#
#     #sub_net = nn.Sequential(*list(net.children())[0][:conv_index+1])
#     for step, data in enumerate(data_loader, 0):
#         # 准备数据
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)
#         for i in range(len(layer_indexes)):
#             # forward
#             sub_net[i].eval()
#             outputs = sub_net[i](images)
#             outputs=outputs.detach().cpu().numpy()
#             outputs=outputs[0,:num_image_show,:,:]
#             outputs=pixel_transform(outputs)
#
#             #using pca to reduce the num of channels
#             image_dim_reduced=np.swapaxes(np.swapaxes(outputs,0,1),1,2)
#             shape=image_dim_reduced.shape
#             image_dim_reduced=np.resize(image_dim_reduced,(shape[0]*shape[1],shape[2]))
#             pca = PCA(n_components=40)#int(image_dim_reduced.shape[1]*0.5))  # 加载PCA算法，设置降维后主成分数目为32
#             image_dim_reduced = pca.fit_transform(image_dim_reduced)  # 对样本进行降维
#             image_dim_reduced=np.resize(image_dim_reduced,(shape[0],shape[1],image_dim_reduced.shape[1]))
#             image_dim_reduced = np.swapaxes(np.swapaxes(image_dim_reduced, 1, 2), 0, 1)
#             image_dim_reduced=pixel_transform(image_dim_reduced)
#             plt.figure(figsize=[14,20],clear=True,num=layer_indexes[i]+100)
#             for j in range(32):
#                 im=Image.fromarray(image_dim_reduced[j])
#                 plt.subplot(math.sqrt(num_image_show),math.sqrt(num_image_show),j+1)
#                 plt.imshow(im,cmap='Greys_r')
#
#
#             plt.figure(figsize=[14,20],clear=True,num=layer_indexes[i])
#             for j in range(num_image_show):
#                 im=Image.fromarray(outputs[j])
#                 plt.subplot(math.sqrt(num_image_show),math.sqrt(num_image_show),j+1)
#                 plt.imshow(im,cmap='Greys_r')
#         plt.show()
#         break

def pixel_transform(feature_maps):
    #把feature maps数值移至0-255区间
    mean = feature_maps.mean()
    transform = 255 / 2 - mean
    feature_maps = feature_maps + transform  # 把所有像素提至255的中点附近
    max = feature_maps.max()
    min = feature_maps.min()
    mean = feature_maps.mean()
    if max - mean > mean - min:
        ratio = (255 - mean) / (max - mean)
    else:
        ratio = mean / (mean - min)
    feature_maps = ratio * (feature_maps - mean) + mean  # 把像素划入0-255
    return feature_maps






if __name__ == "__main__":

    # save the output to log
    print('save log in:./log.txt')

    # sys.stdout = logger.Logger( '../data/log2.txt', sys.stdout)
    # sys.stderr = logger.Logger( '../data/log2.txt', sys.stderr)  # redirect std err, if necessary
    #
    # net= vgg.vgg16_bn(pretrained=False)
    #
    # # m1=nn.Linear(2048,4096)
    # # nn.init.normal_(m1.weight, 0, 0.01)
    # # nn.init.constant_(m1.bias, 0)
    # # net.classifier[0]=m1
    # #
    # # m3=nn.Linear(4096,200)
    # # nn.init.normal_(m3.weight, 0, 0.01)
    # # nn.init.constant_(m3.bias, 0)
    # # net.classifier[6]=m3
    #
    # # net.classifier = nn.Sequential(
    # #     nn.Dropout(),
    # #     nn.Linear(2048, 512),
    # #     nn.ReLU(True),
    # #     nn.Dropout(),
    # #     nn.Linear(512, 512),
    # #     nn.ReLU(True),
    # #     nn.Linear(512, 200),
    # # )
    # # for m in net.modules():
    # #     if isinstance(m, nn.Linear):
    # #         nn.init.normal_(m.weight, 0, 0.01)
    # #         nn.init.constant_(m.bias, 0)
    # net = net.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # # net=create_net.vgg_tiny_imagenet(net_name='vgg16_bn')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net.to(device)
    # # measure_flops.measure_model(net, dataset_name='cifar10')
    # batch_size=32
    # num_worker=8
    # train_loader= data_loader.create_train_loader(batch_size=batch_size,
    #                                               num_workers=num_worker,
    #                                               dataset_name='tiny_imagenet',
    #                                               default_image_size=224,
    #                                               )
    # validation_loader= data_loader.create_validation_loader(batch_size=batch_size,
    #                                                         num_workers=num_worker,
    #                                                         dataset_name='tiny_imagenet',
    #                                                         default_image_size=224
    #                                                         )
    # for i in range(10):
    #     print(i)
    #     train(net=net,
    #           net_name='vgg16bn_tiny_imagenet'+str(i),
    #           dataset_name='tiny_imagenet',
    #           test_net=False,
    #           # optimizer=optim.SGD,
    #           # learning_rate=0.1,
    #           # learning_rate_decay=True,
    #           # learning_rate_decay_epoch=[ 30, 60, 600],
    #           # learning_rate_decay_factor=0.1,
    #           # weight_decay=0.0006,
    #
    #           optimizer=optim.Adam,
    #           learning_rate=1e-3,
    #           weight_decay=1e-8,
    #           learning_rate_decay=False,
    #
    #           load_net=True,
    #           batch_size=batch_size,
    #           num_epochs=1000,
    #           train_loader=train_loader,
    #           validation_loader=validation_loader,
    #           evaluate_step=1000,
    #           )



    # checkpoint = torch.load(
    #     '/home/victorfang/Desktop/pytorch_model/vgg16bn_cifar10_dead_neural_normal_tar_acc_decent1/checkpoint/sample_num=11050000,accuracy=0.93370.tar')
    #
    # net = checkpoint['net']
    # net.load_state_dict(checkpoint['state_dict'])
    # print(checkpoint['highest_accuracy'])
    #
    # measure_flops.measure_model(net, dataset_name='cifar10')
    #
    # train(net=net,
    #       net_name='temp_train_a_net',
    #       dataset_name='cifar10',
    #       optimizer=optim.SGD,
    #       learning_rate=0.001,
    #       learning_rate_decay=True,
    #       learning_rate_decay_epoch=[50,100,150,250,300,350,400],
    #       learning_rate_decay_factor=0.5,
    #       test_net=False,
    #       load_net=False,
    #       target_accuracy=0.933958988332225,
    #       batch_size=600,
    #       num_epochs=450)





