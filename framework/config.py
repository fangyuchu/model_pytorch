import numpy as np
import os,sys
def get_root_path():
    working_directory=os.path.abspath(sys.argv[0])
    dirs=working_directory.split('/')
    path=['/']
    for name in dirs:
        if name != 'model_pytorch':
            path+=[name]
        else:
            path += [name]
            break
    path+=['data']
    path=os.path.join(*path)
    return path

root_path=get_root_path()       #./model_pytorch/data




#training params
num_epochs=10                       #times for the use of all training data
batch_size=512                       #number of images for one batch
learning_rate=0.1
learning_rate_decay_factor=0.5     #decay factor for learning rate decay
weight_decay=5e-4                   # weight decay (L2 penalty)
num_epochs_per_decay=2.5
dropout_rate=0.5
momentum=0.9
learning_rate_decay_epoch=[]

#dataset processing params
num_workers=6


#dataset params
#imagenet
imagenet=dict()
imagenet['num_class']=1001                                          #number of the classes
imagenet['label_offset']=1                                          #offset of the label
imagenet['mean']=[0.485, 0.456, 0.406]
imagenet['std']=[0.229, 0.224, 0.225]
imagenet['train_set_size']=1271167
imagenet['validation_set_size']=50000
imagenet['train_set_path']=os.path.join(root_path,'dataset/imagenet/train')
imagenet['validation_set_path']=os.path.join(root_path,'dataset/imagenet/validation')
#(array([0.47068703, 0.44848716, 0.39994222], dtype=float32), array([0.28111452, 0.27503234, 0.28819305], dtype=float32))

imagenet['default_image_size']=224
#cifar10
cifar10=dict()
cifar10['num_class']=10
cifar10['train_set_size']=50000
cifar10['mean']=[x / 255 for x in [125.3, 123.0, 113.9]]
cifar10['std']=[x / 255 for x in [63.0, 62.1, 66.7]]
cifar10['dataset_path']=os.path.join(root_path,'dataset/cifar10')
cifar10['validation_set_size']=10000
cifar10['default_image_size']=32

#cifar100
cifar100=dict()
cifar100['num_class']=100
cifar100['train_set_size']=50000
cifar100['mean']=[0.5071, 0.4867, 0.4408]
cifar100['std']=[0.2675, 0.2565, 0.2761]
cifar100['dataset_path']=os.path.join(root_path,'dataset/cifar100')
cifar100['validation_set_size']=10000
cifar100['default_image_size']=32

#tiny_imagenet
tiny_imagenet=dict()
tiny_imagenet['num_class']=200
tiny_imagenet['train_set_size']=100000
tiny_imagenet['mean']=[0.485, 0.456, 0.406]
tiny_imagenet['std']=[0.229, 0.224, 0.225]
tiny_imagenet['train_set_path']=os.path.join(root_path,'dataset/tiny_imagenet/train')
tiny_imagenet['train+val_set_path']=os.path.join(root_path,'dataset/tiny_imagenet/train+val')
tiny_imagenet['validation_set_path']=os.path.join(root_path,'dataset/tiny_imagenet/val')
tiny_imagenet['validation_set_size']=10000
tiny_imagenet['default_image_size']=224

#model saving params
#how often to write summary and checkpoint
evaluate_step=4000


#cifar10
#learning configuration
#deep residual network
# weight_decay=1e-4
# momentum=0.9
# batch_size=128
# lr=0.1
# lr_decay_factor=0.1
# decay_epoch=[80,120]
#num_epoch=160

#soft filter pruning
# lr=0.1
# lr_decay_factor=0.1
# decay_epoch=[150,225]
# weight_decay=5e-4
# momentum=0.9
# batch_size=128
#num_epoch=300

#my schedule
# num_epochs=450,
# lr_decay_factor=0.5,
# decay_epch=[50,100,150,200,250,300,350,400]




