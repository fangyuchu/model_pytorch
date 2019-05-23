import numpy as np

#training params
num_epochs=10                       #times for the use of all training data
batch_size=32                       #number of images for one batch
learning_rate=5e-5
learning_rate_decay_factor=0.93     #decay factor for learning rate decay
weight_decay=5e-4                   # weight decay (L2 penalty)
num_epochs_per_decay=2.5
dropout_rate=0.5
momentum=0.9

#dataset processing params
num_workers=5


#dataset params
#imagenet
imagenet=dict()
imagenet['num_class']=1001                                          #number of the classes
imagenet['label_offset']=1                                          #offset of the label
imagenet['mean']=[0.485, 0.456, 0.406]
imagenet['std']=[0.229, 0.224, 0.225]
imagenet['train_set_size']=1271167
imagenet['validation_set_size']=50000
imagenet['train_set_path']='/home/victorfang/Desktop/imagenet所有数据/imagenet_train'
imagenet['validation_set_path']='/home/victorfang/Desktop/imagenet所有数据/imagenet_validation'
imagenet['default_image_size']=224
#cifar10
cifar10=dict()
cifar10['num_class']=10
cifar10['train_set_size']=50000
cifar10['mean']=[0.485, 0.456, 0.406]
cifar10['std']=[0.229, 0.224, 0.225]
cifar10['train_set_path']='/home/victorfang/Desktop/cifar10'
cifar10['validation_set_path']='/home/victorfang/Desktop/cifar10'
cifar10['validation_set_size']=10000
cifar10['default_image_size']=32


#model saving params
#how often to write summary and checkpoint
checkpoint_step=4000

# Path for tf.summary.FileWriter and to store model checkpoints
root_path='/home/victorfang/Desktop/pytorch_model/'
checkpoint_path = "_model_saved/checkpoints"
highest_accuracy_path='_model_saved/accuracy.txt'
sample_num_path='_model_saved/sample_num.txt'
epoch_path='_model_saved/epoch.txt'