import torch
import train
import config as conf
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import math
import prune_and_train
import measure_flops
import evaluate
import numpy as np
import data_loader
from sklearn import svm
import vgg
import predict_dead_filter
from predict_dead_filter import fc
import prune
import generate_random_data
import resnet
import create_net
import matplotlib.pyplot as plt
import resnet_copied
from torch import optim

import torch
import gensim

net=vgg.vgg16_bn(pretrained=True)
m = nn.BatchNorm2d(100)

torch.manual_seed(2)

datas = [('你 叫 什么 名字 ?', 'n v n n f'), ('今天 天气 怎么样 ?', 'n n adj f'), ]
words = [data[0].split() for data in datas]
tags = [data[1].split() for data in datas]

id2word = gensim.corpora.Dictionary(words)
word2id = id2word.token2id

id2tag = gensim.corpora.Dictionary(tags)
tag2id = id2tag.token2id


def sen2id(inputs):
    return [word2id[word] for word in inputs]


def tags2id(inputs):
    return [tag2id[word] for word in inputs]


# print(sen2id('你 叫 什么 名字'.split()))
def formart_input(inputs):
    return torch.autograd.Variable(torch.LongTensor(sen2id(inputs)))


def formart_tag(inputs):
    return torch.autograd.Variable(torch.LongTensor(tags2id(inputs)), )


class LSTMTagger(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, voacb_size, target_size):
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.voacb_size = voacb_size
        self.target_size = target_size
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim)
        self.log_softmax = torch.nn.LogSoftmax()
        self.embedding = torch.nn.Embedding(self.voacb_size, self.embedding_dim)
        self.hidden = (torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                       torch.autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
        self.out2tag = torch.nn.Linear(self.hidden_dim, self.target_size)

    def forward(self, inputs):
        input = self.embedding((inputs))
        out, self.hidden = self.lstm(input.view(-1, 1, self.embedding_dim), self.hidden)
        tags = self.log_softmax(self.out2tag(out.view(-1, self.hidden_dim)))
        return tags


model = LSTMTagger(3, 3, len(word2id), len(tag2id))
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(100):
    model.zero_grad()
    input = formart_input('你 叫 什么 名字'.split())
    tags = formart_tag('n n adj f'.split())
    out = model(input)
    loss = loss_function(out, tags)
    loss.backward(retain_variables=True)
    optimizer.step()
    print(loss.data[0])
input = formart_input('你 叫 什么'.split())
out = model(input)
out = torch.max(out, 1)[1]
print([id2tag[out.data[i]] for i in range(0, out.size()[0])])





























device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_imagenet_prune_train/checkpoint/flop=12804890484,accuracy=0.90876.tar')
# checkpoint=torch.load('/home/victorfang/PycharmProjects/model_pytorch/model_saved/vgg16bn_tinyimagenet_prune_train/checkpoint/flop=11394264872,accuracy=0.71270.tar')
net=checkpoint['net']
# net=resnet_copied.resnet56()

#
# # checkpoint = torch.load('./baseline/resnet56_cifar10,accuracy=0.93280.tar')
# checkpoint=torch.load('./baseline/resnet56_cifar10,accuracy=0.94230.tar')
# net = resnet_copied.resnet56().to(device)

# # checkpoint=torch.load('/home/disk_new/model_saved/resnet56_cifar10_DeadNeural_realdata_good_baseline_过得去/代表/sample_num=13300000,accuracy=0.93610，flop=65931914.tar')
# # net=checkpoint['net']
#
net.load_state_dict(checkpoint['state_dict'])
print(checkpoint['highest_accuracy'])
train.train(net=net,
            net_name='vgg16bn_imagenet_prune_train2',
            dataset_name='imagenet',
            test_net=True,
            num_epochs=20,
            checkpoint_step=4000,
            target_accuracy=0.9140571220008835,
            batch_size=24,
            top_acc=5,


            optimizer=optim.SGD,
            learning_rate=0.0000001,
            # weight_decay=0.0006,
            momentum=0.9,
            learning_rate_decay=True,
            learning_rate_decay_epoch=[5, 10, 15],
            learning_rate_decay_factor=0.1,
            )