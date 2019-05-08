import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import vgg
from train import create_net


# 超参数
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

# 下载MNIST数据
# train_data = torchvision.datasets.MNIST(
#     root='./mnist/',
#     train=True,
#     transform=torchvision.transforms.ToTensor(),
#     download=DOWNLOAD_MNIST,
# )

# 输出一个样本
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[2])
# plt.show()

# Dataloader
#train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            #todo:之前激活函数是tanh，是否有必要改？
            #todo:自编码器的层数是否太多
            nn.Linear(3*3*512, 2304),
            nn.ReLU(),
            nn.Linear(2304, 576),
            nn.ReLU(),
            nn.Linear(576, 144),
            nn.ReLU(),
            nn.Linear(144, 18),
        )

        self.decoder = nn.Sequential(
            nn.Linear(18, 144),
            nn.ReLU(),
            nn.Linear(144, 576),
            nn.ReLU(),
            nn.Linear(576, 2304),
            nn.ReLU(),
            nn.Linear(2304,3 * 3 * 512),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train(x,y):
    auto_encoder = AutoEncoder()
    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR)
    loss_func = nn.MSELoss()






    # for epoch in range(EPOCH):
    #     for step, (x, y) in enumerate(train_loader):
    #
    #         encoded, decoded = auto_encoder(x)
    #
    #         loss = loss_func(decoded, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         print("{}loss is:{}".format(datetime.now(),loss))
    #
    #         if step % 100 == 0:
    #             print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
    #
    #             # plotting decoded image (second row)

                #for i in range(N_TEST_IMG):

          

model_urls = [

    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'vgg11_bn',
    'vgg13_bn',
    'vgg16_bn',
    'vgg19_bn'
]

root='/home/victorfang/PycharmProjects/model_pytorch/data/model_params/'

def download_weight():
    for net_name in model_urls:
        net=create_net(net_name,pretrained=True)
        for i in range(len(net.features)):
            mod=net.features[i]
            if isinstance(mod, torch.nn.modules.conv.Conv2d):
                weight=mod.weight.data.cpu().numpy()
                np.save(root+'weight/'+net_name+','+str(i),weight)
                bias=mod.bias.data.cpu().numpy()
                np.save(root+'bias/'+net_name+','+str(i),bias)

if __name__ == "__main__":
    download_weight()