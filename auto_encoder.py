import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
import numpy as np

# 超参数
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

# 下载MNIST数据
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# 输出一个样本
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[2])
# plt.show()

# Dataloader
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


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


autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()



# original data (first row) for viewing
view_data = train_data.train_data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor) / 255.


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28 * 28))
        b_y = Variable(x.view(-1, 28 * 28))
        b_label = Variable(y)

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(Variable(view_data))
            #for i in range(N_TEST_IMG):

          





