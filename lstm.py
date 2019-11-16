from torch import nn
from torch.autograd import Variable
import torch
import numpy as np

class lstm(nn.Module):
    def __init__(self, input_size=9, hidden_size=4, output_size=1, num_layer=1):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer,batch_first=True)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, c_n) = self.layer1(x)
        b,s, h = x.size()
        _,batch,hidden_size=h_n.size()
        # x = x.view(s * b, h)
        h_n=h_n.view(batch,hidden_size)
        y = self.layer2(h_n)
        return y

if __name__ == "__main__":
    x=torch.from_numpy(np.ones([5,50,9])).float()#batch,seq_len,input_size
    model=lstm()
    y=model(x)
    print(y)
