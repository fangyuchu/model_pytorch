import torch

lstm=torch.nn.LSTM(input_size=27,hidden_size=2,num_layers=2,batch_first=True)
#todo:lstm 用在时序中时，batch size和sequence len分别是什么？
x=torch.randn(1,5,27)
h0=torch.randn(2,1,2)
c0=torch.randn(2,1,2)
output=lstm(x,(h0,c0))
print()