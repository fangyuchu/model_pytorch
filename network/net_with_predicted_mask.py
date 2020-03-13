from filter_characteristic import filter_feature_extractor
from network import vgg, resnet
from torch import nn
import torch
from prune.prune_module import get_module
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



class conv2d_with_mask(nn.modules.Conv2d):
    def __init__(self, conv):
        super(conv2d_with_mask, self).__init__(
            conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride,
            padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=(conv.bias is not None))
        self.weight=conv.weight
        if self.bias is not None:
            self.bias=conv.bias
        self.mask=None


    def forward(self,input):
        masked_weight=self.weight*self.mask.view(-1,1,1,1)
        if self.bias is None:
            masked_bias=None
        else:
            masked_bias=self.bias*self.mask
        #todo:考虑conv减了之后bn怎么办
        out=nn.functional.conv2d(input, masked_weight, masked_bias, self.stride,
                       self.padding, self.dilation, self.groups)

        return out

class predicted_mask_net(nn.Module):
    def __init__(self,net,net_name,dataset_name,feature_len=15,gcn_rounds=2,only_gcn=False,only_inner_features=False):
        super(predicted_mask_net, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net=self.transform(net).to(device)
        self.extractor=filter_feature_extractor.extractor(feature_len=feature_len,gcn_rounds=gcn_rounds,only_gcn=only_gcn,only_inner_features=only_inner_features).to(device)
        self.net_name=net_name
        self.dataset_name=dataset_name
        self.feature_len=feature_len
        self.gcn_rounds=gcn_rounds


    def transform(self,net):
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                device=mod.weight.device
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = conv2d_with_mask(mod).to(device)  # replace conv

        return net

    def update_mask(self):
        #todo:只有在训练的时候才需要不断更新
        print('a')

        mask = self.extractor(self.net, self.net_name, self.dataset_name)
        lo = hi = 0
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                hi += mod.out_channels
                _modules = get_module(model=self.net, name=name)
                _modules[name.split('.')[-1]].mask = mask[lo:hi]
                lo = hi

    def forward(self, input):
        self.update_mask()
        return self.net(input)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=resnet.resnet50(pretrained=True).to(device)
    tran_net=predicted_mask_net(net, net_name='resnet50', dataset_name='imagenet')
    tran_net=nn.DataParallel(tran_net)
    from framework import data_loader
    dl=data_loader.create_validation_loader(batch_size=2048, num_workers=4, dataset_name='imagenet', )
    from framework.evaluate import evaluate_net
    evaluate_net(tran_net, dl, save_net=False, net_name='resnet50')
    # criterion = nn.CrossEntropyLoss()
    # for step, data in enumerate(train_loader, 0):
    #     images, labels = data
    #     images, labels = images.to(device), labels.to(device)
    #     prediction=tran_net(images)
    #     loss = criterion(prediction, labels)
    #     loss.backward()
    #     print()