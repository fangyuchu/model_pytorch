from filter_characteristic import filter_feature_extractor
from network import vgg, resnet,storage
from torch import nn
import torch
from prune.prune_module import get_module
from network.modules import conv2d_with_mask
import os
import copy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"





class predicted_mask_net(nn.Module):
    def __init__(self,net,net_name,dataset_name,feature_len=15,gcn_rounds=2,only_gcn=False,only_inner_features=False):
        super(predicted_mask_net, self).__init__()
        self.net=self.transform(net)#.to(device)
        self.extractor=filter_feature_extractor.extractor(feature_len=feature_len,gcn_rounds=gcn_rounds,only_gcn=only_gcn,only_inner_features=only_inner_features)
        self.net_name=net_name
        self.dataset_name=dataset_name
        self.feature_len=feature_len
        self.gcn_rounds=gcn_rounds
        self.data_parallel=True

    def train(self, mode=True):
        super().train(mode)
        if mode is False:
            self.update_mask()  # update the mask when being switched to eval mode
        return self

    def eval(self):
        return self.train(False)

    def copy(self):
        '''
        self-made deepcopy
        :return: a cloned network
        '''
        checkpoint = storage.get_net_information(self, self.dataset_name, self.net_name)
        copied_net = storage.restore_net(checkpoint, pretrained=True)
        copied_net.to(self.extractor.network[0].weight.device)
        return copied_net

    def transform(self,net):
        for name, mod in net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                device=mod.weight.device
                _modules = get_module(model=net, name=name)
                _modules[name.split('.')[-1]] = conv2d_with_mask(mod).to(device)  # replace conv
            if isinstance(mod,nn.BatchNorm2d):
                mod.track_running_stats=False                                     #track running stats for bn will cause huge shake when a filter is pruned
        return net

    def update_mask(self):
        mask = self.extractor(self, self.net_name, self.dataset_name)           #predict mask using extractor
        # print(mask.reshape(-1))
        lo = hi = 0
        for name, mod in self.net.named_modules():
            if isinstance(mod, nn.Conv2d) and 'downsample' not in name:
                hi += mod.out_channels
                _modules = get_module(model=self.net, name=name)
                mod.mask=mask[lo:hi]                   #update mask for each conv
                if torch.sum(mod.mask==0)==mod.out_channels:
                    raise Exception('all filters are pruned')
                lo = hi

    def forward(self, input):
        if self.training is True or self.data_parallel is True:
            self.update_mask()                                                      #mask only need to be updated when training.
        return self.net(input)


