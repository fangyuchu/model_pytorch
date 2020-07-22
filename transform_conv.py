import torch
# import cv2
from network import vgg
from network.modules import conv2d_with_mask


def conv_to_matrix(conv):
    '''
    transform  4-d filters in conv to a matrix
    :param conv: conv module
    :return: 2-d tensor. each row is one filter.
    '''
    weight = conv.weight.detach()
    matrix = weight.view(weight.size(0), -1)
    return matrix

def pca(tensor_2d,dim):
    '''

    :param tensor_2d: each row is a piece of data
    :param dim:
    :return: tensor after dimension reduction,each row is a piece of data
    '''
    if dim>tensor_2d.shape[1]:
        raise Exception('Required dim is larger than feature len.(dim:{}>tensor_2d.shape[1]:{})'.format(dim,tensor_2d.shape[1]))
    u,s,v=torch.svd(tensor_2d)
    projection_matrix=v[:,:dim]
    return torch.matmul(tensor_2d,projection_matrix)

# def conv_dct(net):
#     '''
#     transform all conv into frequency matrix
#     :param net:
#     :return:a list containing frequency matrix for each conv layer
#     '''
#     frequency_matrix=[]
#     for mod in net.modules():
#         if isinstance(mod,torch.nn.Conv2d):
#             weight_matrix=conv_to_matrix(mod).cpu().numpy()
#             frequency_matrix+=[cv2.dct(weight_matrix)]
#     return frequency_matrix


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("using: "+torch.cuda.get_device_name(torch.cuda.current_device())+" of capacity: "+str(torch.cuda.get_device_capability(torch.cuda.current_device())))
#     '''选择网络模型'''
#     net= vgg.vgg16_bn(pretrained=False).to(device)
#     conv_dct(net)