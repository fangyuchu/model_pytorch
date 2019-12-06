import torch
import vgg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
co-author Zeng Yao
'''
ROOT = "Save_Load"
MODELNAME = "lenet5"
ModelName=None
lenet=True
AlexNet=False
Vgg=False

def conv_to_matrix(conv):
    '''
    transform  4-d filters in conv to a matrix
    :param conv: conv module
    :return: 2-d numpy array. each row is one filter.
    '''
    weight = conv.weight.data
    matrix = weight.view(weight.size(0), -1).cpu().numpy()
    return matrix

def conv_dct(net):
    '''
    transform all conv into frequency matrix
    :param net:
    :return:a list containing frequency matrix for each conv layer
    '''
    frequency_matrix=[]
    for mod in net.modules():
        if isinstance(mod,torch.nn.Conv2d):
            weight_matrix=conv_to_matrix(mod)
            frequency_matrix+=[cv2.dct(weight_matrix.T)]

    return frequency_matrix

def plotConvolution(num_conv):
    i = 0
    conv_spatial = None
    for mod in net.features:
        if isinstance(mod, torch.nn.modules.conv.Conv2d):
            i += 1
            if i == num_conv:
                conv_spatial = mod
                break
    spatial_weight=conv_spatial.weight.data
    c_out, c_in, h, w = spatial_weight.shape
    M = c_in * h * w
    N = c_out
    '''将tensor转化成二维矩阵,不可直接使用reshape'''
    print("Layer of CONV: " + str(num_conv) + ". Number of filters is: " + str(len(spatial_weight)))
    W_spatial_matrix=spatial_weight.view(spatial_weight.size(0),-1).t().cpu().numpy()


    '''2-D DCT 变换到频率域'''
    W_frequency = cv2.dct(W_spatial_matrix)
    '''2-D DCT 反变换到空间域'''
    W_idct = cv2.idct(W_frequency)
    for i in range(M):
        for j in range(N):
            if W_spatial_matrix[i, j] - W_idct[i, j] < 0.0001:
                continue
            else:
                print("in i=", i, "j=", j, "原来的矩阵为", W_spatial_matrix[i, j], "反变换后的矩阵为", W_idct[i, j])
    '''绘制频率域的三维柱状图'''
    print("plotting......")
    # plt.rcParams['savefig.dpi'] = 900  # 图片像素
    # plt.rcParams['figure.dpi'] = 900  # 分辨率
    plt.rcParams['figure.figsize'] = (15, 20)  # 长和高
    # plt.savefig("plot.png")
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    plt.title("Frequency-Domain of " + ModelName + " in layer CONV_" + str(num_conv))
    x = np.linspace(0, M - 1, M)  # 矩阵的列数
    y = np.linspace(0, N - 1, N)  # 矩阵的行数
    xpos, ypos = np.meshgrid(x, y)  # xpos, ypos都是N*M的网格
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)
    dx = 0.5 * np.ones_like(xpos)
    dy = 0.5 * N / M * np.ones_like(xpos)
    dz = np.abs(W_frequency).flatten()
    ax.bar3d(x=xpos, y=ypos, z=zpos, dx=dx, dy=dy, dz=dz,
             color='lightsteelblue')  # x,y为柱子在平面上的坐标，z为全0数组表示柱子起始高度，dx,dy为柱子的长宽，dz为柱子的高度
    plt.ylabel("Filter number="+str(c_out))
    plt.xlabel("c_in*d*d=" + str(c_in) + "*" + str(h) + "*" + str(w))
    '''绘制空间域的三维柱状图'''
    ax = fig.add_subplot(212, projection='3d')
    plt.title("Spatial-Domain of " + ModelName + " in layer CONV_" + str(num_conv))
    x = np.linspace(0, M - 1, M)  # 矩阵的列数
    y = np.linspace(0, N - 1, N)  # 矩阵的行数
    xpos, ypos = np.meshgrid(x, y)  # xpos, ypos都是N*M的网格
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    dx = 0.5 * np.ones_like(xpos)  # dx,dy指定了柱子的长度和宽度，dz指定了柱子的高度
    dy = 0.5 * N / M * np.ones_like(xpos)
    dz = np.abs(W_spatial_matrix).flatten()
    ax.bar3d(x=xpos, y=ypos, z=zpos, dx=dx, dy=dy, dz=dz, color='lightcyan')  # 绘制三维柱状图
    plt.ylabel("Filter number="+str(c_out))
    plt.xlabel("c_in*d*d=" + str(c_in) + "*" + str(h) + "*" + str(w))
    plt.savefig("./plotResult/Figure-of-" + ModelName + "-in-CONV-" + str(num_conv))  # 保存到指定路径,必须在plt.show()之前调用

    return plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using: "+torch.cuda.get_device_name(torch.cuda.current_device())+" of capacity: "+str(torch.cuda.get_device_capability(torch.cuda.current_device())))
    '''选择网络模型'''

    net=vgg.vgg16_bn(pretrained=True).to(device)
    ModelName = "VGG16"
    # conv_dct(net)
    # highest_accuracy,global_step=get_Accuracy_GlobalStep_fromFile(root=ROOT,modelName=MODELNAME,fileName="accuracy_baseline.txt")
    # net=get_Net_FromFile(net=net,root=ROOT,modelName=MODELNAME,check_point_dir="checkpoints_baseline",globalStep=global_step,sparsity=0)

    '''指定绘制的是第几层'''
    # for layer in range(8,13):
    #     plt = plotConvolution(layer)
        # plt.show()
    plt = plotConvolution(1)
    # plt.show()
# Process finished with exit code 137 (interrupted by signal 9: SIGKILL)



