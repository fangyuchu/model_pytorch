import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import config as conf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    batch_size=500
    dataset_path = conf.cifar10['train_set_path']
    data_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=dataset_path, train=True, transform=transforms.Compose([
                transforms.ToTensor(),
            ])
         ),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
    stat = np.zeros(shape=[3, 256])
    x_axis=[0+i/256 for i in range(256)]

    for step, data in enumerate(data_loader, 0):
        print(step*batch_size)
        images, labels = data
        #a=transforms.ToPILImage(images)
        images=images.data.numpy()
        for i in range(3):
            for img in images:
                img_one_chanel=img[i].flatten()

                stat[i]+=np.histogram(img_one_chanel,bins=256,range=(0,1))[0]

                # plt.figure()
                # plt.title('channel' + str(i))
                # plt.bar(x_axis, stat[0], )
                # plt.xlabel('value')
                # plt.ylabel('frequency')
                # plt.show()
                #
                # print()

    for i in range(3):
        plt.figure()
        plt.title('channel '+str(i))
        plt.bar(x_axis, stat[i], )
        plt.xlabel('value')
        plt.ylabel('frequency')
        plt.show()


