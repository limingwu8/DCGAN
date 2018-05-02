import torch
import matplotlib.pyplot as plt
import numpy as np

class Opt(object):
    batch_size = 100
    ngpu = 1
    nz = 100
    ngf = 32
    ndf = 32
    imageSize = 64
    nc = 3
    lr = 0.0002
    betas = (0.5, 0.999)
    epoch = 50
    save_model = 1
    which_pc = 0    # 0: train on civs linux, 1: train on p219
    dataset_dir = '/home/liming/Documents/dataset/faces' if which_pc==0 else '/home/PNW/wu1114/Documents/dataset/faces'
    root = './DCGAN-anime/'
    results_dir = './DCGAN-anime/images/'
    checkpoint_dir = './DCGAN-anime/checkpoint/'
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}



def show_batch(batch):
    batch = batch.data.cpu().numpy()

    # for i in range(batch.shape[0]):
    #     img = np.squeeze(np.transpose(batch[i], (0, 2, 3, 1)))
    #     plt.figure()
    #     plt.imshow(img)
    # plt.show()

    img = np.squeeze(batch[0].transpose((1, 2, 0)))
    plt.figure()
    plt.imshow(img)
    plt.show()