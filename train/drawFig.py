import matplotlib.pyplot as plt
import numpy as np
from train.makeFile import mkdir
import torch


def draw_save_loss(gen_loss, dis_loss, epochs, settled_batch=0):
    # epochs: 整数值，两个loss：list 数组
    x = np.arange(epochs-settled_batch)+settled_batch
    # list 转 tensor 转 numpy，转换要取消梯度
    gen_loss = torch.Tensor(gen_loss).detach().numpy()
    # dis_loss = torch.Tensor(dis_loss).detach().numpy()

    plt.figure()
    plt.plot(x, gen_loss, linewidth=1, label='gen_loss')
    # plt.plot(x, dis_loss, linewidth=1, label='dis_loss')
    plt.legend(loc='upper right')
    mkdir('./trainResult/pic/epoch%d' % epochs)
    plt.savefig('./trainResult/pic/epoch%d/loss.jpg' % epochs)





