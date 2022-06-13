import torch
import torch.nn.functional as F

from dgmr.dgmr import DGMR
from dgmr.losses import loss_hinge_disc, grid_cell_regularizer, loss_hinge_gen
import numpy as np
from train import dataLoad
from train.drawFig import draw_save_loss
from train.visualizePic import save_generated_images
from train.visualizePic import save_raw_images

import os


# 创建文件夹
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 指定batch加载
batch_size_train = 16
train_x, train_y = dataLoad.load_small_random_train(batch_size_train)
# numpy 转为 tensor
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

images = torch.cat((train_x, train_y), dim=1)

save_raw_images(images)

