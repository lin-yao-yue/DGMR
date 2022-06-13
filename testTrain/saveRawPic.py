import torch
import torch.nn.functional as F

from datasets import load_dataset
from dgmr import DGMR
from dgmr.losses import loss_hinge_disc, grid_cell_regularizer, loss_hinge_gen

from testTrain import dataLoad


from torchvision import transforms

# 加载数据集
dataLoadTool = dataLoad
x, y = dataLoadTool.dataLoad()
# (2, 4, 1, 256, 256)
x = torch.from_numpy(x)
y = torch.from_numpy(y)

# 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
toPIL = transforms.ToPILImage()
for batch in range(x.shape[0]):
    for time in range(x.shape[1]):
        pic = toPIL(x[batch][time])
        pic.save('E:/DGMR/trainResult/raw/x/batch%dTime%d.jpg' % (batch, time))

toPIL = transforms.ToPILImage()
for batch in range(y.shape[0]):
    for time in range(y.shape[1]):
        pic = toPIL(y[batch][time])
        pic.save('E:/DGMR/trainResult/raw/y/batch%dTime%d.jpg' % (batch, time))
