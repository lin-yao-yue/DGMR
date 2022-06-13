from torch import device
from wandb.wandb_torch import torch

# from package包.python文件 import 函数或变量
from testTrain import dataLoad

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import shapely.geometry as sgeom
import Cython

import torchvision.transforms as transforms
from PIL import Image

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()


# 输入图片地址
# 返回tensor变量
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image


imageTensor = image_loader("E:\\DGMR\\trainResult1\\raw\\x\\batch0Time0.jpg")
# tensor转numpy
imageNumpy = imageTensor.numpy()

fig = plt.figure()
# 返回画布中的 axes（轴域）对象
ax = plt.axes()
# 关闭画布中的 x 轴和 y 轴
ax.set_axis_off()
# plt.close()  # Prevents extra axes being plotted below animation
'''
axes.imshow: 
    在2D常规栅格上显示图像或数据，
    将2D标量数据缩放到[0，1]范围
    再根据 cmap 指定的映射方式，将其渲染为伪彩色图像
field: 图像数据
vmin, vmax: 定义颜色图覆盖的数据范围
cmap: 颜色图实例或注册的颜色图名称
'''
img = ax.imshow(imageNumpy[0, 0, :, :], vmin=0, vmax=10, cmap="jet")

plt.show()

'''
之所以会出现图片亮度的差异，是因为：
保存时，将 tensor 转换为 PLT，即将 float 数值转变为 0-255 之间的数值，再转换为 .jpg
重新载入时，只加载了 0-255 数值的 tensor 就直接使用，没有再将其转换为原始的 float 型 tensor
'''

