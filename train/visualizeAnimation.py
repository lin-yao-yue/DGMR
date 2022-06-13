from torch import device
from wandb.wandb_torch import torch

from testTrain import dataLoad

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import shapely.geometry as sgeom
import Cython

import torchvision.transforms as transforms
from PIL import Image

matplotlib.rc('animation', html='jshtml')


def plot_animation(field, figsize=None,
                   vmin=0, vmax=10, cmap="jet", **imshow_args):
    # 创建一个画布 figure 指定 size，返回画布对象
    fig = plt.figure(figsize=figsize)
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
    img = ax.imshow(field[0, 0, ...], vmin=vmin, vmax=vmax, cmap=cmap, **imshow_args)
    '''
    print(field[0, ..., 0].shape)
    (256, 256)
    print(img)
    AxesImage(80, 52.8;496x369.6)
    '''
    # 动画的每一帧都会调用animate，当前帧为 i
    def animate(i):
        img.set_data(field[i, 0, ...])
        return (img,)

    '''
    a.fig 绘制动图的画布名称
    b.animate 自定义动画函数
    c.frames 动画长度，一次循环包含的帧数
    d.init_func 自定义开始帧，即传入刚定义的函数init,初始化函数
    e.interval 更新频率，以ms计，即每多少 ms 更新一帧，将当前帧送给 animate(n)
    f.blit 选择更新所有点，还是仅更新产生变化的点。应选择True，但mac用户请选择False，否则无法显
    '''
    return animation.FuncAnimation(
        fig, animate, frames=field.shape[0], interval=24, blit=False)


