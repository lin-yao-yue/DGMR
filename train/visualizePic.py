import os
import matplotlib.pyplot as plt


# 创建文件夹
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


# 创建一个画布 figure
fig = plt.figure()
# 返回画布中的 axes（轴域）对象
ax = plt.axes()
# 关闭画布中的 x 轴和 y 轴
ax.set_axis_off()
# vmin, vmax: 定义颜色图覆盖的数据范围
'''
生成并保存原始的x，y图片

# 加载数据集
dataLoadTool = dataLoad
# tensor数据 (num_samples, 4, 1, 256, 256) (num_samples, 18, 1, 256, 256)
x, y = dataLoadTool.dataLoad()
# numpy 转为 tensor
x = torch.from_numpy(x)
y = torch.from_numpy(y)

for i in range(4):
    curX = x[0, i, 0, :, :]
    curX = curX.numpy()
    ax.imshow(curX, vmin=0, vmax=10, cmap="jet")
    # 保存figure
    plt.savefig("E:/DGMR/x/%d.jpg" % (i + 1))
for i in range(18):
    curY = y[0, i, 0, :, :]
    curY = curY.numpy()
    ax.imshow(curY, vmin=0, vmax=10, cmap="jet")
    # 保存figure
    plt.savefig("E:/DGMR/y/%d.jpg" % (i + 1))
'''


def save_generated_images(generated_images, epoch):
    for batch in range(generated_images.shape[0]):
        # 只保存前16个batch
        if batch >= 16:
            break
        for time in range(generated_images.shape[1]):
            image_2D = generated_images[batch, time, 0, :, :]
            # 取消当前梯度下进行保存图片操作
            numpy_image = image_2D.detach().numpy()
            ax.imshow(numpy_image, vmin=0, vmax=10, cmap="jet")
            # 创建文件夹
            file = "./trainResult/train/epoch%d/batch%d" % (epoch + 1, batch+1)
            mkdir(file)
            # 保存图片
            plt.savefig('./trainResult/train/epoch%d/batch%d/time%d.jpg' % (epoch + 1, batch+1, time+1))


# 保存原始图片
def save_raw_images(images, epoch):
    for batch in range(images.shape[0]):
        # 只保存前16个batch
        if batch >= 16:
            break
        for time in range(images.shape[1]):
            image_2D = images[batch, time, 0, :, :]
            # 取消当前梯度下进行保存图片操作
            numpy_image = image_2D.detach().numpy()
            ax.imshow(numpy_image, vmin=0, vmax=10, cmap="jet")
            # 创建文件夹
            file = "./trainResult/train/epoch%d/raw/batch%d" % (epoch + 1, batch+1)
            mkdir(file)
            # 保存图片
            plt.savefig('./trainResult/train/epoch%d/raw/batch%d/time%d.jpg' % (epoch + 1, batch+1, time+1))
