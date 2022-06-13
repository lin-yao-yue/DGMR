import numpy as np
import matplotlib.pyplot as plt
from train.drawFig import draw_save_loss
from train.makeFile import mkdir


def load_data():
    # 加载数据集 (506, 14)
    datafile = './trainResult/loss/gen_loss.txt'
    data = np.loadtxt(datafile, dtype=float)
    # 将原始数据Reshape

    return data


dis_loss = []
gen_loss = load_data()
epoch = len(gen_loss)
x = np.arange(epoch)
plt.figure(dpi=300, figsize=(50, 8))
plt.plot(x, gen_loss, linewidth=1, label='gen_loss')
# plt.plot(x, dis_loss, linewidth=1, label='dis_loss')
plt.legend(loc='upper right')
mkdir('./trainResult/pic')
plt.savefig('./trainResult/pic/loss.jpg')

