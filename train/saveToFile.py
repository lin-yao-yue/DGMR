from train.makeFile import mkdir
import torch
import numpy as np


def save_loss(filename, data):  # filename为写入文件的路径，data为要写入数据列表.
    # 先取消梯度
    loss = torch.Tensor(data).detach().numpy()
    # 重新转换为 list
    loss = loss.tolist()
    file = open(filename, 'a')
    for i in range(len(loss)):
        s = str(loss[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功")
