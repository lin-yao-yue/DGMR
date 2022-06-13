import torch
import torch.nn.functional as F
from train.saveToFile import save_loss
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
batch_size_train = 4
batch_size_valid = 4

num_epochs = 40000
# 减小 context_channels 和 latent_channels 的大小，注意：后者是前者的两倍
model = DGMR(
    forecast_steps=18,
    input_channels=1,
    output_shape=256,
    latent_channels=256,
    context_channels=128,
    num_samples=batch_size_train,
)

# 将模型移到GPU上
model = model.to(device)

gen_loss = []
dis_loss = []

# 加载之前训练好的参数
# 150 epoch 两小时
'''
单训练gan：
400: 3小时 
500: 3小时50分钟
600: 5小时
700：8小时
850：10小时
20600结果不错
'''
settled_batch = 28350
save_pic_epochs = 28850
save_model_epochs = 28850
model.generator.load_state_dict(torch.load('./trainResult/model/epoch%d/Generator.pth' % settled_batch))
model.discriminator.load_state_dict(torch.load('./trainResult/model/epoch%d/Discriminator.pth' % settled_batch))


def train():
    for epoch in range(num_epochs):
        epoch = epoch+settled_batch

        # 每次random加载训练集

        # batch, time_step, channel, h , w
        train_x, train_y = dataLoad.load_small_random_train(batch_size_train)
        # numpy 转为 tensor
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        # 数据移到GPU上
        images = train_x.to(device)
        future_images = train_y.to(device)

        model.global_iteration += 1
        g_opt, d_opt = model.configure_optimizers()[0]
        ##########################
        # Optimize Discriminator #
        ##########################
        '''print("Epoch %d Optimize Discriminator:" % (epoch + 1))
        # 将 2 重设为 1
        for i in range(1):
            # 使用生成器生成图片，即并不是用随机矩阵作为输入，而是将要预测的图片放入生成器中
            predictions = model(images)
            # Cat along time dimension [B, T, C, H, W]
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)
            # Cat long batch for the real+generated
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

            # 辨别器的输入既包含真实的(x->y(x))，也包含Generator的(x->G(x))，
            concatenated_outputs = model.discriminator(concatenated_inputs)

            # D(y(x)), D(G(x))
            score_real, score_generated = torch.split(concatenated_outputs, 1, dim=1)
            discriminator_loss = loss_hinge_disc(score_generated, score_real)
            d_opt.zero_grad()
            discriminator_loss.backward()
            # model.manual_backward(discriminator_loss)
            d_opt.step()

        dis_loss.append(discriminator_loss)
        print('epoch %d, loss %f' % (epoch + 1, discriminator_loss))
        print("Epoch %d Optimize Discriminator End\n" % (epoch + 1))
'''
        ######################
        # Optimize Generator #
        ######################
        print("Epoch %d Optimize Generator:" % (epoch + 1))
        # 将 6 重设为 1
        predictions = [model(images) for _ in range(1)]
        # [B, T, C, H, W]
        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)
        # Concat along time dimension
        generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
        real_sequence = torch.cat([images, future_images], dim=1)
        # Cat long batch for the real+generated, for each example in the range
        # For each of the 6 examples
        generated_scores = []
        # 遍历每一组（x-->G(x)）
        for g_seq in generated_sequence:
            # [x->y(x), x->G(x)]
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            # [D(y(x)), D(G(x))]
            concatenated_outputs = model.discriminator(concatenated_inputs)
            # D(y(x)), D(G(x))
            score_real, score_generated = torch.split(concatenated_outputs, 1, dim=1)
            # [D(G(x1)),D(G(x2)).....D(G(x6))]
            generated_scores.append(score_generated)
        # 六个预测结果求平均
        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + model.grid_lambda * grid_cell_reg

        gen_loss.append(generator_loss)
        print('epoch %d, loss %f' % (epoch + 1, generator_loss))
        g_opt.zero_grad()
        generator_loss.backward()
        # model.manual_backward(generator_loss)
        g_opt.step()

        print("Epoch %d Optimize Generator End\n" % (epoch + 1))

        print("Epoch %d End\n" % (epoch + 1))

        # 保存训练结果
        if (epoch+1) % save_pic_epochs == 0:
            generated_images = model(images)
            # 保存当前 epoch 生成的图片
            save_generated_images(generated_images, epoch)
            # 保存原始图片
            raw_x_y = torch.cat((train_x, train_y), dim=1)
            save_raw_images(raw_x_y, epoch)

        # 保存训练模型，画出损失图像
        if (epoch+1) % save_model_epochs == 0:
            G = model.generator
            D = model.discriminator
            modelFile = "./trainResult/model/epoch%d" % (epoch + 1)
            mkdir(modelFile)
            torch.save(G.state_dict(), './trainResult/model/epoch%d/Generator.pth' % (epoch + 1))
            torch.save(D.state_dict(), './trainResult/model/epoch%d/Discriminator.pth' % (epoch + 1))
            print("模型保存完毕")
            # 保存该模型下训练的损失图像
            draw_save_loss(gen_loss, dis_loss, epoch+1, settled_batch)
            # 保存损失要提前创建 txt 文件,从 1500 开始保存
            save_loss("./trainResult/loss/gen_loss.txt", gen_loss)
            # save_loss("./trainResult/loss/dis_loss.txt", dis_loss)


if __name__ == "__main__":
    train()
    







