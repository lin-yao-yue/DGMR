import torch
import torch.nn.functional as F
from train.saveToFile import save_loss
from dgmr.dgmr import DGMR
from dgmr.losses import loss_hinge_disc, grid_cell_regularizer, loss_hinge_gen
import numpy as np
from train import dataLoad
from train.drawFig import draw_save_loss
from train.visualizeAnimation import plot_animation
from train.visualizePic import save_generated_images
from train.visualizePic import save_raw_images
import matplotlib.pyplot as plt
import os


settled_batch = 26500
batch_size_train = 4
batch_size_valid = 4

model = DGMR(
    forecast_steps=18,
    input_channels=1,
    output_shape=256,
    latent_channels=256,
    context_channels=128,
    num_samples=batch_size_train,
)


model.generator.load_state_dict(torch.load('./trainResult/model/epoch%d/Generator.pth' % settled_batch))
valid_x, valid_y = dataLoad.load_small_random_valid(batch_size_valid)
# numpy 转为 tensor
# batch, time_step, channel, h , w
images = torch.from_numpy(valid_x)
future_images = torch.from_numpy(valid_y)


def valid_visual_pic():
    generated_images = model(images)

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

    print(generator_loss)

    # 保存当前 epoch 生成的图片
    save_generated_images(generated_images, 0)
    # 保存原始图片
    raw_x_y = torch.cat((valid_x, valid_y), dim=1)
    save_raw_images(raw_x_y, 0)


def valid_visual_animation():
    raw = np.concatenate((valid_x, valid_y), axis=1)
    gen_images = model(images)
    pre = np.concatenate((valid_x, gen_images.detach().numpy()), axis=1)
    print(pre.shape)
    print(raw.shape)


    ani_raw = plot_animation(raw[0])
    ani_raw.save('E:/DGMR/visualizeRow.gif', writer='Pillow', fps=100)


valid_visual_animation()
