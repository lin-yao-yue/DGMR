import os

import numpy as np
import tensorflow as tf
import random

_FEATURES = {name: tf.io.FixedLenFeature([], dtype)
             for name, dtype in [
                 ("radar", tf.string), ("sample_prob", tf.float32),
                 ("osgb_extent_top", tf.int64), ("osgb_extent_left", tf.int64),
                 ("osgb_extent_bottom", tf.int64), ("osgb_extent_right", tf.int64),
                 ("end_time_timestamp", tf.int64),
             ]}

_SHAPE_BY_SPLIT_VARIANT = {
    ("train", "random_crops_256"): (24, 256, 256, 1),
    ("valid", "subsampled_tiles_256_20min_stride"): (24, 256, 256, 1),
    ("test", "full_frame_20min_stride"): (24, 1536, 1280, 1),
    ("test", "subsampled_overlapping_padded_tiles_512_20min_stride"): (24, 512, 512, 1),
}

_MM_PER_HOUR_INCREMENT = 1 / 32.
_MAX_MM_PER_HOUR = 128.
_INT16_MASK_VALUE = -1


# dataset.map的格式
def parse_and_preprocess_row(row, split, variant):
    result = tf.io.parse_example(row, _FEATURES)
    shape = _SHAPE_BY_SPLIT_VARIANT[(split, variant)]
    radar_bytes = result.pop("radar")
    radar_int16 = tf.reshape(tf.io.decode_raw(radar_bytes, tf.int16), shape)
    mask = tf.not_equal(radar_int16, _INT16_MASK_VALUE)
    radar = tf.cast(radar_int16, tf.float32) * _MM_PER_HOUR_INCREMENT
    radar = tf.clip_by_value(
        radar, _INT16_MASK_VALUE * _MM_PER_HOUR_INCREMENT, _MAX_MM_PER_HOUR)
    result["radar_frames"] = radar
    result["radar_mask"] = mask
    return result


# 加载tfrecord数据
def load_tfrecords(srcfile, split, variant, shuffle_files):
    # Session 是 Tensorflow 为了控制,和输出文件的执行的语句.
    # 运行 session.run() 可以获得运算结果
    sess = tf.compat.v1.Session()

    # load tfrecord file
    dataset = tf.data.TFRecordDataset(srcfile)
    # parse data into tensor
    dataset = dataset.map(lambda row: parse_and_preprocess_row(row, split, variant))
    return dataset


def reader(srcfile="./data/train/random_crops_256/seq-24-00000-of-02100.tfrecord",
           split="train",
           variant="random_crops_256",
           shuffle_files=False):
    return load_tfrecords(srcfile, split, variant, shuffle_files)


train_dataset = reader(srcfile="./data/train/random_crops_256/seq-24-00000-of-02100.tfrecord",
                       split="train",
                       variant="random_crops_256",
                       shuffle_files=False)

valid_dataset = reader(srcfile="./data/valid/subsampled_tiles_256_20min_stride/seq-24-00000-of-00033.tfrecord",
                       split="valid",
                       variant="subsampled_tiles_256_20min_stride",
                       shuffle_files=False)

# 输入含有4个timeSteps，对应的输出将会有18个timeStep
NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES - NUM_INPUT_FRAMES: -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames


def horizontally_concatenate_batch_train(samples):
    n, t, h, w, c = samples.shape
    # N,T,H,W,C => N,T,C,H,W
    return tf.transpose(samples, [0, 1, 4, 2, 3])


def load_small(batch_size, data_set):
    input_frames = tf.convert_to_tensor(np.empty(shape=(1, 24, 256, 256, 1)), dtype=tf.float32)
    target_frames = tf.convert_to_tensor(np.empty(shape=(1, 24, 256, 256, 1)), dtype=tf.float32)
    # 要将迭代器单独提出来，否则循环过程中一直使用的是同一迭代位置的指针
    # 共包含 1458 个 batch
    itr = iter(data_set)
    for i in range(batch_size):
        row = next(itr)
        # 24, 256, 256, 1
        frame = row["radar_frames"]
        input_frame, target_frame = extract_input_and_target_frames(frame)
        # 1, 24, 256, 256, 1
        input_frame = tf.expand_dims(input_frame, 0)
        target_frame = tf.expand_dims(target_frame, 0)
        if i == 0:
            input_frames = input_frame
            target_frames = target_frame
        else:
            input_frames = tf.concat((input_frames, input_frame), 0)
            target_frames = tf.concat((target_frames, target_frame), 0)

    '''
    print(input_frames.shape)
    print(target_frames.shape)
    (batch_size, 4, 256, 256, 1)
    (batch_size, 18, 256, 256, 1)
    '''
    # 调整张量顺序
    input_frames_train = horizontally_concatenate_batch_train(input_frames)
    target_frames_train = horizontally_concatenate_batch_train(target_frames)
    x = np.array(input_frames_train)
    y = np.array(target_frames_train)
    '''
    print(x.shape)
    print(y.shape)
    (batch_size, 4, 1, 256, 256)
    (batch_size, 18, 1, 256, 256)
    '''
    return x, y


def load_all(data_set):
    input_frames = tf.convert_to_tensor(np.empty(shape=(1, 24, 256, 256, 1)), dtype=tf.float32)
    target_frames = tf.convert_to_tensor(np.empty(shape=(1, 24, 256, 256, 1)), dtype=tf.float32)
    itr = iter(data_set)
    i = 0
    while True:
        try:
            row = next(itr)
            # 24, 256, 256, 1
            frame = row["radar_frames"]
            input_frame, target_frame = extract_input_and_target_frames(frame)
            # 1, 24, 256, 256, 1
            input_frame = tf.expand_dims(input_frame, 0)
            target_frame = tf.expand_dims(target_frame, 0)
            if i == 0:
                input_frames = input_frame
                target_frames = target_frame
                i += 1
            else:
                input_frames = tf.concat((input_frames, input_frame), 0)
                target_frames = tf.concat((target_frames, target_frame), 0)
        except StopIteration:
            break

    input_frames_train = horizontally_concatenate_batch_train(input_frames)
    target_frames_train = horizontally_concatenate_batch_train(target_frames)
    x = np.array(input_frames_train)
    y = np.array(target_frames_train)
    return x, y


def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


def load_train_small(batch_size):
    return load_small(batch_size, train_dataset)


def load_train_all():
    return load_all(train_dataset)


def load_valid_small(batch_size):
    return load_small(batch_size, valid_dataset)


def load_valid_all():
    return load_all(valid_dataset)


'''
print(len(list(iter(train_dataset))))
1458
print(list(iter(train_dataset))[0]["radar_frames"].numpy().shape)
(24, 256, 256, 1)
'''
train_dataset_lis = list(iter(train_dataset))
train_dataset_size = len(train_dataset_lis)

valid_dataset_lis = list(iter(valid_dataset))
valid_dataset_size = len(valid_dataset_lis)


def load_small_random(batch_size, data_set_lis):
    input_frames = tf.convert_to_tensor(np.empty(shape=(1, 24, 256, 256, 1)), dtype=tf.float32)
    target_frames = tf.convert_to_tensor(np.empty(shape=(1, 24, 256, 256, 1)), dtype=tf.float32)
    # 随机数决定起始下标，闭区间
    begin_sub = random.randint(0, len(data_set_lis)-1)
    for i in range(batch_size):
        row = data_set_lis[(i+begin_sub) % len(data_set_lis)]
        # 24, 256, 256, 1
        frame = row["radar_frames"]
        input_frame, target_frame = extract_input_and_target_frames(frame)
        # 1, 24, 256, 256, 1
        input_frame = tf.expand_dims(input_frame, 0)
        target_frame = tf.expand_dims(target_frame, 0)
        if i == 0:
            input_frames = input_frame
            target_frames = target_frame
        else:
            input_frames = tf.concat((input_frames, input_frame), 0)
            target_frames = tf.concat((target_frames, target_frame), 0)
    # 调整张量顺序
    input_frames_train = horizontally_concatenate_batch_train(input_frames)
    target_frames_train = horizontally_concatenate_batch_train(target_frames)
    x = np.array(input_frames_train)
    y = np.array(target_frames_train)

    return x, y


def load_small_random_train(batch_size):
    return load_small_random(batch_size, train_dataset_lis)


def load_small_random_valid(batch_size):
    return load_small_random(batch_size, valid_dataset_lis)

