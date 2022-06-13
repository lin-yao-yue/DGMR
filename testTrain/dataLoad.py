import os

import numpy
import tensorflow as tf
from matplotlib import animation
import matplotlib
import matplotlib.pyplot as plt

# from train.run import NUM_TARGET_FRAMES, NUM_INPUT_FRAMES

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
def load_tfrecords(srcfile, split="train", variant="random_crops_256", shuffle_files=False):
    # Session 是 Tensorflow 为了控制,和输出文件的执行的语句.
    # 运行 session.run() 可以获得运算结果
    sess = tf.compat.v1.Session()

    # load tfrecord file
    dataset = tf.data.TFRecordDataset(srcfile)
    # parse data into tensor
    dataset = dataset.map(lambda row: parse_and_preprocess_row(row, split, variant))
    return dataset


def reader(srcfile="E:/DGMR/data/20200718/train/random_crops_256/seq-24-00000-of-02100.tfrecord",
           split="train",
           variant="random_crops_256",
           shuffle_files=False):
    return load_tfrecords(srcfile, split, variant, shuffle_files)


'''
validset = reader(
    srcfile="E:/DGMR/data/20200718/valid/subsampled_tiles_256_20min_stride/seq-24-00000-of-00033.tfrecord",
    split="valid",
    variant="subsampled_tiles_256_20min_stride")
row = next(iter(validset))
print(row["radar_frames"].numpy().shape)
(24, 256, 256, 1)
'''

dataset = load_tfrecords(srcfile="E:/DGMR/data/20200718/train/random_crops_256/seq-24-00000-of-02100.tfrecord")

"""
print(type(dataset))
<class 'tensorflow.python.data.ops.dataset_ops.MapDataset'>
"""

"""
# 读取训练集一行
row = next(iter(dataset))

print(row)
{
'end_time_timestamp': <tf.Tensor: shape=(), dtype=int64, numpy=1514725200>, 
'osgb_extent_bottom': <tf.Tensor: shape=(), dtype=int64, numpy=555000>, 
'osgb_extent_left': <tf.Tensor: shape=(), dtype=int64, numpy=-9000>, 
'osgb_extent_right': <tf.Tensor: shape=(), dtype=int64, numpy=247000>, 
'osgb_extent_top': <tf.Tensor: shape=(), dtype=int64, numpy=811000>, 
'sample_prob': <tf.Tensor: shape=(), dtype=float32, numpy=9.889281e-06>, 
'radar_frames': <tf.Tensor: shape=(24, 256, 256, 1), dtype=float32, numpy=array([[[[]]]]), dtype=float32)>,
'radar_mask': <tf.Tensor: shape=(24, 256, 256, 1), dtype=bool, numpy=array([[[[]]]])
}

print(row["radar_frames"].numpy().shape)
(24, 256, 256, 1)
"""

# 输入含有4个timeSteps，对应的输出将会有18个timeStep
NUM_INPUT_FRAMES = 4
NUM_TARGET_FRAMES = 18


def extract_input_and_target_frames(radar_frames):
    """Extract input and target frames from a dataset row's radar_frames."""
    # We align our targets to the end of the window, and inputs precede targets.
    input_frames = radar_frames[-NUM_TARGET_FRAMES - NUM_INPUT_FRAMES: -NUM_TARGET_FRAMES]
    target_frames = radar_frames[-NUM_TARGET_FRAMES:]
    return input_frames, target_frames


# iter(dataset)：获取可迭代对象的迭代器
# next()：返回迭代器指向的下一个项目，迭代器后移
row = next(iter(dataset))
input_frames, target_frames = extract_input_and_target_frames(row["radar_frames"])
"""
print(input_frames)
# <class 'tensorflow.python.framework.ops.EagerTensor'> (4, 256, 256, 1)
print(target_frames)
# <class 'tensorflow.python.framework.ops.EagerTensor'> (18, 256, 256, 1)
"""
num_samples = 1
# Add a batch dimension and tile along it to create a copy of the input for each sample:
input_frames = tf.expand_dims(input_frames, 0)
input_frames = tf.tile(input_frames, multiples=[num_samples, 1, 1, 1, 1])
target_frames = tf.expand_dims(target_frames, 0)
target_frames = tf.tile(target_frames, multiples=[num_samples, 1, 1, 1, 1])


# batch time h w channel
# print(input_frames.shape)

def horizontally_concatenate_batch_train(samples):
    n, t, h, w, c = samples.shape
    # N,T,H,W,C => N,T,C,H,W
    return tf.transpose(samples, [0, 1, 4, 2, 3])


input_frames_train = horizontally_concatenate_batch_train(input_frames)
target_frames_train = horizontally_concatenate_batch_train(target_frames)

# print(input_frames.shape)
# (1, 4, 1, 256, 256)
# print(target_frames.shape)
# (1, 18, 1, 256, 256)
# 由 NUM_INPUT_FRAMES 和 NUM_TARGET_FRAMES 确定
x = numpy.array(input_frames_train)
y = numpy.array(target_frames_train)


# print(type(x))
# print(type(y))
# <class 'numpy.ndarray'>

def dataLoad():
    return x, y
