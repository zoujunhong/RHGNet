import tensorflow.compat.v1 as tf

import cv2
import numpy as np
import torch

import os

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [240, 320]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def clevr_dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.
  Args:
    tfrecords_path: str. Path to the dataset file.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.
  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)

if not os.path.exists('./CLEVR_with_mask'):
  os.makedirs('./CLEVR_with_mask')
  os.makedirs('./CLEVR_with_mask/image')
  os.makedirs('./CLEVR_with_mask/object_mask')

tf_records_path = './clevr_with_masks_train.tfrecords'
batch_size = 1

dataset = clevr_dataset(tf_records_path)
batched_dataset = dataset.batch(batch_size)  # optional batching
iterator = batched_dataset.make_one_shot_iterator()



for i in range(100000):
    if i % 100 == 0:
      print('{}/100000'.format(i))
    data = iterator.get_next()
    with tf.Session():
        z_np = data['image'].numpy()
        cv2.imwrite('./CLEVR_with_mask/image/CLEVR_{}.jpg'.format(i), z_np.squeeze()[:,:,::-1])

        z_np = data['mask'].numpy()
        # size = data['size'].numpy()
        # material = data['material'].numpy()
        # shape = data['shape'].numpy()
        # color = data['color'].numpy()
        # print(z_np)
        # print(z_np.squeeze().shape)
        # print(np.max(z_np),np.min(z_np))
        z_np = np.argmax(z_np.squeeze(0),axis=0)
        # z_size = np.zeros_like(z_np)
        # z_mate = np.zeros_like(z_np)
        # z_shape = np.zeros_like(z_np)
        # z_color = np.zeros_like(z_np)
        # for j in range(MAX_NUM_ENTITIES):
        #   z_size[z_np == j] = size.squeeze()[j]
        #   z_mate[z_np == j] = material.squeeze()[j]
        #   z_shape[z_np == j] = shape.squeeze()[j]
        #   z_color[z_np == j] = color.squeeze()[j]
        
        # z_size[z_size==0] = 255
        # z_size -= 1
        # z_size[z_size==254]=255
        # z_mate[z_mate==0] = 255
        # z_mate -= 1
        # z_mate[z_mate==254]=255
        # z_shape[z_shape==0] = 255
        # z_shape -= 1
        # z_shape[z_shape==254]=255
        # z_color[z_color==0] = 255
        # z_color -= 1
        # z_color[z_color==254]=255

        # to_save = np.concatenate((z_size,z_mate,z_shape), axis=-1)
        cv2.imwrite('./CLEVR_with_mask/object_mask/CLEVR_{}.png'.format(i),z_np.squeeze())
        # cv2.imwrite('./CLEVR_with_mask/seg_mask/clevr_{}.png'.format(i),to_save)
        # cv2.imwrite('./CLEVR_with_mask/seg_mask/clevr_color_{}.png'.format(i),z_color)