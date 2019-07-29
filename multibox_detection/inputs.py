"""
Input pipeline for training the detector. 

Some of the augmentation code came from the inception examples
in the tensorflow repo, so I am including their license. 
"""

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from scipy.misc import imresize
import tensorflow as tf
import sys
from tensorflow.python.ops import control_flow_ops
import pdb

def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def distorted_bounding_box_crop(image, image_height, image_width,
  xmin, ymin, xmax, ymax, num_bboxes,
  min_object_covered = 0.7,
  aspect_ratio_range = (0.7, 1.4),
  area_range = (0.5, 1.0),
  max_attempts = 100,
  minimum_area = 50
):

  # combine the bounding boxes (the shape should be [bbox_coords, num_bboxes])
  bboxes = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
  # order the bboxes so that they have the shape: [num_bboxes, bbox_coords]
  bboxes = tf.transpose(bboxes, [1, 0])

  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
    tf.shape(image),
    bounding_boxes=tf.expand_dims(bboxes, 0),
    min_object_covered=min_object_covered,
    aspect_ratio_range=aspect_ratio_range,
    area_range=area_range,
    max_attempts=max_attempts,
    use_image_if_no_bounding_boxes=True
  )
  bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

  # Crop the image to the specified bounding box.
  cropped_image = tf.slice(image, bbox_begin, bbox_size)

  # offset the ground truth bounding boxes
  bbox_begin = tf.cast(tf.slice(bbox_begin, [0], [2]), tf.float32)
  bbox_size = tf.cast(tf.slice(bbox_size, [0], [2]), tf.float32)

  image_width = tf.cast(image_width, tf.float32)
  image_height = tf.cast(image_height, tf.float32)

  cropped_bbox_ymin = tf.slice(bbox_begin, [0], [1])
  cropped_bbox_xmin = tf.slice(bbox_begin, [1], [1])
  bbox_end = bbox_begin + bbox_size
  cropped_bbox_ymax = tf.slice(bbox_end, [0], [1])
  cropped_bbox_xmax = tf.slice(bbox_end, [1], [1])

  scaled_ymin = ymin * image_height
  scaled_ymax = ymax * image_height
  scaled_xmin = xmin * image_width
  scaled_xmax = xmax * image_width

  scaled_ymin = tf.maximum(scaled_ymin, tf.tile(cropped_bbox_ymin, [num_bboxes])) - cropped_bbox_ymin
  scaled_xmin = tf.maximum(scaled_xmin, tf.tile(cropped_bbox_xmin, [num_bboxes])) - cropped_bbox_xmin
  scaled_ymax = tf.minimum(scaled_ymax, tf.tile(cropped_bbox_ymax, [num_bboxes])) - cropped_bbox_ymin
  scaled_xmax = tf.minimum(scaled_xmax, tf.tile(cropped_bbox_xmax, [num_bboxes])) - cropped_bbox_xmin

  scaled_ymin = tf.clip_by_value(scaled_ymin, 0.0, image_height)
  scaled_xmin = tf.clip_by_value(scaled_xmin, 0.0, image_width)
  scaled_ymax = tf.clip_by_value(scaled_ymax, 0.0, image_height)
  scaled_xmax = tf.clip_by_value(scaled_xmax, 0.0, image_width)

  scaled_areas = (scaled_xmax - scaled_xmin) * (scaled_ymax - scaled_ymin)
  valid_areas = tf.cast(scaled_areas > minimum_area, tf.int32)

  num_bboxes = tf.reduce_sum(valid_areas)

  _, scaled_ymin = tf.dynamic_partition(scaled_ymin, valid_areas, 2)
  _, scaled_xmin = tf.dynamic_partition(scaled_xmin, valid_areas, 2)
  _, scaled_ymax = tf.dynamic_partition(scaled_ymax, valid_areas, 2)
  _, scaled_xmax = tf.dynamic_partition(scaled_xmax, valid_areas, 2)

  scaled_ymin = tf.expand_dims(scaled_ymin, 0)
  scaled_xmin = tf.expand_dims(scaled_xmin, 0)
  scaled_ymax = tf.expand_dims(scaled_ymax, 0)
  scaled_xmax = tf.expand_dims(scaled_xmax, 0)

  bbox_height = tf.cast(tf.slice(bbox_size, [0], [1]), tf.float32)
  bbox_width = tf.cast(tf.slice(bbox_size, [1], [1]), tf.float32)

  xmin = scaled_xmin / bbox_width
  ymin = scaled_ymin / bbox_height
  xmax = scaled_xmax / bbox_width
  ymax = scaled_ymax / bbox_height

  return  tf.tuple([cropped_image, xmin, ymin, xmax, ymax, num_bboxes])

def distorted_shifted_bounding_box(xmin, ymin, xmax, ymax, num_bboxes, image_height, image_width, max_num_pixels_to_shift = 5):

  image_width = tf.cast(image_width, tf.float32)
  image_height = tf.cast(image_height, tf.float32)
  one_pixel_width = 1. / image_width
  one_pixel_height = 1. / image_height
  max_width_shift = one_pixel_width * max_num_pixels_to_shift
  max_height_shift = one_pixel_height * max_num_pixels_to_shift

  xmin -= tf.random_uniform([1, num_bboxes], minval=0, maxval=max_width_shift, dtype=tf.float32)
  xmax += tf.random_uniform([1, num_bboxes], minval=0, maxval=max_width_shift, dtype=tf.float32)
  ymin -= tf.random_uniform([1, num_bboxes], minval=0, maxval=max_height_shift, dtype=tf.float32)
  ymax += tf.random_uniform([1, num_bboxes], minval=0, maxval=max_height_shift, dtype=tf.float32)

  # ensure that the coordinates are still valid
  ymin = tf.clip_by_value(ymin, 0.0, 1.)
  xmin = tf.clip_by_value(xmin, 0.0, 1.)
  ymax = tf.clip_by_value(ymax, 0.0, 1.)
  xmax = tf.clip_by_value(xmax, 0.0, 1.)

  return tf.tuple([xmin, ymin, xmax, ymax])

def input_nodes(

  tfrecords,

  max_num_bboxes,

  # number of times to read the tfrecords
  num_epochs=None,

  # Data queue feeding the model
  batch_size=32,
  num_threads=2,
  shuffle_batch = True,
  capacity = 1000,
  min_after_dequeue = 96,

  # And tensorboard summaries of the images
  add_summaries=True,

  # Global configuration
  cfg=None):

  with tf.name_scope('inputs'):

    # A producer to generate tfrecord file paths
    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=num_epochs
    )

    # Construct a Reader to read examples from the tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Parse an Example to access the Features
    features = tf.parse_single_example(
      serialized_example,
      features = {
        'image/id' : tf.FixedLenFeature([], tf.string),
        'image/encoded'  : tf.FixedLenFeature([], tf.string),
        'image/height' : tf.FixedLenFeature([], tf.int64),
        'image/width' : tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/count' : tf.FixedLenFeature([], tf.int64)
      }
    )

    # Read in a jpeg image
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    
    # Convert the pixel values to be in the range [0,1]
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_height = features['image/height']
    image_width = features['image/width']
    
    image_id = features['image/id']

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
    
    num_bboxes = tf.cast(features['image/object/bbox/count'], tf.int32)
    no_bboxes = tf.equal(num_bboxes, 0)

    # Add a summary of the original data
    if add_summaries:
      bboxes_to_draw = tf.cond(no_bboxes, lambda:  tf.constant([[0, 0, 1, 1]], tf.float32), lambda: tf.transpose(tf.concat(axis=0, values=[ymin, xmin, ymax, xmax]), [1, 0]))
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bboxes_to_draw)
      tf.summary.image('original_image', image_with_bboxes)

    # Perturb the bounding box coordinates
    r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    do_perturb = tf.logical_and(tf.less(r, cfg.DO_RANDOM_BBOX_SHIFT), tf.greater(num_bboxes, 0))
    xmin, ymin, xmax, ymax = tf.cond(do_perturb,
      lambda: distorted_shifted_bounding_box(xmin, ymin, xmax, ymax, num_bboxes, image_height, image_width, cfg.RANDOM_BBOX_SHIFT_EXTENT),
      lambda: tf.tuple([xmin, ymin, xmax, ymax])
    ) 

    # Take a crop from the image
    r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    do_crop = tf.less(r, cfg.DO_RANDOM_CROP)
    cropped_image, xmin, ymin, xmax, ymax, num_bboxes = tf.cond(do_crop,
      lambda: distorted_bounding_box_crop(image, image_height, image_width, xmin, ymin, xmax, ymax, num_bboxes,
        min_object_covered = cfg.RANDOM_CROP_MIN_OBJECT_COVERED,
        aspect_ratio_range = cfg.RANDOM_CROP_ASPECT_RATIO_RANGE,
        area_range = cfg.RANDOM_CROP_AREA_RANGE,
        max_attempts = cfg.RANDOM_CROP_MAX_ATTEMPTS,
        minimum_area= cfg.RANDOM_CROP_MINIMUM_AREA
      ),
      lambda: tf.tuple([image, xmin, ymin, xmax, ymax, num_bboxes])
    )
    cropped_image.set_shape([None, None, 3])

    # Resize the image 
    num_resize_cases = 4
    resized_image = apply_with_random_selector(
      cropped_image,
      lambda x, method: tf.image.resize_images(x, size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE], method=method),
      num_cases=num_resize_cases
    )

    # Add a summary 
    if add_summaries:
      bboxes_to_draw = tf.cond(no_bboxes, lambda:  tf.constant([[0, 0, 1, 1]], tf.float32), lambda: tf.transpose(tf.concat(axis=0, values=[ymin, xmin, ymax, xmax]), [1, 0]))
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(resized_image, 0), bboxes_to_draw)
      tf.summary.image('cropped_resized_image', image_with_bboxes)

    # Distort the colors
    r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
    do_color_distortion = tf.less(r, cfg.DO_COLOR_DISTORTION)
    num_color_cases = 1 if cfg.COLOR_DISTORT_FAST else 4
    distorted_image = apply_with_random_selector(
      resized_image,
      lambda x, ordering: distort_color(x, ordering, fast_mode=cfg.COLOR_DISTORT_FAST),
      num_cases=num_color_cases)
    image = tf.cond(do_color_distortion, lambda: tf.identity(distorted_image), lambda: tf.identity(resized_image))
    image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])

    # Randomly flip the image:
    if cfg.DO_RANDOM_FLIP_LEFT_RIGHT:
      r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
      do_flip = tf.less(r, 0.5)
      image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))
      xmin, xmax = tf.cond(do_flip, lambda: tf.tuple([1. - xmax, 1. - xmin]), lambda: tf.tuple([xmin, xmax])) 

    # Add a summary
    if add_summaries:
      bboxes_to_draw = tf.cond(no_bboxes, lambda:  tf.constant([[0, 0, 1, 1]], tf.float32), lambda: tf.transpose(tf.concat(axis=0, values=[ymin, xmin, ymax, xmax]), [1, 0]))
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bboxes_to_draw)
      tf.summary.image('final_distorted_image', image_with_bboxes)

    # combine the bounding boxes (the shape should be [bbox_coords, num_bboxes])
    bboxes = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
    # order the bboxes so that they have the shape: [num_bboxes, bbox_coords]
    bboxes = tf.transpose(bboxes, [1, 0])
    
    # pad the number of boxes so that all images have `max_num_bboxes`
    num_rows_to_pad = tf.maximum(0, max_num_bboxes - num_bboxes)   
    bboxes = tf.cond(no_bboxes, lambda: tf.zeros([max_num_bboxes, 4]), lambda: tf.pad(bboxes, tf.stack([tf.stack([0, num_rows_to_pad]), [0, 0]])))
    bboxes.set_shape([max_num_bboxes, 4])

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    if shuffle_batch:
      images, batched_bboxes, batched_num_bboxes, image_ids = tf.train.shuffle_batch(
        [image, bboxes, num_bboxes, image_id],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue= min_after_dequeue, # 3 * batch_size,
        seed = cfg.RANDOM_SEED,
      )

    else:
      images, batched_bboxes, batched_num_bboxes, image_ids = tf.train.batch(
        [image, bboxes, num_bboxes, image_id],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        enqueue_many=False
      )
    
    return images, batched_bboxes, batched_num_bboxes, image_ids
    
    
  
