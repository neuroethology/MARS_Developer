"""
Input pipeline for training the detector. 
"""

import numpy as np
import tensorflow as tf

def input_nodes(
  
  tfrecords, 

  max_num_bboxes,

  # number of times to read the tfrecords
  num_epochs=1,

  # Data queue feeding the model
  batch_size=32,
  num_threads=2,
  shuffle_batch = True,
  capacity = 1000,

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
        'image/object/bbox/count' : tf.FixedLenFeature([], tf.int64),
        'image/object/area' : tf.VarLenFeature(dtype=tf.float32) # the area of the object, in the original image space
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

    image = tf.image.resize_bilinear(tf.expand_dims(image, 0), [cfg.INPUT_SIZE, cfg.INPUT_SIZE], align_corners=False)
    image = tf.squeeze(image)

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

    areas = features['image/object/area'].values
    areas = tf.pad(areas, [[0, num_rows_to_pad]])
    areas.set_shape([max_num_bboxes])
    

    if shuffle_batch:
      images, batched_bboxes, batched_num_bboxes, batched_areas, image_ids = tf.train.shuffle_batch(
        [image, bboxes, num_bboxes, areas, image_id],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, 
        min_after_dequeue= 0, 
        seed = cfg.RANDOM_SEED,
      )

    else:
      images, batched_bboxes, batched_num_bboxes, batched_areas, image_ids = tf.train.batch(
        [image, bboxes, num_bboxes, areas, image_id],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity,
        enqueue_many=False
      )
    
    return images, batched_bboxes, batched_num_bboxes, batched_areas, image_ids
    
    
  
