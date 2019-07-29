
import numpy as np
import tensorflow as tf

from inputs import reshape_bboxes, extract_resized_crop_bboxes

def input_nodes(
  
  tfrecords, 

  # number of times to read the tfrecords
  num_epochs=None,

  # Data queue feeding the model
  batch_size=8,
  num_threads=2,

  capacity = 1000,

  # Global configuration
  cfg=None):

  with tf.name_scope('inputs'):

    # A producer to generate tfrecord file paths
    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=num_epochs,
      shuffle=False
    )

    # Construct a Reader to read examples from the tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Parse an Example to access the Features
    features = tf.parse_single_example(
      serialized_example,
      features = {
        'image/id' : tf.FixedLenFeature([], tf.string),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded'  : tf.FixedLenFeature([], tf.string),
        'image/height' : tf.FixedLenFeature([], tf.int64),
        'image/width' : tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/count' : tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/score' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label' : tf.VarLenFeature(dtype=tf.int64)
      }
    )

    # Read in a jpeg image
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)

    image_height = tf.cast(features['image/height'], tf.float32)
    image_width = tf.cast(features['image/width'], tf.float32)
    
    image_id = features['image/id']
    filename  = features['image/filename']

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
    
    num_bboxes = tf.cast(features['image/object/bbox/count'], tf.int32)
    no_bboxes = tf.equal(num_bboxes, 0)

    #scores = features['image/object/bbox/score'].values #tf.sparse_tensor_to_dense(features['image/object/bbox/score'].values)
    scores = tf.ones([num_bboxes])#tf.reshape(scores, [num_bboxes]) # 
    
    labels = features['image/object/bbox/label'].values
    labels = tf.reshape(labels, [num_bboxes])

    # computed the bbox coords to use for cropping and crop them out
    if not cfg.LOOSE_BBOX_CROP:
      crop_bboxes = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
      crop_bboxes = tf.transpose(crop_bboxes, [1, 0])
      params = [image, crop_bboxes, cfg.INPUT_SIZE]
      cropped_images = tf.py_func(extract_resized_crop_bboxes, params, [tf.uint8])[0]
    else:
      if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      crop_x1, crop_y1, crop_x2, crop_y2 = tf.py_func(reshape_bboxes, [xmin, ymin, xmax, ymax], [tf.float32, tf.float32, tf.float32, tf.float32])
      crop_bboxes = tf.transpose(tf.concat(axis=0, values=[
          tf.expand_dims(crop_y1, 0), 
          tf.expand_dims(crop_x1, 0), 
          tf.expand_dims(crop_y2, 0), 
          tf.expand_dims(crop_x2, 0)]), [1, 0])
      cropped_images = tf.image.crop_and_resize(tf.expand_dims(image, 0), crop_bboxes, tf.zeros([num_bboxes], dtype=tf.int32), crop_size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE], method="bilinear", extrapolation_value=0, name=None)
      
      crop_bboxes = tf.concat(axis=0, values=[tf.expand_dims(crop_x1, 0), tf.expand_dims(crop_y1, 0), tf.expand_dims(crop_x2, 0), tf.expand_dims(crop_y2, 0)])
      crop_bboxes = tf.transpose(crop_bboxes, [1,0])
    
    # Convert the pixel values to be in the range [0,1]
    if cropped_images.dtype != tf.float32:
      cropped_images = tf.image.convert_image_dtype(cropped_images, dtype=tf.float32)

    # Get the images in the range [-1, 1]
    cropped_images = tf.subtract(cropped_images, 0.5)
    cropped_images = tf.multiply(cropped_images, 2.0)

    # Set the shape of everything for the queue
    cropped_images.set_shape([None, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    image_ids = tf.tile([[image_id]], [num_bboxes, 1])
    image_ids.set_shape([None, 1])

    bboxes = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
    bboxes = tf.transpose(bboxes, [1, 0])
    bboxes.set_shape([None, 4])
    
    scores = tf.reshape(scores, [-1, 1])
    scores.set_shape([None, 1])

    labels = tf.reshape(labels, [-1, 1])
    labels.set_shape([None, 1])

    filenames = tf.tile([[filename]], [num_bboxes, 1])
    filenames.set_shape([None,1])

    # We need some book keeping data in order to map the detected keypoints back to image space
    image_height_widths = tf.tile([[image_height, image_width]], [num_bboxes, 1])
    image_height_widths.set_shape([None, 2])
    #crop_bboxes = tf.concat(0, [tf.expand_dims(crop_x1, 0), tf.expand_dims(crop_y1, 0), tf.expand_dims(crop_x2, 0), tf.expand_dims(crop_y2, 0)])
    #crop_bboxes = tf.transpose(crop_bboxes, [1,0])
    crop_bboxes.set_shape([None, 4])

    batched_images, batched_bboxes, batched_scores, batched_image_ids, batched_labels, batched_image_height_widths, batched_crop_bboxes, batched_filenames = tf.train.batch(
      [cropped_images, bboxes, scores, image_ids, labels, image_height_widths, crop_bboxes,filenames],
      batch_size=batch_size,
      num_threads=num_threads,
      capacity= capacity,
      enqueue_many=True
    )

  # return a batch of images and their labels
  return batched_images, batched_bboxes, batched_scores, batched_image_ids, batched_labels, batched_image_height_widths, batched_crop_bboxes, batched_filenames

