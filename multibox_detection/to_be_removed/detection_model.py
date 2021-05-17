# detection_model.py

import tensorflow as tf
import tensorflow.keras as k


def add_detection_heads(inputs, num_bboxes_per_cell):
  endpoints = {}
  
  with tf.name_scope('Multibox'):

    # 8 x 8 grid cells
    with tf.name_scope("8x8"):
      """ TODO: how to handle training variables? """
      # 8 x 8 x 2048 
      branch8x8 = k.layers.Conv2D(96, [1, 1], padding='same')(inputs)
      # 8 x 8 x 96
      branch8x8 = k.layers.Conv2D(96, [3, 3], padding='same')(branch8x8)
      # 8 x 8 x 96
      endpoints['8x8_locations'] = k.layers.Conv2D(num_bboxes_per_cell * 4, [1, 1], padding='same')(branch8x8)
      # 8 x 8 x 96
      endpoints['8x8_confidences'] = k.layers.Conv2D(num_bboxes_per_cell, [1, 1], padding='same')(branch8x8)

    # 6 x 6 grid cells
    with tf.name_scope("6x6"):
      """ TODO: how to handle training variables """
      # 8 x 8 x 2048 
      branch6x6 = k.layers.Conv2D(96, [3, 3], padding='same')(inputs)
      # 8 x 8 x 96
      branch6x6 = k.layers.Conv2D(96, [3, 3], padding='valid')(branch6x6)
      # 6 x 6 x 96
      endpoints['6x6_locations'] = k.layers.Conv2D(num_bboxes_per_cell * 4, [1, 1],padding='same')(branch6x6)
      # 6 x 6 x 96
      endpoints['6x6_confidences'] = k.layers.Conv2D(num_bboxes_per_cell, [1, 1],padding='same')(branch6x6)
    
    # 8 x 8 x 2048
    net = k.layers.Conv2D(256, [3, 3], padding='same', stride=[2,2])(inputs)

    # 4 x 4 grid cells
    with tf.name_scope("4x4"):
      """ TODO: how to handle training variables """
      # 4 x 4 x 256
      branch4x4 = k.layers.Conv2D(128, [3, 3], padding='same')(net)
      # 4 x 4 x 128
      endpoints['4x4_locations'] = k.layers.Conv2D(num_bboxes_per_cell * 4, [1, 1], padding='same')(branch4x4)
      # 4 x 4 x 128
      endpoints['4x4_confidences'] = k.layers.Conv2D(num_bboxes_per_cell, [1, 1], padding='same')(branch4x4)

    # 3 x 3 grid cells
    with tf.name_scope("3x3"):
      """ TODO: how to handle training variables """
      # 4 x 4 x 256
      branch3x3 = k.layers.Conv2D(128, [1, 1], padding='same')(net)
      # 4 x 4 x 128
      branch3x3 = k.layers.Conv2D(96, [2, 2], padding='valid')(branch3x3)
      # 3 x 3 x 96
      endpoints['3x3_locations'] = k.layers.Conv2D(num_bboxes_per_cell * 4, [1, 1], padding='same')(branch3x3)
      # 3 x 3 x 96
      endpoints['3x3_confidences'] = k.layers.Conv2D(num_bboxes_per_cell, [1, 1], padding='same')(branch3x3)
      
    # 2 x 2 grid cells
    with tf.name_scope("2x2"):
      """ TODO: how to handle training variables """
      # 4 x 4 x 256
      branch2x2 = k.layers.Conv2D(128, [1, 1], padding='same')(net)
      # 4 x 4 x 128
      branch2x2 = k.layers.Conv2D(96, [3, 3], padding='valid')(branch2x2)
      # 2 x 2 x 96
      endpoints['2x2_locations'] = k.layers.Conv2D(num_bboxes_per_cell * 4, [1, 1], padding='same')(branch2x2)
      # 2 x 2 x 96
      endpoints['2x2_confidences'] = k.layers.Conv2D(num_bboxes_per_cell, [1, 1], padding='same')(branch2x2)
      
    # 1 x 1 grid cell
    with tf.name_scope("1x1"):
      """ TODO: how to handle training variables """
      # 8 x 8 x 2048
      branch1x1 = k.layers.AveragePooling2d(pool_size=(8, 8), strides=(1,1), padding='valid')(inputs)
      # 1 x 1 x 2048
      endpoints['1x1_locations'] = k.layers.Conv2D(4, [1, 1], padding='same')(branch1x1)
      # 1 x 1 x 2048
      selfendpoints['1x1_confidences'] = slim.Conv2D(1, [1, 1], padding='same')(branch1x1)
    
    batch_size = tf.shape(inputs)[0]#inputs.get_shape().as_list()[0]

    # reshape the locations and confidences for easy concatenation
    detect_8_locations = tf.reshape(endpoints['8x8_locations'], [batch_size, -1])
    detect_8_confidences = tf.reshape(endpoints['8x8_confidences'], [batch_size, -1])

    detect_6_locations = tf.reshape(endpoints['6x6_locations'], [batch_size, -1])
    detect_6_confidences = tf.reshape(endpoints['6x6_confidences'], [batch_size, -1])

    detect_4_locations = tf.reshape(endpoints['4x4_locations'], [batch_size, -1])
    detect_4_confidences = tf.reshape(endpoints['4x4_confidences'], [batch_size, -1])

    detect_3_locations = tf.reshape(endpoints['3x3_locations'], [batch_size, -1])
    detect_3_confidences = tf.reshape(endpoints['3x3_confidences'], [batch_size, -1])

    detect_2_locations = tf.reshape(endpoints['2x2_locations'], [batch_size, -1])
    detect_2_confidences = tf.reshape(endpoints['2x2_confidences'], [batch_size, -1])

    detect_1_locations = tf.reshape(endpoints['1x1_locations'], [batch_size, -1])
    detect_1_confidences = tf.reshape(endpoints['1x1_confidences'], [batch_size, -1])    
          
    # Collect all of the locations and confidences 
    locations = tf.concat(axis=1, values=[detect_8_locations, detect_6_locations, detect_4_locations, detect_3_locations, detect_2_locations, detect_1_locations])
    locations = tf.reshape(locations, [batch_size, -1, 4])
    
    confidences = tf.concat(axis=1, values=[detect_8_confidences, detect_6_confidences, detect_4_confidences, detect_3_confidences, detect_2_confidences, detect_1_confidences])
    confidences = tf.reshape(confidences, [batch_size, -1, 1])
    confidences = tf.sigmoid(confidences)

  return locations, confidences

def build(inputs, num_bboxes_per_cell, reuse=False, scope=''):
    
  # Build the Inception-v3 model
  features, _ = inception_resnet_v2(inputs, reuse=reuse, scope='InceptionResnetV2')
  
  # Save off the original variables (for ease of restoring)
  model_variables = slim.get_model_variables()
  original_inception_vars = {var.op.name:var for var in model_variables}

  # Add on the detection heads
  locs, confs = add_detection_heads(features, num_bboxes_per_cell)

  model = k.Model(inputs, (locs, confs), name='multibox_detection')
  
  return model, original_inception_vars