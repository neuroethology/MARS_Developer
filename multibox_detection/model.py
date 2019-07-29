import tensorflow as tf

slim = tf.contrib.slim


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_resnet_v2(inputs,
                        reuse=None,
                        scope='InceptionResnetV2'):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):

      # 149 x 149 x 32
      net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                        scope='Conv2d_1a_3x3')
      end_points['Conv2d_1a_3x3'] = net
      # 147 x 147 x 32
      net = slim.conv2d(net, 32, 3, padding='VALID',
                        scope='Conv2d_2a_3x3')
      end_points['Conv2d_2a_3x3'] = net
      # 147 x 147 x 64
      net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
      end_points['Conv2d_2b_3x3'] = net
      # 73 x 73 x 64
      net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                            scope='MaxPool_3a_3x3')
      end_points['MaxPool_3a_3x3'] = net
      # 73 x 73 x 80
      net = slim.conv2d(net, 80, 1, padding='VALID',
                        scope='Conv2d_3b_1x1')
      end_points['Conv2d_3b_1x1'] = net
      # 71 x 71 x 192
      net = slim.conv2d(net, 192, 3, padding='VALID',
                        scope='Conv2d_4a_3x3')
      end_points['Conv2d_4a_3x3'] = net
      # 35 x 35 x 192
      net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                            scope='MaxPool_5a_3x3')
      end_points['MaxPool_5a_3x3'] = net

      # 35 x 35 x 320
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                      scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                      scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                        scope='AvgPool_0a_3x3')
          tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                      scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[tower_conv, tower_conv1_1,
                            tower_conv2_2, tower_pool_1])

      end_points['Mixed_5b'] = net
      net = slim.repeat(net, 10, block35, scale=0.17)

      # 17 x 17 x 1024
      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                      stride=2, padding='VALID',
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                        scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[tower_conv, tower_conv1_2, tower_pool])

      end_points['Mixed_6a'] = net
      net = slim.repeat(net, 20, block17, scale=0.10)

      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                      padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                      padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                      padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                        scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[tower_conv_1, tower_conv1_1,
                            tower_conv2_2, tower_pool])

      end_points['Mixed_7a'] = net

      net = slim.repeat(net, 9, block8, scale=0.20)
      net = block8(net, activation_fn=None)
      
      # GVH: Not sure if we want or need this convolution
      # 8 x 8 x 2080
      net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
      end_points['Conv2d_7b_1x1'] = net
    
    # 8 x 8 x 1536
    return net, end_points

def build_detection_heads(inputs, num_bboxes_per_cell, scope='Multibox', reuse=None):
  
  endpoints = {}
  
  with tf.variable_scope(scope, 'Multibox', [inputs], reuse=reuse):

    with slim.arg_scope([slim.conv2d, slim.avg_pool2d], stride=1, padding='SAME'):

      # 8 x 8 grid cells
      with tf.variable_scope("8x8"):
        # 8 x 8 x 2048 
        branch8x8 = slim.conv2d(inputs, 96, [1, 1])
        # 8 x 8 x 96
        branch8x8 = slim.conv2d(branch8x8, 96, [3, 3])
        # 8 x 8 x 96
        endpoints['8x8_locations'] = slim.conv2d(branch8x8, num_bboxes_per_cell * 4, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
        # 8 x 8 x 96
        endpoints['8x8_confidences'] = slim.conv2d(branch8x8, num_bboxes_per_cell, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )

      # 6 x 6 grid cells
      with tf.variable_scope("6x6"):
        # 8 x 8 x 2048 
        branch6x6 = slim.conv2d(inputs, 96, [3, 3])
        # 8 x 8 x 96
        branch6x6 = slim.conv2d(branch6x6, 96, [3, 3], padding = "VALID")
        # 6 x 6 x 96
        endpoints['6x6_locations'] = slim.conv2d(branch6x6, num_bboxes_per_cell * 4, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
        # 6 x 6 x 96
        endpoints['6x6_confidences'] = slim.conv2d(branch6x6, num_bboxes_per_cell, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
      
      # 8 x 8 x 2048
      net = slim.conv2d(inputs, 256, [3, 3], stride=2)

      # 4 x 4 grid cells
      with tf.variable_scope("4x4"):
        # 4 x 4 x 256
        branch4x4 = slim.conv2d(net, 128, [3, 3])
        # 4 x 4 x 128
        endpoints['4x4_locations'] = slim.conv2d(branch4x4, num_bboxes_per_cell * 4, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
        # 4 x 4 x 128
        endpoints['4x4_confidences'] = slim.conv2d(branch4x4, num_bboxes_per_cell, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )

      # 3 x 3 grid cells
      with tf.variable_scope("3x3"):
        # 4 x 4 x 256
        branch3x3 = slim.conv2d(net, 128, [1, 1])
        # 4 x 4 x 128
        branch3x3 = slim.conv2d(branch3x3, 96, [2, 2], padding="VALID")
        # 3 x 3 x 96
        endpoints['3x3_locations'] = slim.conv2d(branch3x3, num_bboxes_per_cell * 4, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
        # 3 x 3 x 96
        endpoints['3x3_confidences'] = slim.conv2d(branch3x3, num_bboxes_per_cell, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
        
      # 2 x 2 grid cells
      with tf.variable_scope("2x2"):
        # 4 x 4 x 256
        branch2x2 = slim.conv2d(net, 128, [1, 1])
        # 4 x 4 x 128
        branch2x2 = slim.conv2d(branch2x2, 96, [3, 3], padding = "VALID")
        # 2 x 2 x 96
        endpoints['2x2_locations'] = slim.conv2d(branch2x2, num_bboxes_per_cell * 4, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
        # 2 x 2 x 96
        endpoints['2x2_confidences'] = slim.conv2d(branch2x2, num_bboxes_per_cell, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
        
      # 1 x 1 grid cell
      with tf.variable_scope("1x1"):
        # 8 x 8 x 2048
        branch1x1 = slim.avg_pool2d(inputs, [8, 8], padding="VALID")
        # 1 x 1 x 2048
        endpoints['1x1_locations'] = slim.conv2d(branch1x1, 4, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
        # 1 x 1 x 2048
        endpoints['1x1_confidences'] = slim.conv2d(branch1x1, 1, [1, 1],
          activation_fn=None, normalizer_fn=None, biases_initializer=None
        )
      
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
    
  return locations, confidences, endpoints

def build(inputs, num_bboxes_per_cell, reuse=False, scope=''):
    
  # Build the Inception-v3 model
  features, _ = inception_resnet_v2(inputs, reuse=reuse, scope='InceptionResnetV2')
  
  # Save off the original variables (for ease of restoring)
  model_variables = slim.get_model_variables()
  original_inception_vars = {var.op.name:var for var in model_variables}

  # Add on the detection heads
  locs, confs, _ = build_detection_heads(features, num_bboxes_per_cell)
  
  return locs, confs, original_inception_vars