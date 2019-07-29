"""
Export a model, loading in the moving averages and saving those.
"""

import argparse
import os
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util

from detector import model
from detector.config import parse_config_file


def export(model_path, export_path, export_version, cfg):
  
  graph = tf.Graph()

  # Force all Variables to reside on the CPU.
  with graph.as_default():
    
    # For now we'll assume that the user is sending us a raveled array, totally preprocessed. 
    image_data = tf.placeholder(tf.float32, [None, cfg.INPUT_SIZE * cfg.INPUT_SIZE * 3], name="images")
    batched_images = tf.reshape(image_data, [-1, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    
    batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      'variables_collections' : [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
      'is_training' : False
    }
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)):
      
      
      locations, confidences, inception_vars = model.build(
        inputs = batched_images,
        num_bboxes_per_cell = cfg.NUM_BBOXES_PER_CELL,
        reuse=False,
        scope=''
      )

    ema = tf.train.ExponentialMovingAverage(
      decay=cfg.MOVING_AVERAGE_DECAY
    )   
    shadow_vars = {
      ema.average_name(var) : var
      for var in slim.get_model_variables()
    }

    saver = tf.train.Saver(shadow_vars, reshape=True)

    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    sess = tf.Session(graph=graph, config=sess_config)

    with sess.as_default():
      
      tf.global_variables_initializer().run()
      
      saver.restore(sess, model_path)

      v2c = graph_util.convert_variables_to_constants
      deploy_graph_def = v2c(sess, graph.as_graph_def(), [locations.name[:-2], confidences.name[:-2]])
    
      if not os.path.exists(export_path):
          os.makedirs(export_path)
      save_path = os.path.join(export_path, 'constant_model-%d.pb' % (export_version,))
      with open(save_path, 'wb') as f:
          f.write(deploy_graph_def.SerializeToString())

def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to a model or a directory of checkpoints. The latest model will be used.',
                          required=True, type=str)

    parser.add_argument('--export_path', dest='export_path',
                          help='Path to a directory where the exported model will be saved.',
                          required=True, type=str)
    
    parser.add_argument('--export_version', dest='export_version',
                        help='Version number of the model.',
                        required=True, type=int)
    
    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file.',
                        required=True, type=str)
    

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    print "Called with:"
    print pprint.pprint(args)

    cfg = parse_config_file(args.config_file)

    print "Configurations:"
    print pprint.pprint(cfg)

    export(args.checkpoint_path, args.export_path, args.export_version, cfg=cfg)