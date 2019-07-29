"""
File for detecting parts on images without ground truth.
"""
import argparse
import cPickle as pickle
import json
import logging
import numpy as np
import os
import pprint
import scipy
from scipy.misc import imresize
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import sys
import tensorflow as tf
from tensorflow.contrib import slim
import time
from matplotlib import pyplot as plt

from config import parse_config_file
import detect_inputs as inputs
import model

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import pdb

def get_local_maxima(data, x_offset, y_offset, input_width, input_height, image_width, image_height, threshold=0.000002, neighborhood_size=15):
  """ Return the local maxima of the heatmaps
  Args:
    data: the heatmaps
    x_offset : the normalized x_offset coordinate, used to transform the local maxima back to image space
    y_offset : the normalized y_offset coordinate

  """
  
  keypoints = []

  heatmap_height, heatmap_width, num_parts = data.shape
  heatmap_height = float(heatmap_height)
  heatmap_width = float(heatmap_width)

  input_width = float(input_width)
  input_height = float(input_height)

  image_width = float(image_width)
  image_height = float(image_height)

  for k in xrange(num_parts):

    data1 = data[:, :, k]
    data_max = filters.maximum_filter(data1, neighborhood_size)
    maxima = (data1 == data_max)
    data_min = filters.minimum_filter(data1, neighborhood_size)
    diff = (data_max - data_min) > (threshold * data1.max())
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x1, y1, v1 = [], [], []
    for dy,dx in slices:

      x_center = (dx.start + dx.stop - 1) / 2.
      x_center_int = int(np.round(x_center))
      normalized_x_center = x_center * (input_width / heatmap_width) * (1. / image_width) + x_offset

      y_center = (dy.start + dy.stop - 1) / 2.
      y_center_int = int(np.round(y_center))
      normalized_y_center = y_center * (input_height / heatmap_height) * (1. / image_height) + y_offset

      x1.append(float(normalized_x_center))
      y1.append(float(normalized_y_center))
      v1.append(float(data1[y_center_int, x_center_int]))

    keypoints.append({'x': x1, 'y': y1, 'score': v1})
  
  return keypoints

def detect(tfrecords, checkpoint_path, save_dir, max_iterations, iterations_per_record, cfg):

  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()
  
  with graph.as_default():
    
    batched_images, batched_bboxes, batched_scores, batched_image_ids, batched_labels, batched_image_height_widths, batched_crop_bboxes,batched_filenames = inputs.input_nodes(
      tfrecords=tfrecords,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity = cfg.QUEUE_CAPACITY,
      cfg=cfg
    )
    
    batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections' : [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training' : False
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
      
      predicted_heatmaps = model.build(
        input = batched_images, 
        num_parts = cfg.PARTS.NUM_PARTS
      )
    
    ema = tf.train.ExponentialMovingAverage(
      decay=cfg.MOVING_AVERAGE_DECAY
    )   
    shadow_vars = {
      ema.average_name(var) : var
      for var in slim.get_model_variables()
    }

    saver = tf.train.Saver(shadow_vars, reshape=True)
    
    fetches = [predicted_heatmaps[-1], batched_bboxes, batched_scores, batched_image_ids, batched_labels, batched_image_height_widths, batched_crop_bboxes]

    # Now create a training coordinator that will control the different threads
    coord = tf.train.Coordinator()
    
    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    session = tf.Session(graph=graph, config=sess_config)
    
    with session.as_default():

      # make sure to initialize all of the variables
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      
      # launch the queue runner threads
      threads = tf.train.start_queue_runners(sess=session, coord=coord)

      results = []
      #output_writer = None
      
      try:
        
        if tf.gfile.IsDirectory(checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        
        if checkpoint_path is None:
          print "ERROR: No checkpoint file found."
          return

        # Restores from checkpoint
        saver.restore(session, checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
        print "Found model for global step: %d" % (global_step,)
        
        # we will store results into a tfrecord file
        output_writer_iteration = 0
        #output_path = os.path.join(save_dir, 'heatmap_results-%d-%d.tfrecords' % (global_step, output_writer_iteration))
        #output_writer = tf.python_io.TFRecordWriter(output_path)
        
        
        step = 0
        print_str = ', '.join([
          'Step: %d',
          'Time/image network (ms): %.1f',
          'Time/image post proc (ms): %.1f'
        ])
        while not coord.should_stop():
          t = time.time()
          outputs = session.run(fetches)
          dt = time.time() - t
          t = time.time()
          for b in range(cfg.BATCH_SIZE):

            heatmaps = outputs[0][b]
            # print(heatmaps.shape)
            # raw_input('hi')
            # return
            bbox = outputs[1][b]
            score = outputs[2][b]
            image_id = outputs[3][b]
            label = outputs[4][b]
            image_height_widths = outputs[5][b]
            crop_bbox = outputs[6][b]
            
            # Attempt to compress the heatmaps by just saving local maxima
            heatmaps = np.clip(heatmaps, 0., 1.)
            
             # We need to transform the keypoints back to the original image space.
            image_height, image_width = image_height_widths

            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox 
            crop_w, crop_h = np.array([crop_x2 - crop_x1, crop_y2 - crop_y1]) * np.array([image_width, image_height], dtype=np.float32)

            if cfg.LOOSE_BBOX_CROP:
              restrict_to_bbox=True
              if restrict_to_bbox:
                # Crop out the portion of the heatmap that corresponds to the bounding box of the object

                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

                heatmap_bbox_x1 = int(np.round((bbox_x1 - crop_x1) * ( image_width / crop_w ) * cfg.HEATMAP_SIZE ))
                heatmap_bbox_y1 = int(np.round((bbox_y1 - crop_y1) * ( image_height / crop_h) * cfg.HEATMAP_SIZE ))
                heatmap_bbox_x2 = int(np.round((bbox_x2 - crop_x1) * ( image_width / crop_w ) * cfg.HEATMAP_SIZE ))
                heatmap_bbox_y2 = int(np.round((bbox_y2 - crop_y1) * ( image_height / crop_h) * cfg.HEATMAP_SIZE ))

                #print "%d:%d, %d:%d" % (heatmap_bbox_y1, heatmap_bbox_y2, heatmap_bbox_x1, heatmap_bbox_x2)

                heatmaps_bbox = heatmaps[heatmap_bbox_y1:heatmap_bbox_y2, heatmap_bbox_x1:heatmap_bbox_x2]

                bbox_w = (bbox_x2 - bbox_x1) * image_width 
                bbox_h = (bbox_y2 - bbox_y1) * image_height

                keypoints = get_local_maxima(heatmaps_bbox, bbox_x1, bbox_y1, bbox_w, bbox_h, image_width, image_height)

              else:

                keypoints = get_local_maxima(heatmaps, crop_x1, crop_y1, crop_w, crop_h, image_width, image_height)
            else:
              bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
              bbox_w = (bbox_x2 - bbox_x1) * image_width 
              bbox_h = (bbox_y2 - bbox_y1) * image_height
              
              if bbox_h > bbox_w:
                input_size = bbox_h
              else:
                input_size = bbox_w

              keypoints = get_local_maxima(heatmaps, bbox_x1, bbox_y1, input_size, input_size, image_width, image_height)

            # Convert to types that can be saved in the tfrecord file
            image_id = int(np.asscalar(image_id))
            bbox = bbox.tolist()
            score = float(np.asscalar(score))
            label = int(np.asscalar(label))
            
            results.append({
              "image_id" : image_id, 
              "bbox" : bbox,
              "score" : score,
              "keypoints" : keypoints,
              "label" : label
            })
          
          dtt = time.time() - t
          print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000, (dtt / cfg.BATCH_SIZE) * 1000)  
          step += 1
          
          if (step % iterations_per_record) == 0:
            output_path = os.path.join(save_dir, 'heatmap_results-%d-%d.json' % (global_step, output_writer_iteration))
            with open(output_path, 'w') as f: 
              json.dump(results, f)
            output_writer_iteration += 1
            results = []

          if max_iterations > 0 and step == max_iterations:
              break
          
      except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)
      
      # When done, ask the threads to stop. It is innocuous to request stop twice.
      coord.request_stop()
      # And wait for them to actually do it.
      coord.join(threads)
      
      #if output_writer != None:
      #  output_writer.close()
      
      if len(results) > 0:
        output_path = os.path.join(save_dir, 'heatmap_results-%d-%d.json' % (global_step, output_writer_iteration))
        with open(output_path, 'w') as f: 
          json.dump(results, f)

def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=False, default=None)
                          
    parser.add_argument('--save_dir', dest='save_dir',
                        help='Path to the directory where the results will be saved',
                        required=True, type=str)
                        
    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run. Set to 0 to run on all records.',
                        required=False, type=int, default=0)
    
    parser.add_argument('--iterations_per_record', dest='iterations_per_record',
                        help='The number of iterations to store in a tfrecord file before creating another one.',
                        required=False, type=int, default=1000)

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = parse_config_file(args.config_file)

    print "Configurations:"
    print pprint.pprint(cfg)

    detect(
      tfrecords=args.tfrecords,
      checkpoint_path=args.checkpoint_path,
      save_dir = args.save_dir,
      max_iterations = args.max_iterations,
      iterations_per_record = args.iterations_per_record,
      cfg=cfg
    )