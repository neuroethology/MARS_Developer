"""
Compute object proposals on images.   
"""

import argparse
import cPickle as pickle
import json
import logging
import os
import pprint
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import model
from config import parse_config_file


def extract_patches(image, patch_dims, strides, non_edge_restriction=0.1):
  """
  Args:
    image (np.array) : the image to extract the patches from
    patch_dims (tuple) : the (height, width) size of the patch to extract from the image (assumed to be square)
    strides (tuple) : the (y, x) stride of the patches (in height and width)
    
  Returns:
    list : the patches 
    list : offsets for each patch 
    list : a list of restriction values (one for each side) for the detected bounding boxes.
  """
  
  image_height, image_width = image.shape[:2]
  patch_height, patch_width = patch_dims
  h_stride, w_stride = strides
  
  patches = []
  patch_offsets = []
  patch_restrictions = []

  max_h = image_height-patch_height+1
  max_w = image_width-patch_width+1
  for h in range(0,max_h,h_stride):
    for w in range(0,max_w,w_stride):
      
      p = image[h:h+patch_height, w:w+patch_width]
      patches.append(p)
      patch_offsets.append((h, w))
  
      x1_restriction = 0. if w == 0 else non_edge_restriction
      y1_restriction = 0. if h == 0 else non_edge_restriction
      x2_restriction = 1. if w + patch_width == image_width else 1. - non_edge_restriction
      y2_restriction = 1. if h + patch_height == image_height else 1. - non_edge_restriction
      patch_restrictions.append([x1_restriction, y1_restriction, x2_restriction, y2_restriction])

  patches = np.array(patches).astype(np.float32)
  patch_offsets = np.array(patch_offsets).astype(np.int32)
  patch_restrictions = np.array(patch_restrictions).astype(np.float32)

  # print image.shape
  # print patches.shape
  # print patch_offsets.shape
  # print 

  if patches.shape[0] == 0:
    #print "Bad image?"
    #print image.shape
    patches = np.zeros([0, patch_height, patch_width, 3], dtype=np.float32)
    patch_offsets = np.zeros([0, 2], dtype=np.int32)
    patch_restrictions = np.zeros([0, 4], dtype=np.float32)

  return [patches, patch_offsets, patch_restrictions, np.int32(len(patches))]
      
def filter_proposals(bboxes, confidences, restrictions=None):
  """We want to filter out proposals that are not completely contained in the square [.1, .1, .9, .9]
  
  Args: 
    bboxes np.array: proposed bboxes [x1, y1, x2, y2] in normalized coordinates
    confidences np.array: confidences for the proposed boxes
  
  Returns:
    np.array : the filtered bboxes
    np.array : the confidences for the bboxes
  """
  
  if restrictions is None:
    restrictions = [0.1, 0.1, 0.9, 0.9]

  filtered_bboxes = []
  filtered_confidences = []
  
  for bbox, conf in zip(bboxes, confidences):
    if bbox[0] < restrictions[0]:
      continue
    if bbox[1] < restrictions[1]:
      continue
    if bbox[2] > restrictions[2]:
      continue
    if bbox[3] > restrictions[3]:
      continue
    filtered_bboxes.append(bbox)
    filtered_confidences.append(conf)
  
  return np.array(filtered_bboxes), np.array(filtered_confidences)

def convert_proposals(bboxes, offset, patch_dims, image_dims, is_flipped=0):
  """Convert the coordinates of the proposed bboxes to account for the offset of the patch
  
  Args:
    bboxes (np.array) : the proposed bboxes [x1, y1, x2, y2] in normalized coordinates
    offset (tuple) : the (y, x) offset of the patch in relation to the image
    patch_dims (tuple) : the (height, width) dimensions of the patch
    image_dims (tuple) : the (height, width) dimensions of the image
  Returns:
    np.array : the converted bounding boxes
  """
  
  x_scale = patch_dims[1] / float(image_dims[1])
  y_scale = patch_dims[0] / float(image_dims[0])
  
  x_offset = offset[1] / float(image_dims[1])
  y_offset = offset[0] / float(image_dims[0])

  converted_bboxes = bboxes * np.array([x_scale, y_scale, x_scale, y_scale]) + np.array([x_offset, y_offset, x_offset, y_offset])
  
  if is_flipped:
    converted_bboxes[:, [0, 2]] = converted_bboxes[:, [2, 0]]
    converted_bboxes[:, 0] = 1. - converted_bboxes[:, 0]
    converted_bboxes[:, 2] = 1. - converted_bboxes[:, 2]

  return converted_bboxes


def input_nodes(
  # An array of paths to tfrecords files
  tfrecords,

  # number of times to read the tfrecords
  num_epochs=1,

  # Data queue feeding the model
  batch_size=32,
  num_threads=2,
  capacity = 1000,
  
  # Global configuration
  cfg=None):

  with tf.name_scope('inputs'):

    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=num_epochs,
      shuffle=False
    )

    # Construct a Reader to read examples from the .tfrecords file
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
      }
    )
    
    # Read in a jpeg image
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    
    # Convert the pixel values to be in the range [0,1]
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image_id = features['image/id']
    
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    
    image_height = features['image/height']
    image_width = features['image/width']

    flipped_image = tf.image.flip_left_right(image)

    total_patches = 0
    patches = tf.zeros([0, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3], dtype=tf.float32)
    patch_offsets = tf.zeros([0, 2], dtype = tf.int32)
    patch_dims = tf.zeros([0, 2], dtype=tf.int32)
    patch_is_flipped = tf.zeros([0,1], dtype=tf.int32)
    patch_bbox_restrictions = tf.zeros([0, 4], dtype=tf.float32)
    patch_max_to_keep = tf.zeros([0, 1], dtype=tf.int32)
    
    # Add the original image
    if cfg.DETECTION.USE_ORIGINAL_IMAGE:
      resized_image = tf.image.resize_bilinear(tf.expand_dims(image, 0), [cfg.INPUT_SIZE, cfg.INPUT_SIZE],
                                        align_corners=False)
      image_offsets = np.array([[0, 0]], dtype=np.int32)
      image_dims = tf.cast([[image_height, image_width]], dtype=tf.int32)
      image_is_flipped = np.array([[0]], dtype=np.int32)
      image_bbox_restrictions = np.array([[0, 0, 1, 1]], dtype=np.float32)
      image_max_to_keep = np.array([[cfg.DETECTION.ORIGINAL_IMAGE_MAX_TO_KEEP]], dtype=np.int32)

      total_patches += 1
      patches = tf.concat(axis=0, values=[patches, resized_image])
      patch_offsets = tf.concat(axis=0, values=[patch_offsets, image_offsets])
      patch_dims = tf.concat(axis=0, values=[patch_dims, image_dims])
      patch_is_flipped = tf.concat(axis=0, values=[patch_is_flipped, image_is_flipped])
      patch_bbox_restrictions = tf.concat(axis=0, values=[patch_bbox_restrictions, image_bbox_restrictions])
      patch_max_to_keep = tf.concat(axis=0, values=[patch_max_to_keep, image_max_to_keep])
    
    # Add a flipped version of the original image
    if cfg.DETECTION.USE_FLIPPED_ORIGINAL_IMAGE:
      flipped_resized_image = tf.image.resize_bilinear(tf.expand_dims(flipped_image, 0), [cfg.INPUT_SIZE, cfg.INPUT_SIZE],
                                        align_corners=False)
      flipped_image_offsets = np.array([[0, 0]], dtype=np.int32)
      flipped_image_dims = tf.cast([[image_height, image_width]], dtype=tf.int32)
      flipped_image_is_flipped = np.array([[1]], dtype=np.int32)
      flipped_image_restrictions = np.array([[0, 0, 1, 1]], dtype=np.float32)
      flipped_image_max_to_keep = np.array([[cfg.DETECTION.FLIPPED_IMAGE_MAX_TO_KEEP]], dtype=np.int32)

      total_patches += 1
      patches = tf.concat(axis=0, values=[patches, flipped_resized_image])
      patch_offsets = tf.concat(axis=0, values=[patch_offsets, flipped_image_offsets])
      patch_dims = tf.concat(axis=0, values=[patch_dims, flipped_image_dims])
      patch_is_flipped = tf.concat(axis=0, values=[patch_is_flipped, flipped_image_is_flipped])
      patch_bbox_restrictions = tf.concat(axis=0, values=[patch_bbox_restrictions, flipped_image_restrictions])
      patch_max_to_keep = tf.concat(axis=0, values=[patch_max_to_keep, flipped_image_max_to_keep])

    # Extract the crops
    for crop_info in cfg.DETECTION.get('CROPS', []):
      params = []
      if crop_info.FLIP:
        params.append(flipped_image)
      else:
        params.append(image)
      
      crop_dims = (crop_info.HEIGHT, crop_info.WIDTH)
      crop_strides = (crop_info.HEIGHT_STRIDE, crop_info.WIDTH_STRIDE)

      params.append(crop_dims)
      params.append(crop_strides)

      output = tf.py_func(extract_patches, params, [tf.float32, tf.int32, tf.float32, tf.int32])
      num_cropped_patches = output[3]
      cropped_patches = output[0]
      cropped_patches.set_shape([None, crop_info.HEIGHT, crop_info.WIDTH, 3])
      cropped_patches = tf.cond(tf.greater(num_cropped_patches, 0), 
        lambda: tf.image.resize_images(cropped_patches, size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE], method=0, align_corners=False),
        lambda: tf.zeros([0, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
      )
      cropped_patch_offsets = output[1]
      cropped_patch_restrictions = output[2]
      cropped_patch_dims = tf.tile([crop_dims], [num_cropped_patches, 1])
      cropped_patch_is_flipped = tf.ones([num_cropped_patches, 1], dtype=np.int32) if crop_info.FLIP else tf.zeros([num_cropped_patches, 1], dtype=np.int32)
      cropped_patch_max_to_keep = tf.tile([[crop_info.MAX_TO_KEEP]], [num_cropped_patches, 1])

      total_patches += num_cropped_patches
      patches = tf.concat(axis=0, values=[patches, cropped_patches])
      patch_offsets = tf.concat(axis=0, values=[patch_offsets, cropped_patch_offsets])
      patch_dims = tf.concat(axis=0, values=[patch_dims, cropped_patch_dims])
      patch_is_flipped = tf.concat(axis=0, values=[patch_is_flipped, cropped_patch_is_flipped])
      patch_bbox_restrictions = tf.concat(axis=0, values=[patch_bbox_restrictions, cropped_patch_restrictions])
      patch_max_to_keep = tf.concat(axis=0, values=[patch_max_to_keep, cropped_patch_max_to_keep])

   
    image_height_widths = tf.tile([[image_height, image_width]], [total_patches, 1])
    image_ids = tf.tile([[image_id]], [total_patches, 1])

    # Set the shape of everything for the queue
    patches.set_shape([None, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    patch_offsets.set_shape([None, 2])
    patch_dims.set_shape([None, 2])
    patch_is_flipped.set_shape([None, 1])
    patch_bbox_restrictions.set_shape([None, 4])
    patch_max_to_keep.set_shape([None, 1])
    image_height_widths.set_shape([None, 2])
    image_ids.set_shape([None, 1])

    batched_images, batched_offsets, batched_dims, batched_is_flipped, batched_bbox_restrictions, batched_max_to_keep, batched_heights_widths, batched_image_ids = tf.train.batch(
      [patches, patch_offsets, patch_dims, patch_is_flipped, patch_bbox_restrictions, patch_max_to_keep, image_height_widths, image_ids],
      batch_size=batch_size,
      num_threads=num_threads,
      capacity= capacity, 
      enqueue_many=True
    )

  # return a batch of images and their labels
  return batched_images, batched_offsets, batched_dims, batched_is_flipped, batched_bbox_restrictions, batched_max_to_keep, batched_heights_widths, batched_image_ids

def detect(tfrecords, bbox_priors, checkpoint_path, save_dir, max_detections, max_iterations, cfg):
  
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()

  # Force all Variables to reside on the CPU.
  with graph.as_default():

    batched_images, batched_offsets, batched_dims, batched_is_flipped, batched_bbox_restrictions, batched_max_to_keep, batched_heights_widths, batched_image_ids = input_nodes(
      tfrecords=tfrecords,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity = cfg.QUEUE_CAPACITY,
      cfg=cfg
    )

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
  
    
    # Restore the parameters
    saver = tf.train.Saver(shadow_vars, reshape=True)

    fetches = [locations, confidences, batched_offsets, batched_dims, batched_is_flipped, batched_bbox_restrictions, batched_max_to_keep, batched_heights_widths, batched_image_ids]
    
    coord = tf.train.Coordinator()

    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    sess = tf.Session(graph=graph, config=sess_config)

    detection_results = []
    with sess.as_default():

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      try:

        if tf.gfile.IsDirectory(checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        
        if checkpoint_path is None:
          print "ERROR: No checkpoint file found."
          return

        # Restores from checkpoint
        saver.restore(sess, checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
        print "Found model for global step: %d" % (global_step,)
        
        print_str = ', '.join([
          'Step: %d',
          'Time/image (ms): %.1f'
        ])

        step = 0
        while not coord.should_stop():
          
          t = time.time()
          outputs = sess.run(fetches)
          dt = time.time()-t
          
          locs = outputs[0]
          confs = outputs[1]
          patch_offsets = outputs[2]
          patch_dims = outputs[3]
          patch_is_flipped = outputs[4]
          patch_bbox_restrictions = outputs[5]
          patch_max_to_keep = outputs[6]
          image_height_widths = outputs[7]
          image_ids = outputs[8]

          for b in range(cfg.BATCH_SIZE):

            img_id = int(np.asscalar(image_ids[b]))
            
            predicted_bboxes = locs[b] + bbox_priors
            predicted_bboxes = np.clip(predicted_bboxes, 0., 1.)
            predicted_confs = confs[b]

            filtered_bboxes, filtered_confs = filter_proposals(predicted_bboxes, predicted_confs, patch_bbox_restrictions[b])

            # No valid predictions? 
            if filtered_bboxes.shape[0] == 0:
              continue
            
            # Lets get rid of some of the predictions
            num_preds_to_keep = np.asscalar(patch_max_to_keep[b])
            sorted_idxs = np.argsort(filtered_confs.ravel())[::-1]
            sorted_idxs = sorted_idxs[:num_preds_to_keep]
            filtered_bboxes = filtered_bboxes[sorted_idxs]
            filtered_confs = filtered_confs[sorted_idxs]
            
            # Convert the bounding boxes to the original image dimensions
            converted_bboxes = convert_proposals(
              bboxes = filtered_bboxes, 
              offset = patch_offsets[b], 
              patch_dims = patch_dims[b], 
              image_dims = image_height_widths[b],
              is_flipped= patch_is_flipped[b]
            )

            for k in range(converted_bboxes.shape[0]):  
              detection_results.append({
                "image_id" : img_id,
                "bbox" : converted_bboxes[k].tolist(),
                "score" : float(np.asscalar(filtered_confs[k])),
              })

          step += 1
          print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000)

          if max_iterations > 0 and step == max_iterations:
            break  

      except tf.errors.OutOfRangeError as e:
        pass
        
      coord.request_stop()
      coord.join(threads)

      # save the results
      save_path = os.path.join(save_dir, "results-dense-%d.json" % global_step)
      with open(save_path, 'w') as f: 
        json.dump(detection_results, f)

def parse_args():

    parser = argparse.ArgumentParser(description='Detect objects using a pretrained Multibox model')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)
    
    parser.add_argument('--priors', dest='priors',
                          help='path to the bounding box priors pickle file', type=str,
                          required=True)
    
    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Either a path to a specific model, or a path to a directory where checkpoint files are stored. If a directory, the latest model will be tested against.', type=str,
                          required=True, default=None)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run. Set to 0 to run on all records.',
                        required=False, type=int, default=0)
    
    parser.add_argument('--max_detections', dest='max_detections',
                        help='Maximum number of detection to store per image',
                        required=False, type=int, default=100) 
    
    parser.add_argument('--save_dir', dest='save_dir',
                        help='Directory to save the json result file.',
                        required=True, type=str)               

    args = parser.parse_args()
    
    return args

def main():
  args = parse_args()
  print "Command line arguments:"
  pprint.pprint(vars(args))
  print

  cfg = parse_config_file(args.config_file)
  print "Configurations:"
  pprint.pprint(cfg)
  print 
    
  with open(args.priors) as f:
    bbox_priors = pickle.load(f)
  bbox_priors = np.array(bbox_priors).astype(np.float32)
  
  detect(
    tfrecords=args.tfrecords,
    bbox_priors=bbox_priors,
    checkpoint_path=args.checkpoint_path,
    save_dir = args.save_dir,
    max_detections = args.max_detections,
    max_iterations = args.max_iterations,
    cfg=cfg
  ) 

if __name__ == '__main__':
  main()
    