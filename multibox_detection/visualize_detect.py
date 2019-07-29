"""
Visualize detection results. 
"""

import argparse
import cPickle as pickle
import logging
import pdb
import pprint
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from matplotlib import pyplot as plt

import model
from config import parse_config_file
from detect import input_nodes, filter_proposals, convert_proposals


def detect_visualize(tfrecords, bbox_priors, checkpoint_path, cfg):
  
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

    fetches = [locations, confidences, batched_offsets, batched_dims, batched_is_flipped, batched_bbox_restrictions, batched_max_to_keep, batched_heights_widths, batched_image_ids, batched_images]
    
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


    # Little utility to convert the float images to uint8
    image_to_convert = tf.placeholder(tf.float32)
    convert_image_to_uint8 = tf.image.convert_image_dtype(tf.add(tf.div(image_to_convert, 2.0), 0.5), tf.uint8) 

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
        
        plt.ion()
        original_image = None
        current_image_id = None

        step = 0
        done = False
        while not coord.should_stop() and not done:
          
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
          
          images = outputs[9]

          for b in range(cfg.BATCH_SIZE):
            
            print "Patch Dims: ", patch_dims[b]
            print "Patch Offset: ", patch_offsets[b]
            print "Max to keep: ", patch_max_to_keep[b]
            print "Patch restrictions: ", patch_bbox_restrictions[b]
            print "Image HxW: ", image_height_widths[b]
            print

            if current_image_id is None or current_image_id != image_ids[b]:
              original_image = images[b]
              original_image = sess.run(convert_image_to_uint8, {image_to_convert : images[b]})
              current_image_id = image_ids[b]

            img_id = int(np.asscalar(image_ids[b]))
            
            predicted_bboxes = locs[b] + bbox_priors
            predicted_bboxes = np.clip(predicted_bboxes, 0., 1.)
            predicted_confs = confs[b]

            # Keep only the predictions that are completely contained in the [0.1, 0.1, 0.9, 0.9] square
            # for this patch
            #if patch_restrict[b]:
            #  filtered_bboxes, filtered_confs = filter_proposals(predicted_bboxes, predicted_confs) 
            #else:
            #  filtered_bboxes = predicted_bboxes
            #  filtered_confs = predicted_confs
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
            
            plt.figure('Cropped Size')
            uint8_image = sess.run(convert_image_to_uint8, {image_to_convert : images[b]})
            plt.imshow(uint8_image)
            num_detections_to_render = min(filtered_bboxes.shape[0], 10)
            for i in range(num_detections_to_render):
        
              loc = filtered_bboxes[i].ravel()
              conf = filtered_confs[i]
              
              #print "Location: ", loc
              #print "Conf: ", conf
              
              # Plot the predicted location in red
              xmin, ymin, xmax, ymax = loc * cfg.INPUT_SIZE
              if i<2:
                plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
              else:
                plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'k-')


            # Convert the bounding boxes to the original image dimensions
            converted_bboxes = convert_proposals(
              bboxes = filtered_bboxes, 
              offset = patch_offsets[b], 
              patch_dims = patch_dims[b], 
              image_dims = image_height_widths[b],
              is_flipped= patch_is_flipped[b]
            )

            plt.figure('Resized')
            plt.imshow(original_image)
            num_detections_to_render = min(converted_bboxes.shape[0], 10)
            for i in range(num_detections_to_render):
        
              loc = converted_bboxes[i].ravel()
              conf = filtered_confs[i]
              
              #print "Location: ", loc
              #print "Conf: ", conf
              
              # Plot the predicted location in red
              xmin, ymin, xmax, ymax = loc * cfg.INPUT_SIZE
              plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')

            r = raw_input("press button: ")
            if r == "q":
              done=True
              break
            
            plt.close('all')


          step += 1
          print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000) 

      except tf.errors.OutOfRangeError as e:
        pass
        
      coord.request_stop()
      coord.join(threads)

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
  
  detect_visualize(
    tfrecords=args.tfrecords,
    bbox_priors=bbox_priors,
    checkpoint_path=args.checkpoint_path,
    cfg=cfg
  ) 

if __name__ == '__main__':
  main()
    