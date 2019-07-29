"""
Visualize detections on validation images (where we have ground truth detections).
"""
import os, sys
import argparse
import cPickle as pickle
import logging
import pprint
import time

import inputs
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from matplotlib import pyplot as plt

import model
from config import parse_config_file
import pdb


def visualize(tfrecords, bbox_priors, checkpoint_path, cfg, save_dir):
  
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  graph = tf.Graph()

  # Force all Variables to reside on the CPU.
  with graph.as_default():

    images, batched_bboxes, batched_num_bboxes, image_ids = inputs.input_nodes(
      tfrecords=tfrecords,
      max_num_bboxes = cfg.MAX_NUM_BBOXES,
      num_epochs=None,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      add_summaries = True,
      shuffle_batch=False,
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
        inputs = images,
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

    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    sess = tf.Session(graph=graph, config=sess_config)
    
    height = cfg.HEIGHT
    width = cfg.WIDTH
    #utility to convert image and resize to original size
    image_to_resize = tf.placeholder(tf.float32, [cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    resize_image_op = tf.image.resize_images(image_to_resize, [height, width])
    convert_image_to_uint8 = tf.image.convert_image_dtype(tf.add(tf.div(resize_image_op, 2.0), 0.5), tf.uint8)

    with sess.as_default():

      fetches = [locations, confidences, images, batched_bboxes, batched_num_bboxes,image_ids]
      
      coord = tf.train.Coordinator()
      
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      plt.ion()
      
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
        
        total_sample_count = 0
        step = 0
        done = False

        fig = plt.figure(frameon=False)
        count=0
        while not coord.should_stop() and not done:
          
          t = time.time()
          outputs = sess.run(fetches)
          dt = time.time()-t
          
          locs = outputs[0]
          confs = outputs[1]
          imgs = outputs[2]

          gt_bboxes = outputs[3]
          gt_num_bboxes = outputs[4]

          im_ids = outputs[5]
          
          print locs.shape
          print confs.shape
          
          for b in range(cfg.BATCH_SIZE):
            
            # Show the image
            image = imgs[b]
            image_id = im_ids[b]
            res_image = sess.run(convert_image_to_uint8, {image_to_resize: image})

            ax = plt.axes([0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            fig.add_axes(ax)
            plt.imshow(res_image)
            plt.text(10, 40, 'ID = ' + image_id, fontsize=12, color='r')

            num_gt_bboxes_in_image = gt_num_bboxes[b]
            print "Number of GT Boxes: %d" % (num_gt_bboxes_in_image,)

            # Draw the GT Boxes in blue
            for i in range(num_gt_bboxes_in_image):
              gt_bbox = gt_bboxes[b][i]
              xmin, xmax = gt_bbox[[0, 2]] * width
              ymin, ymax = gt_bbox[[1, 3]] * height
              plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')

            indices = np.argsort(confs[b].ravel())[::-1]
            print "Top 10 Detection Confidences: ", confs[b][indices[:10]].ravel().tolist()

            xdt = 80
            # Draw the most confident boxes in red
            num_detections_to_render = num_gt_bboxes_in_image if num_gt_bboxes_in_image > 0 else 5
            for i, index in enumerate(indices[0:num_detections_to_render]):
            
              loc = locs[b][index].ravel()
              conf = confs[b][index]
              prior = bbox_priors[index]
              
              print "Location: ", loc
              print "Prior: ", prior
              print "Index: ", index
              print "Image id: ", image_id

              # Plot the predicted location in red prior +loc
              pred_loc = prior + loc
              xmin, xmax = pred_loc[[0, 2]] * width
              ymin, ymax = pred_loc[[1, 3]] * height
              plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
              
              # Plot the prior in green
              xmin, xmax = prior[[0, 2]] * width
              ymin, ymax = prior[[1, 3]] * height
              plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'g-')

              plt.text(10, xdt, 'conf bbox = %d: %f' % (i,conf) , fontsize=12, color='r')

              print "Pred Confidence for box %d: %f" % (i, conf)
              xdt+=40
              
            #plt.show()
            plt.savefig(save_dir + '%05d' % int(image_id)+'.pdf')
            plt.savefig(save_dir + '%05d' % int(image_id)+'.png', close=True)

            # t = raw_input("push button ")
            # if t == 'q':
            #   done = True
            #   break
            plt.clf()
            count=+1
            if count==int((cfg.NUM_TRAIN_EXAMPLES-1)/2.):
              break


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

    parser.add_argument('--save_dir', dest='save_dir',
                        help='Path to the directory where to save results and plots',
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

  
  visualize(
    tfrecords=args.tfrecords,
    bbox_priors=bbox_priors,
    checkpoint_path=args.checkpoint_path,
    save_dir = args.save_dir,
    cfg=cfg
  ) 

if __name__ == '__main__':
  main()    
            
