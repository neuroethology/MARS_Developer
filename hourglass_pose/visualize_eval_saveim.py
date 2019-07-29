"""
File for detecting parts on images without ground truth.
"""
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import pprint
from scipy import interpolate
from scipy.misc import imresize
import sys
import tensorflow as tf
from tensorflow.contrib import slim
import time

from config import parse_config_file
import eval_inputs as inputs
import model
import pdb


def visualize(tfrecords, checkpoint_path, cfg,savedir):

  if not os.path.exists(savedir):os.makedirs(savedir)

  tf.logging.set_verbosity(tf.logging.DEBUG)

  graph = tf.Graph()
  
  num_parts = cfg.PARTS.NUM_PARTS
  
  with graph.as_default():
    
    batched_images, batched_bboxes, batched_parts, batched_part_visibilities, batched_image_ids, batched_image_height_widths, batched_crop_bboxes = inputs.input_nodes(
      tfrecords=tfrecords,
      num_parts = num_parts,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity = cfg.QUEUE_CAPACITY,
      shuffle_batch=True,
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

    # Bundle up a graph that gets everything we need.
    fetches = [predicted_heatmaps[-1], batched_bboxes, batched_parts, batched_part_visibilities, batched_image_ids, batched_image_height_widths, batched_crop_bboxes, batched_images]

    # Now create a training coordinator that will control the different threads
    coord = tf.train.Coordinator()

    # Set up our session to use the correct amount of GPU memory.
    sess_config = tf.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    session = tf.Session(graph=graph, config=sess_config) # Setup session.
    
    with session.as_default():

      # make sure to initialize all of the variables
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      
      # launch the queue runner threads
      threads = tf.train.start_queue_runners(sess=session, coord=coord)

      # Attempt to load the existing model from a checkpoint.
      try:
        
        if tf.gfile.IsDirectory(checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        
        if checkpoint_path is None:
          print "ERROR: No checkpoint file found."
          return

        # Restores the model from the checkpoint we found.
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

        # Turn the interactive mode for MatplotLib on.
        plt.ioff()

        # Set up image converting and image resizing graphs, respectively.
        image_to_convert = tf.placeholder(tf.float32)
        convert_to_uint8 = tf.image.convert_image_dtype(tf.add(tf.div(image_to_convert, 2.0), 0.5), tf.uint8)
        
        image_to_resize = tf.placeholder(tf.float32)
        resize_to_input_size = tf.image.resize_bilinear(image_to_resize, size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE])

        # Make the indices for a (?)x3 array, where (?) is the minimum number such that (?)x3 >= NUM_PARTS
        num_part_cols = 3
        num_part_rows = int(np.ceil(cfg.PARTS.NUM_PARTS / (num_part_cols * 1.)))

        # Initialize for the coming while loop.
        done = False
        step = 0
        print_str = ', '.join([
          'Step: %d',
          'Time/image network (ms): %.1f',
          'Time/image post proc (ms): %.1f'
        ])
        while not coord.should_stop() and not done:
          t = time.time()
          # Fetch the outputs (heatmaps, images, etc...)
          outputs = session.run(fetches)
          dt = time.time() - t
          t = time.time()
          for b in range(cfg.BATCH_SIZE):
            # Make a figure for the heatmaps
            fig = plt.figure("heat maps",figsize=(15,15))
            plt.clf() 

            # Extract the outputs
            heatmaps = outputs[0][b]
            bbox = outputs[1][b]
            parts = outputs[2][b]
            part_visibilities = outputs[3][b]
            image_id = outputs[4][b]
            image_height_widths = outputs[5][b]
            crop_bbox = outputs[6][b]
            image = outputs[7][b]

            # Convert the image to uint8
            int8_image = session.run(convert_to_uint8, {image_to_convert : image})

            heatmaps = np.clip(heatmaps, 0., 1.) # Enable double-ended saturation of the image.
            heatmaps = np.expand_dims(heatmaps, 0) # Add another row to the heatmaps array; (Necessary because it has 3 channels?)
            resized_heatmaps = session.run(resize_to_input_size, {image_to_resize : heatmaps}) # Resize the heatmaps
            resized_heatmaps = np.squeeze(resized_heatmaps) # Get rid of that row we added.

            image_height, image_width = image_height_widths
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
            crop_w, crop_h = np.array([crop_x2 - crop_x1, crop_y2 - crop_y1]) * np.array([image_width, image_height], dtype=np.float32)

            # Not sure what this is doing...
            # if crop_h > crop_w:
            #   new_height = cfg.INPUT_SIZE
            #   width_factor = new_height / float(crop_h)
            #   im_scale = width_factor
            # else:
            #   new_width = cfg.INPUT_SIZE
            #   height_factor = new_width / float(crop_w)
            #   im_scale = height_factor

            for j in range(cfg.PARTS.NUM_PARTS):
            
              heatmap = resized_heatmaps[:,:,j]
              
              # fig.add_subplot(num_part_rows, num_part_cols, j+1)
              plt.imshow(int8_image)

              # rescale the values of the heatmap 
              f = interpolate.interp1d([0., 1.], [0, 255])
              int_heatmap = f(heatmap).astype(np.uint8)

              # Add the heatmap as an alpha channel over the image
              blank_image = np.zeros(image.shape, np.uint8)
              blank_image[:] = [255, 0, 0]
              heat_map_alpha = np.dstack((blank_image, int_heatmap))
              # plt.imshow(heat_map_alpha)
              plt.axis('off')
              # plt.title(cfg.PARTS.NAMES[j])
              
              # Render the argmax point
              x, y = np.array(np.unravel_index(np.argmax(heatmap), heatmap.shape)[::-1])
              plt.plot(x, y, color=cfg.PARTS.COLORS[j], marker=cfg.PARTS.SYMBOLS[j],markersize=15, label=cfg.PARTS.NAMES[j])
              
              # Render the ground truth part location
              part_v = part_visibilities[j]
              if part_v:
                indx = j*2 # There's an x and a y, so we have to go two parts by two parts
                part_x, part_y = parts[indx:indx+2]

                # We need to transform the ground truth annotations to the crop space
                input_size = cfg.INPUT_SIZE
                part_x = (part_x - crop_x1) * input_size / (crop_x2 - crop_x1)
                part_y = (part_y - crop_y1) * input_size / (crop_y2 - crop_y1)

                plt.plot(part_x, part_y, color=cfg.PARTS.COLORS[j], marker='*',markersize=15, label=cfg.PARTS.NAMES[j])
              else:
                print "Part not visible"

              # print(str(image_id[0]))

              # print "%s : max %0.3f, min %0.3f" % (cfg.PARTS.NAMES[j], np.max(heatmap), np.min(heatmap))

            # plt.show()
            plt.tight_layout()
            plt.savefig(savedir+ str(image_id[0]) + '_' + str(b)+'.png')
            plt.savefig(savedir+ str(image_id[0]) +'_' + str(b)+ '.pdf')
            #plt.pause(0.0001)
            # r = raw_input("Push button")
            # if r != "":
            #   done = True
            #   break
          dtt = time.time() - t
          print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000, (dtt / cfg.BATCH_SIZE) * 1000)
          step += 1

          
          
      except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)
      
      # When done, ask the threads to stop. It is innocuous to request stop twice.
      coord.request_stop()
      # And wait for them to actually do it.
      coord.join(threads)
      


def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=False, default=None)
                        
    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--savedir', dest='savedir',
                        help='Path to the saved irectory',
                        required=True, type=str)

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = parse_config_file(args.config_file)

    print "Configurations:"
    print pprint.pprint(cfg)

    visualize(
      tfrecords=args.tfrecords,
      checkpoint_path=args.checkpoint_path,
      cfg=cfg,
      savedir=args.savedir
    )