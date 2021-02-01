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
import sys
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.util import deprecation
import time
import cv2


from config import parse_config_file
from detect import get_local_maxima
import detect_inputs_imsize as inputs
import model_pose
import pdb

deprecation._PRINT_DEPRECATION_WARNINGS = False


def detect(tfrecords, checkpoint_path, cfg, save_dir):

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

  graph = tf.Graph()
  
  with graph.as_default():
    
    batched_images, batched_bboxes, batched_scores, batched_image_ids, batched_labels, batched_image_height_widths, batched_crop_bboxes, batched_ori_images = inputs.input_nodes(
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
        'variables_collections' : [tf.compat.v1.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training' : False
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
      
      predicted_heatmaps = model_pose.build(
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

    saver = tf.compat.v1.train.Saver(shadow_vars, reshape=True)
    
    fetches = [batched_images, predicted_heatmaps[-1], batched_bboxes, batched_scores, batched_image_ids, batched_labels, batched_image_height_widths, batched_crop_bboxes, batched_ori_images]

    # Now create a training coordinator that will control the different threads
    coord = tf.train.Coordinator()
    
    sess_config = tf.compat.v1.ConfigProto(
      log_device_placement=False,
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.compat.v1.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )
    session = tf.compat.v1.Session(graph=graph, config=sess_config)
    
    with session.as_default():

      # make sure to initialize all of the variables
      tf.compat.v1.global_variables_initializer().run()
      tf.compat.v1.local_variables_initializer().run()
      
      # launch the queue runner threads
      threads = tf.train.start_queue_runners(sess=session, coord=coord)
      
      try:
        
        if tf.io.gfile.isdir(checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        
        if checkpoint_path is None:
          print("ERROR: No checkpoint file found.")
          return

        # Restores from checkpoint
        saver.restore(session, checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
        print("Found model for global step: %d" % (global_step,))
        
        # we will store results into a tfrecord file
        output_writer_iteration = 0
        #output_path = os.path.join(save_dir, 'heatmap_results-%d-%d.tfrecords' % (global_step, output_writer_iteration))
        #output_writer = tf.python_io.TFRecordWriter(output_path)
        
        # plt.ion()

        image_to_convert = tf.compat.v1.placeholder(tf.float32)
        convert_to_uint8 = tf.compat.v1.image.convert_image_dtype(tf.add(tf.div(image_to_convert, 2.0), 0.5), tf.uint8)
        
        image_to_resize = tf.compat.v1.placeholder(tf.float32)
        resize_to_input_size = tf.compat.v1.image.resize_bilinear(image_to_resize, size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE])

        done = False
        step = 0
        print_str = ', '.join([
          'Step: %d',
          'Time/image network (ms): %.1f',
          'Time/image post proc (ms): %.1f'
        ])

        fig = plt.figure(figsize=(float(cfg.WIDTH) / 100.0, float(cfg.HEIGHT) / 100.0), dpi=100,
                         frameon=False)
        fig.patch.set_visible(False)
        mi = ['k','w']
        c = np.array([['b','r','g','orange','r','g','c','r','g','r','g'],
                      ['#21a5f2','#ed42b9','#46db41','#ef571a','#ed42b9','#46db41','#35fbff','#ed42b9','#46db41','#ed42b9','#46db41']])
        sk = np.array([['#cd87ff','#74c8f9','#feff95'],['#ae16f9','#21a5f2','#f99c4a']])
        ec = np.array([['#cd87ff','#cd87ff','#cd87ff','#74c8f9','#74c8f9','#feff95','#feff95','#74c8f9','#74c8f9','#feff95','#feff95',],
                       ['#ae16f9', '#ae16f9', '#ae16f9', '#21a5f2', '#21a5f2', '#f99c4a', '#f99c4a','#21a5f2', '#21a5f2', '#f99c4a', '#f99c4a']])

        while not coord.should_stop() and not done:

          t = time.time()
          outputs = session.run(fetches)
          dt = time.time() - t
          t = time.time()
          
          for b in range(cfg.BATCH_SIZE):

            image = outputs[0][b]
            heatmaps = outputs[1][b]
            bbox = outputs[2][b]
            score = outputs[3][b]
            image_id = outputs[4][b]
            image_height_widths = outputs[6][b]
            ori_image = outputs[8][b]

            if b == 0:
              ori_image = session.run(convert_to_uint8, {image_to_convert: ori_image})
              # We need to transform the keypoints back to the original image space.
              image_height, image_width = image_height_widths
              ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
              ax.set_xlim([0, cfg.WIDTH])
              ax.set_ylim([cfg.HEIGHT, 0])
              ax.axes.get_yaxis().set_visible(False)
              ax.axes.get_xaxis().set_visible(False)
              ax.axison = False
              ax.set_adjustable('box')
              plt.imshow(ori_image)

            ax.plot(np.NaN, np.NaN, '*', color=mi[b], markeredgecolor='k',markersize=20,
                    label=r"$\bf{Resident}$" if b == 0 else r"$\bf{Intruder}$")

            image_height, image_width = image_height_widths

            int8_image = session.run(convert_to_uint8, {image_to_convert : image})

            heatmaps = np.clip(heatmaps, 0., 1.)
            heatmaps = np.expand_dims(heatmaps, 0)
            resized_heatmaps = session.run(resize_to_input_size, {image_to_resize: heatmaps})
            resized_heatmaps = np.squeeze(resized_heatmaps)

            # Unpack the bboxes and rescale them from norm coordinates to image coordinates.
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
            bbox_w = (bbox_x2 - bbox_x1) * image_width
            bbox_h = (bbox_y2 - bbox_y1) * image_height

            if cfg.LOOSE_BBOX_CROP:

              # Now resize the heatmaps to the original bbox size.
              rescaled_heatmaps = cv2.resize(resized_heatmaps, (int(np.round(bbox_w)), int(np.round(bbox_h))),
                                             interpolation=cv2.INTER_LINEAR)

              keypoints = np.zeros((cfg.PARTS.NUM_PARTS, 3))
              # For each part-heatmap, extract out the keypoint, then place it in the original image's coordinates.
              for j in range(cfg.PARTS.NUM_PARTS):

                # Get the current heatmap.
                hm = rescaled_heatmaps[:, :, j]
                score_pred = float(np.max(hm))

                # Extract out the keypoint.
                x, y = np.array(np.unravel_index(np.argmax(hm), hm.shape)[::-1])

                # Place it in the original image's coordinates.
                imx = x + bbox_x1 * image_width
                imy = y + bbox_y1 * image_height

                # Store it.
                keypoints[j, :] = [imx, imy, score_pred]
            else:

              if bbox_h > bbox_w:
                new_height = cfg.INPUT_SIZE
                height_factor = float(1.0)
                width_factor = new_height / float(bbox_h)
                new_width =int(np.round(bbox_w * width_factor))
                im_scale = width_factor
              else:
                new_width = cfg.INPUT_SIZE
                width_factor = float(1.0)
                height_factor = new_width / float(bbox_w)
                new_height = int(np.round(bbox_h * height_factor))
                im_scale = height_factor

              # plt.plot([bbox_x1*cfg.WIDTH, bbox_x2*cfg.WIDTH, bbox_x2*cfg.WIDTH, bbox_x1*cfg.WIDTH, bbox_x1*cfg.WIDTH],[bbox_y1*cfg.HEIGHT, bbox_y1*cfg.HEIGHT, bbox_y2*cfg.HEIGHT, bbox_y2*cfg.HEIGHT, bbox_y1*cfg.HEIGHT], 'r-',lw=3)
              keypoints = keypoints2 = np.zeros((cfg.PARTS.NUM_PARTS,3))
              for j in range(cfg.PARTS.NUM_PARTS):

                heatmap = resized_heatmaps[:,:,j]

                #back to original image coordinates
                # imx = int(x/float(new_width) * bbox_w)  + bbox_x1 * image_width
                # imy = int(y/float(new_height) * bbox_h ) + bbox_y1 * image_height
                # keypoints2[j,:]=[imx,imy,score_pred]
                # plt.plot(imx,imy,color='r', marker=cfg.PARTS.SYMBOLS[j], markersize= ms)

                heatmap_resized = heatmap[:new_height,:new_width]
                heatmap_resized  = cv2.resize(heatmap_resized,(bbox_w, bbox_h),interpolation = cv2.INTER_LINEAR)
                score_pred_res = np.max(heatmap_resized)
                x, y = np.array(np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)[::-1])
                imx = x + bbox_x1 * image_width
                imy = y + bbox_y1 * image_height
                keypoints[j,:]=[imx,imy,score_pred_res]
                #plot detection
            # plot skeleton
            xs = keypoints[:,0]; ys = keypoints[:,1]; ss = keypoints[:,2]
            plt.plot([xs[0], xs[2]], [ys[0], ys[2]], color = sk[b,0],lw = 1. if np.min([ss[0],ss[2]]) < 0.5 else 3.)  # right head
            plt.plot([xs[0], xs[1]], [ys[0], ys[1]], color = sk[b,0],lw = 1. if np.min(ss[0:2]) < 0.5 else 3.)  # left head
            plt.plot([xs[1], xs[2]], [ys[1], ys[2]], color = sk[b,0],lw = 1.  if np.min(ss[1:3]) < 0.5 else 3.) #neck
            plt.plot([xs[3], xs[4]], [ys[3], ys[4]], color = sk[b,1],lw = 1. if np.min([ss[3],ss[4]]) < 0.5 else 3.) #left side
            plt.plot([xs[4], xs[6]], [ys[4], ys[6]], color = sk[b,2],lw = 1. if np.min([ss[4], ss[6]]) < 0.5 else 3.) #left side
            plt.plot([xs[3], xs[5]], [ys[3], ys[5]], color = sk[b,1],lw = 1. if np.min([ss[3],ss[5]]) < 0.5 else 3.) #right side
            plt.plot([xs[5], xs[6]], [ys[5], ys[6]], color = sk[b,2],lw = 1. if np.min([ss[5],ss[6]]) < 0.5 else 3.) #right side
            if len(xs)>7:
              plt.plot([xs[3], xs[7]], [ys[3], ys[7]], color=sk[b, 1], lw=1. if np.min([ss[3], ss[7]]) < 0.5 else 3.)  # right paw front
              plt.plot([xs[3], xs[8]], [ys[3], ys[8]], color=sk[b, 1], lw=1. if np.min([ss[3], ss[8]]) < 0.5 else 3.)  # left paw front
              plt.plot([xs[4], xs[9]], [ys[4], ys[9]], color=sk[b, 2], lw=1. if np.min([ss[4], ss[9]]) < 0.5 else 3.)  # rigth paw rear
              plt.plot([xs[5], xs[10]], [ys[5], ys[10]], color=sk[b, 2], lw=1. if np.min([ss[5], ss[10]]) < 0.5 else 3.)  # left paw rear

            for j in range(cfg.PARTS.NUM_PARTS):
              ms = min(np.sqrt(ss[j]) * 10.0, 10.0)
              plt.plot(xs[j], ys[j], color=c[b,j], marker=cfg.PARTS.SYMBOLS[j], label=cfg.PARTS.NAMES[j] + ' %.3f' % (ss[j]),
                       markersize=ms, markeredgecolor=ec[b,j],linestyle='None')

          leg=plt.legend(loc='upper right', fontsize='x-small',numpoints=1,markerscale=0.7,handlelength=1,handletextpad=1)
          plt.savefig(save_dir  + image_id[0].decode() + '.png', transparent=True)
          plt.savefig(save_dir  + image_id[0].decode() + '.pdf', transparent=True)

          dtt = time.time() - t
          print(print_str % (step, (dt / cfg.BATCH_SIZE) * 1000, (dtt / cfg.BATCH_SIZE) * 1000))
          step += 1
          plt.clf()

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

    parser.add_argument('--save_dir', dest='save_dir',
                        help='Path to the save directory',
                        required=True, type=str)

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    cfg = parse_config_file(args.config_file)
    save_dir = args.save_dir

    detect(
      tfrecords=args.tfrecords,
      checkpoint_path=args.checkpoint_path,
      cfg=cfg,
      save_dir=save_dir
    )
