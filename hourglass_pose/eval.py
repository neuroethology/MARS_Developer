"""
File for detecting parts on images without ground truth.
"""
import argparse
from cStringIO import StringIO
import json
import numpy as np
import os
import pprint
import sys
import tensorflow as tf
from tensorflow.contrib import slim
import time


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging
from config import parse_config_file
import eval_inputs as inputs
import model
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import pdb


def get_local_maxima(data, x_offset, y_offset, input_width, input_height, image_width, image_height, threshold=0.000002,
                     neighborhood_size=15):
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
    for dy, dx in slices:
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


def eval(tfrecords, checkpoint_path, summary_dir, max_iterations, cfg):

  # Set the logging level.
  logging.getLogger("tensorflow").setLevel(logging.ERROR)

  # Initialize the graph.
  graph = tf.Graph()

  with graph.as_default():
    
    batched_images, batched_bboxes, batched_parts, batched_part_visibilities, batched_image_ids, batched_image_height_widths, batched_crop_bboxes = inputs.input_nodes(
      tfrecords=tfrecords,
      num_parts = cfg.PARTS.NUM_PARTS,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity = cfg.QUEUE_CAPACITY,
      shuffle_batch=True,
      cfg=cfg
    )

    # Set batch_norm parameters.
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

    # Set parameters for the EMA.
    ema = tf.train.ExponentialMovingAverage(
      decay=cfg.MOVING_AVERAGE_DECAY
    )   
    shadow_vars = {
      ema.average_name(var) : var
      for var in slim.get_model_variables()
    }

    saver = tf.train.Saver(shadow_vars, reshape=True)

    # Set up a node to fetch values from when we run the graph.
    fetches = [predicted_heatmaps[-1],
               batched_bboxes,
               batched_parts,
               batched_part_visibilities,
               batched_image_ids,
               batched_image_height_widths,
               batched_crop_bboxes]

    # Create a training coordinator that will control the different threads.
    coord = tf.train.Coordinator()

    # Set up GPU according to config.
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

      # Initialize all the variables.
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      
      # Launch the queue runner threads
      threads = tf.train.start_queue_runners(sess=session, coord=coord)

      dataset_image_ids = set()
      gt_annotations = []
      pred_annotations = []
      gt_annotation_id = 1
      gt_image_id = 1
      try:
        
        if tf.gfile.IsDirectory(checkpoint_path):
          print(checkpoint_path)
          checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
          print(''.join(['is now:', checkpoint_path]))

        if checkpoint_path is None:
          print "ERROR: No checkpoint file found."
          return

        # Restores from checkpoint
        saver.restore(session, checkpoint_path)

        # Assuming the model_checkpoint_path looks something like:
        #   /my-favorite-path/model.ckpt-0,
        # extract global_step from it.
        global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
        print "Found model for global step: %d" % (global_step,)


        step = 0
        print_str = ', '.join([
          'Step: %d',
          'Time/image network (ms): %.1f'
        ])
        while not coord.should_stop():
          t = time.time()
          outputs = session.run(fetches)
          dt = time.time() - t

          for b in range(cfg.BATCH_SIZE):

            heatmaps = outputs[0][b]
            bbox = outputs[1][b]
            parts = outputs[2][b]
            part_visibilities = outputs[3][b]
            image_id = outputs[4][b]
            image_height_widths = outputs[5][b]
            crop_bboxes = outputs[6][b]

             # Transform the keypoints back to the original image space.
            image_height, image_width = image_height_widths
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bboxes
            crop_w, crop_h = np.array([crop_x2 - crop_x1, crop_y2 - crop_y1]) * np.array([image_width, image_height], dtype=np.float32)

            if cfg.LOOSE_BBOX_CROP:
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

            selected_scores = []
            pred_parts = []
            for k in keypoints:
              # Each keypoint prediction may actually have multiple prediction loci
              # --pick the one with the highest score.
              s_idx = np.argsort(k['score']).tolist()
              s_idx.reverse()

              if len(s_idx) == 0:
                x = 0
                y = 0
                v = 0
              else:
                # print k
                x = k['x'][s_idx[0]] * image_width
                y = k['y'][s_idx[0]] * image_height
                v = 1
                selected_scores.append(k['score'][s_idx[0]])
              # Store the predicted parts in a list.
              pred_parts += [x, y, v]
            # avg_score = np.mean(selected_scores)
            # Store the results
            pred_annotations.append({
              'image_id' : gt_image_id,
              'keypoints' : pred_parts,
              'score' : 1.,#avg_score,
              'category_id' : 1
            })

            # Convert the ground-truth parts to unnormalized coords.
            gt_parts_x = parts[0::2] * image_width
            gt_parts_y = parts[1::2] * image_height

            # Put them in the format we need for COCO evaluation.
            gt_parts = np.transpose(np.vstack([gt_parts_x, gt_parts_y, part_visibilities]), [1, 0])
            gt_parts = gt_parts.ravel().tolist()

            # Get the bbox coordinates again.
            x1, y1, x2, y2 = bbox * np.array([image_width, image_height, image_width, image_height])
            w = x2 - x1
            h = y2 - y1

            gt_annotations.append({
              "id" : gt_annotation_id,
              "image_id" : gt_image_id,
              "category_id" : 1,
              "area" : w * h,
              "bbox" : [x1, y1, w, h],
              "iscrowd" : 0,
              "keypoints" : gt_parts,
              "num_keypoints" : np.sum(part_visibilities)
            })

            dataset_image_ids.add(gt_image_id)

            gt_annotation_id += 1
            gt_image_id += 1



          print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000)
          step += 1

          if max_iterations > 0 and step == max_iterations:
              break

      except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)

      # When done, ask the threads to stop. Requesting stop twice is fine.
      coord.request_stop()
      # Wait for them to stop before preceding.
      coord.join(threads)

      gt_dataset = {
        'annotations' : gt_annotations,
        'images' : [{'id' : img_id} for img_id in dataset_image_ids],
        'categories' : [{ 'id' : 1 }]
      }

      # Parse things for COCO evaluation.
      gt_coco = COCO()
      gt_coco.dataset = gt_dataset
      gt_coco.createIndex()

      # Actually perform the evaluation.
      pred_coco = gt_coco.loadRes(pred_annotations)
      cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints')

      cocoEval.evaluate()
      cocoEval.accumulate()

      # Have COCO output summaries of the evaluation.
      old_stdout = sys.stdout
      sys.stdout = captured_stdout = StringIO()
      cocoEval.summarize()
      sys.stdout = old_stdout

      # Store the output as a TF summary, so we can view it in Tensorboard.
      summary_op = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(summary_dir)
      summary = tf.Summary()
      summary.ParseFromString(session.run(summary_op))

      # Since we captured the stdout while we were outputting commandline summaries, we can parse that for our TF
      # summaries right now.
      for line in captured_stdout.getvalue().split('\n'):
        if line != "":
          description, score = line.rsplit("=", 1)
          description = description.strip()
          score = float(score)

          summary.value.add(tag=description, simple_value=score)

          print "%s: %0.3f" % (description, score)
      
      summary_writer.add_summary(summary, global_step)
      summary_writer.flush()
      summary_writer.close()


def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=False, default=None)
                          
    parser.add_argument('--summary_dir', dest='summary_dir',
                        help='Path to the directory where the results will be saved',
                        required=True, type=str)
                        
    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run. Set to 0 to run on all records.',
                        required=False, type=int, default=0)

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    args = parse_args()
    cfg = parse_config_file(args.config_file)

    eval(
      tfrecords=args.tfrecords,
      checkpoint_path=args.checkpoint_path,
      summary_dir = args.summary_dir,
      max_iterations = args.max_iterations,
      cfg=cfg
    )