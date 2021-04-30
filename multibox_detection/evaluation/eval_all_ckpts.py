"""
Use the COCO evaluation pipeline.
"""

import argparse
import pickle
import logging
import pdb
import pprint
import sys
import time
import os
from io import StringIO
# from cStringIO import StringIO

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

sys.path.insert(0, os.path.abspath('../..'))
sys.path.append('./')
sys.path.append('./evaluation/')

from multibox_detection.loss import *
import multibox_detection.model_detection as model
from multibox_detection.config import parse_config_file
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import eval_inputs as inputs

from os import listdir
from os.path import isfile, join

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def eval(tfrecords, bbox_priors, summary_dir, checkpoints, checkpoint_dir, max_iterations, cfg):
  
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()

  # Force all Variables to reside on the CPU.
  with graph.as_default():

    batched_images, batched_bboxes, batched_num_bboxes, batched_areas, batched_image_ids = inputs.input_nodes(
      tfrecords=tfrecords,
      max_num_bboxes = cfg.MAX_NUM_BBOXES,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity=cfg.QUEUE_CAPACITY,
      shuffle_batch=True,
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

    fetches = [locations, confidences, batched_bboxes, batched_num_bboxes, batched_areas, batched_image_ids]
    
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

    dataset_image_ids = set()
    gt_annotations = []
    pred_annotations = [] # {imageID,x1,y1,w,h,score,class}
    gt_annotation_id = 1

    with sess.as_default():

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      try:
        losses = dict().fromkeys(checkpoints)
        for ckpt in checkpoints:
          checkpoint_path = checkpoint_dir + ckpt

          # Restores from checkpoint
          saver.restore(sess, checkpoint_path)
          # Assuming model_checkpoint_path looks something like:
          #   /my-favorite-path/cifar10_train/model.ckpt-0,
          # extract global_step from it.
          global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
          print("Found model for global step: {:d}".format(global_step))
        
          print_str = ', '.join([
            'Step: {:d}',
            'Time/image (ms): {:.1f}'
          ])

          step = 0
          total_loss = 0
          while not coord.should_stop():
          # for step in range(100):
            t = time.time()
            outputs = sess.run(fetches)
            dt = time.time()-t
          
            locs = outputs[0]
            confs = outputs[1]
            all_gt_bboxes = outputs[2]
            all_gt_num_bboxes = outputs[3]
            all_gt_areas = outputs[4]
            image_ids = outputs[5]

            batch_total_loss = 0
            for b in range(cfg.BATCH_SIZE):

              img_id = int(image_ids[b])
            
              predicted_bboxes = locs[b] + bbox_priors
              predicted_bboxes = np.clip(predicted_bboxes, 0., 1.)
              predicted_confs = confs[b]

              gt_bboxes = all_gt_bboxes[b]
              gt_num_bboxes = all_gt_num_bboxes[b]
              gt_areas = all_gt_areas[b]
             
              # Compute the loss 
              location_loss, confidence_loss =  add_loss(locs, confs, all_gt_bboxes, all_gt_num_bboxes, bbox_priors, cfg.LOCATION_LOSS_ALPHA)
              batch_total_loss += location_loss.eval() + confidence_loss.eval()
              
              # Scale the predictions and ground truth boxes
              # GVH: Should check to see if we are preserving aspect ratio or not...
              im_scale = np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE])
              predicted_bboxes = predicted_bboxes * im_scale
              gt_bboxes = gt_bboxes * im_scale
            
              # Sort the predictions based on confidences
              sorted_idxs = np.argsort(predicted_confs.ravel())[::-1]
              sorted_bboxes = predicted_bboxes[sorted_idxs]
              sorted_confs = predicted_confs[sorted_idxs]

              # Store the results
              for k in range(100):
                x1, y1, x2, y2 = sorted_bboxes[k]
                score = sorted_confs[k]
                pred_annotations.append([
                  img_id,
                  x1, y1, x2 - x1, y2 - y1, 
                  score[0],
                  1
                ])
              for k in range(gt_num_bboxes):  
                x1, y1, x2, y2 = gt_bboxes[k]
                w = x2 - x1
                h = y2 - y1
                gt_annotations.append({
                  "id" : gt_annotation_id,
                  "image_id" : img_id,
                  "category_id" : 1,
                  "area" : gt_areas[k],
                  "bbox" : [x1, y1, w, h],
                  "iscrowd" : 0,
                })
                gt_annotation_id += 1
              dataset_image_ids.add(img_id)
            
            total_loss += batch_total_loss / cfg.BATCH_SIZE
            step += 1
            print(print_str.format(step, (dt / cfg.BATCH_SIZE) * 1000))

            if max_iterations > 0 and step == max_iterations:
              break  
                  
          losses[ckpt] = total_loss / step
        
      except tf.errors.OutOfRangeError as e:
        pass
        
      coord.request_stop()
      coord.join(threads)

      pred_annotations = np.array(pred_annotations)

        #print(pred_annotations.shape, gt_annotations[0].shape)
        #stop
      gt_dataset = {
        'annotations' : np.array(gt_annotations),
        'images' : np.array([{'id' : img_id} for img_id in dataset_image_ids]),
        'categories' : [{ 'id' : 1 }]
      }
      #initialize COCO GT api
      gt_coco = COCO()
      gt_coco.dataset = gt_dataset
      gt_coco.createIndex()
      #initialize COCO detections api: output creating index and index created
      pred_coco = gt_coco.loadRes(pred_annotations)
      #running evaluation
      cocoEval = COCOeval(gt_coco, pred_coco, iouType='bbox')

      cocoEval.params.useCats = 0
      #cocoEval.params.areaRange = ("medium","large") # I just created a different gt annotation file
      cocoEval.evaluate() #
      cocoEval.accumulate() #

      old_stdout = sys.stdout
      sys.stdout = captured_stdout = StringIO()
      cocoEval.summarize() # print results
      sys.stdout = old_stdout

      summary_op = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(summary_dir)
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))

      with open(summary_dir + 'cocoEval.pkl', 'wb') as fp:
        pickle.dump(cocoEval, fp)

      for line in captured_stdout.getvalue().split('\n'):
        if line != "":
          description, score = line.rsplit("=", 1)
          description = description.strip()
          score = float(score)

          summary.value.add(tag=description, simple_value=score)

          print(f"{description:s}: {score:0.3f}")
      
      summary_writer.add_summary(summary, global_step)
      summary_writer.flush()
      summary_writer.close()

      return losses

def parse_args():

    parser = argparse.ArgumentParser(description='Detect objects using a pretrained Multibox model')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)
    
    parser.add_argument('--priors', dest='priors',
                          help='path to the bounding box priors pickle file', type=str,
                          required=True)
    
    parser.add_argument('--summary_dir', dest='summary_dir',
                          help='path to directory to store summary files and checkpoint files', type=str,
                          required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to a directory where checkpoint files are stored. This will find every model checkpoint in the directory and evaluate it on the provided validation set.', 
                          type=str, required=True, default=None)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run. Set to 0 to run on all records.',
                        required=False, type=int, default=0)              

    args = parser.parse_args()
    
    return args

def main():
  args = parse_args()
  print("Command line arguments:")
  pprint.pprint(vars(args))
  print

  cfg = parse_config_file(args.config_file)
  print("Configurations:")
  pprint.pprint(cfg)
  print 
    
  with open(args.priors, 'rb') as f:
    bbox_priors = pickle.load(f, encoding='latin1')
  bbox_priors = np.array(bbox_priors).astype(np.float32)

  # Retrieve only the checkpoint files from the provided dir
  onlyfiles = [f for f in listdir(args.checkpoint_path) if isfile(join(args.checkpoint_path, f))]
  onlyckpts = sorted(['.'.join(f.split('.')[:-1]) for f in onlyfiles if 'meta' in f])

  losses = eval(
    tfrecords=args.tfrecords,
    bbox_priors=bbox_priors,
    summary_dir=args.summary_dir,
    checkpoints=onlyckpts,
    checkpoint_dir=args.checkpoint_path,
    max_iterations = args.max_iterations,
    cfg=cfg
  )
 
  best_ckpt = min(losses, key=losses.get) 
  xs = [int(st.split('-')[-1]) for st in losses.keys()]
  ys = [st for st in losses.values()]
  plt.plot(xs, ys)
  plt.scatter(xs, ys, c='b', label='loss at checkpoint')
  plt.scatter(x=int(best_ckpt.split('-')[-1]), y=losses[best_ckpt], c='g', label='best checkpoint')
  plt.title('Loss over all checkpoints')
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss on eval set')
  plt.show() 
  print('stop here')


if __name__ == '__main__':
  main()
