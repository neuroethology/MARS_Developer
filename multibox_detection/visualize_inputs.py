"""
Visualize the inputs to the network. 
"""
import argparse

import inputs
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from config import parse_config_file

def visualize(tfrecords, cfg):
  
  graph = tf.Graph()
  sess = tf.Session(graph = graph)
  
  # run a session to look at the images...
  with sess.as_default(), graph.as_default():

    # Input Nodes
    images, batched_bboxes, batched_num_bboxes, image_ids = inputs.input_nodes(
      tfrecords=tfrecords,
      max_num_bboxes = cfg.MAX_NUM_BBOXES,
      num_epochs=None,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      add_summaries = True,
      shuffle_batch=True,
      cfg=cfg
    )
    
    
    coord = tf.train.Coordinator()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    plt.ion()
    done = False
    while not done:
      
      output = sess.run([images, batched_bboxes])
      for image, bboxes in zip(output[0], output[1]):
          
          plt.imshow(((image / 2. + 0.5) * 255).astype(np.uint8))
          
          # plot the ground truth bounding boxes
          for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox * cfg.INPUT_SIZE
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')
          
          plt.show(block=False)
          
          t = raw_input("push button")
          if t != '':
            done = True
            break 
          plt.clf()


def parse_args():

    parser = argparse.ArgumentParser(description='Visualize the inputs to the multibox detection system.')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files that contain the training data', type=str,
                        nargs='+', required=True)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    args = parser.parse_args()
    return args

def main():
  args = parse_args()
  cfg = parse_config_file(args.config_file)
  visualize(
    tfrecords=args.tfrecords,
    cfg=cfg
  )

  
          
if __name__ == '__main__':
  main()