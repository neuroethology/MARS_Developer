"""
Visualize the training inputs to the network. 
"""

import argparse
import logging
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
import tensorflow as tf


from config import parse_config_file
import train_inputs

def create_solid_rgb_image(shape, color):
  image = np.zeros(shape, np.uint8)
  image[:] = color
  return image


def visualize(tfrecords, cfg):
  
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  graph = tf.Graph()
  sess = tf.Session(graph = graph)
  
  num_parts = cfg.PARTS.NUM_PARTS

  # run a session to look at the images...
  with sess.as_default(), graph.as_default():

    # Get the input nodes.
    input_nodes = train_inputs.input_nodes

    # Actually get the input nodes.
    batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps = input_nodes(
      tfrecords,
      num_epochs=None,
      add_summaries = True,
      cfg=cfg
    )

    # Create a placeholder float image.
    image_to_convert = tf.placeholder(tf.float32)

    # Create an op that converts that float image to a uint8.
    convert_to_uint8 = tf.image.convert_image_dtype(tf.add(tf.div(image_to_convert, 2.0), 0.5), tf.uint8)

    # Create a placeholder float image.
    image_to_resize = tf.placeholder(tf.float32)
    # Create an op that resizes that float image to the size that you'd put into the model.
    resize_to_input_size = tf.image.resize_bilinear(image_to_resize, size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE])

    coord = tf.train.Coordinator()
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    num_part_cols = 3
    num_part_rows = int(np.ceil(num_parts / (num_part_cols * 1.)))

    plt.ion()
    r = ""
    while r == "":
      outputs = sess.run([batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps])
      
      for b in xrange(cfg.BATCH_SIZE):
        # Create the figure we'll plot the images on.
        image_figure = plt.figure("Image")
        image_figure.clear()

        # Get the image.
        image = outputs[0][b]
        # Convert the image to uint8.
        uint8_image = sess.run(convert_to_uint8, {image_to_convert : image})
        # Display the image in the current figure.
        plt.imshow(uint8_image)

        # Get the keypoints' x and y locations.
        parts = outputs[2][b]
        # Get the visibilities of those points.
        part_visibilities = outputs[3][b]

        # For each point, if it's visible, then plot it.
        for p in range(num_parts):
          if part_visibilities[p] > 0:
            idx = 2*p
            x, y = parts[idx:idx+2] * float(cfg.INPUT_SIZE) 
            plt.plot(x, y, color=cfg.PARTS.COLORS[p], marker=cfg.PARTS.SYMBOLS[p], label=cfg.PARTS.NAMES[p])
        
        # Get the heatmaps.
        heatmaps = outputs[1][b]
        # Plot each heatmap.
        for p in range(num_parts):
          heatmap = heatmaps[:,:,p]
          print "%s : max %0.3f, min %0.3f" % (cfg.PARTS.NAMES[p], np.max(heatmap), np.min(heatmap))

        # Cap the heatmaps activation range at 0 and 1.
        heatmaps = np.clip(heatmaps, 0., 1.)
        heatmaps = np.expand_dims(heatmaps, 0)
        # Resize all the heatmaps.
        resized_heatmaps = sess.run(resize_to_input_size, {image_to_resize : heatmaps})
        resized_heatmaps = np.squeeze(resized_heatmaps)

        # Get the background heatmaps.
        background_heatmaps = outputs[5][b]
        resized_background_heatmaps = sess.run(resize_to_input_size, {image_to_resize : np.expand_dims(background_heatmaps, 0)})
        resized_background_heatmaps = np.squeeze(resized_background_heatmaps)

        heatmaps_figure = plt.figure("Heatmaps")
        heatmaps_figure.clear()

        # Now, plot all the heatmaps as overlays on the image.
        for p in range(num_parts):
          # Get the heatmap for a given part.
          heatmap = resized_heatmaps[:,:,p]

          heatmaps_figure.add_subplot(num_part_rows, num_part_cols, p+1)
          plt.imshow(uint8_image)
 
          # Rescale the values of the heatmap.
          f = interpolate.interp1d([np.min(heatmap), np.max(heatmap)], [0, 255])
          int_heatmap = f(heatmap).astype(np.uint8)

          # Add the heatmap as an alpha channel over the image
          blank_image = create_solid_rgb_image(image.shape, [255, 0, 0])
          heat_map_alpha = np.dstack((blank_image, int_heatmap))
          plt.imshow(heat_map_alpha)
          plt.axis('off')
          plt.title(cfg.PARTS.NAMES[p])
        

        # Show the background heatmaps with the ground truth part location
        background_heatmaps_figure = plt.figure("Background Heatmaps")
        background_heatmaps_figure.clear()

        # Now plot all the background heatmaps.
        for p in range(num_parts):
          background_heatmaps_figure.add_subplot(num_part_rows, num_part_cols, p+1)
          
          background_heatmap = resized_background_heatmaps[:,:,p]
          plt.imshow(background_heatmap)

          if part_visibilities[p] > 0:
            idx = 2*p
            x, y = parts[idx:idx+2] * float(cfg.INPUT_SIZE) 
            plt.plot(x, y, color='pink', marker='o', label=cfg.PARTS.NAMES[p])

          plt.axis('off')
          plt.title(cfg.PARTS.NAMES[p])

        plt.show()
        r = raw_input("push button")
        plt.clf()
        if r != "":
          break


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