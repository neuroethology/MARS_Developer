"""
Visualize the inputs to the network, added plot of original image size and image id and filename
"""
import argparse

import inputs
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pdb

from config import parse_config_file


def visualize(tfrecords, cfg):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    # Little utility to convert the float images to uint8
    height = cfg.HEIGHT
    width = cfg.WIDTH


    # run a session to look at the images...
    with sess.as_default(), graph.as_default():
        # utility to convert image and resize to original size
        image_to_resize = tf.placeholder(tf.float32, [cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
        resize_image_op = tf.image.resize_images(image_to_resize, [height, width])
        convert_image_to_uint8 = tf.image.convert_image_dtype(tf.add(tf.div(resize_image_op, 2.0), 0.5), tf.uint8)

        # Input Nodes
        images, batched_bboxes, batched_num_bboxes, image_ids= inputs.input_nodes(
            tfrecords=tfrecords,
            max_num_bboxes=cfg.MAX_NUM_BBOXES,
            num_epochs=None,
            batch_size=cfg.BATCH_SIZE,
            num_threads=cfg.NUM_INPUT_THREADS,
            add_summaries=True,
            shuffle_batch=False,
            cfg=cfg
        )
        coord = tf.train.Coordinator()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        plt.ion()
        done = False
        fig = plt.figure(frameon=False)

        while not done:
            output = sess.run([images, batched_bboxes,image_ids])
            #output[0] = images [ex_per_batch, h,w,channels]
            #output[1] = bboxes [ex_per_batch,num_boxes,num_coord]
            for image, bboxes, image_id in zip(output[0], output[1], output[2]):

                # pdb.set_trace()
                res_image = sess.run(convert_image_to_uint8, {image_to_resize: image})

                ax = plt.axes([0.0, 0.0, 1.0, 1.0])
                ax.set_axis_off()
                ax.axes.get_yaxis().set_visible(False)
                ax.axes.get_xaxis().set_visible(False)
                fig.add_axes(ax)
                plt.imshow(res_image)
                plt.text(10,40,'ID = ' + image_id,fontsize=12, color = 'r')
                # plt.title(path + '\n' + filename,loc='left',fontsize = 12)

                # plot the ground truth bounding boxes
                for bbox in bboxes:
                    xmin, xmax = bbox[[0,2]]*width
                    ymin,ymax = bbox[[1,3]]*height

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