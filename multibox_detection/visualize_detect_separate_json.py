"""
Visualize detection results.
"""
import os
import argparse
import cPickle as pickle
import logging
import pdb
import pprint
import time
import json
from json import encoder

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from matplotlib import pyplot as plt

import model
from config import parse_config_file
from detect_imsize import input_nodes, filter_proposals, convert_proposals

def detect_visualize(tfrecords, cfg, save_dir, jsonstring):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    graph = tf.Graph()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Force all Variables to reside on the CPU.
    with graph.as_default():

        batched_images, _, _, _, _, _, _, batched_image_ids, _, _ = input_nodes(
            tfrecords=tfrecords,
            num_epochs=1,
            batch_size=cfg.BATCH_SIZE,
            num_threads=cfg.NUM_INPUT_THREADS,
            capacity=cfg.QUEUE_CAPACITY,
            cfg=cfg
        )

        coord = tf.train.Coordinator()

        fetches = [batched_image_ids, batched_images]

        sess_config = tf.ConfigProto(
            log_device_placement=False,
            # device_filters = device_filters,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
            )
        )
        sess = tf.Session(graph=graph, config=sess_config)

        # Little utility to convert the float images to uint8
        width = cfg.WIDTH
        height = cfg.HEIGHT
        image_to_resize = tf.placeholder(tf.float32, [299, 299, 3])
        resize_image_op = tf.image.resize_images(image_to_resize, [height, width])
        convert_image_to_uint8 = tf.image.convert_image_dtype(tf.add(tf.div(resize_image_op, 2.0), 0.5), tf.uint8)

        # JSON
        jsonfiles = jsonstring.split( ',' )
        json_results = []
        for jf in jsonfiles:
            with open( jf, 'r' ) as f:
                res = json.load( f )
                json_results.append( res )

        with sess.as_default():

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:

                print_str = ', '.join([
                    'Step: %d',
                    'Time/image (ms): %.1f'
                ])

                plt.ioff()
                fig = plt.figure(frameon=False)

                step = 0
                while not coord.should_stop():

                    t = time.time()
                    outputs = sess.run(fetches)
                    dt = time.time() - t

                    image_ids = outputs[0]
                    images = outputs[1]

                    for b in range(cfg.BATCH_SIZE):

                        img_id = int(np.asscalar(image_ids[b]))     

                        #to reshape to original size
                        uint8_image = sess.run(convert_image_to_uint8, {image_to_resize: images[b]})
                        ax = plt.axes([0.0, 0.0, 1.0, 1.0])
                        ax.set_axis_off()
                        ax.axes.get_yaxis().set_visible(False)
                        ax.axes.get_xaxis().set_visible(False)
                        fig.add_axes(ax)
                        plt.imshow(uint8_image)
                        plt.text(0,-10,'ID = ' + str(img_id), fontsize = 12, color='k')
                        xdt=35
                        colors = ['r','g','b']

                        # Plot each bounding box
                        for j in range(len(json_results)):
                            # Check that we are not doing something weird (read: futureproof)
                            if j > len(colors):
                                print "Add more colors!"
                                exit()

                            # Find all rows that belong to this image id. This could be done way faster
                            rows = [ r for r in json_results[j] if r['image_id'] == img_id ]
                            if len(rows) == 0:
                                plt.text(10, -10-xdt, 'No bounding box in json ' + str(j) , fontsize=12, color=colors[j])
                                xdt += 40
                                continue

                            # pdb.set_trace()

                            # Find best bbox
                            for r in range(len(rows[0]['scores'])):
                                if r == 0 or rows[0]['scores'][r] > max_score:
                                    max_score = rows[0]['scores'][r]
                                    therow = r
                            loc = np.array(rows[0]['bboxes'][therow])
                            conf = rows[0]['scores'][therow]

                            # Plot the predicted location
                            xmin, xmax = loc[[0,2]]*width
                            ymin, ymax = loc[[1,3]]*height
                            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], colors[j]+'-')
                            plt.text(-10, -10-xdt, 'conf bbox = %d: %f' % (j,conf) , fontsize=12, color=colors[j])
                            plt.text(xmin-10,ymin-10, str(j), fontsize = 12, color=colors[j])
                            xdt+=40

                        plt.savefig(save_dir + str(img_id),close=True)
                        plt.savefig(save_dir + str(img_id)+'.pdf',close=True)
                        plt.clf()

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

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='Path to the save the proposed detection on test image',
                        required=True, type=str)

    parser.add_argument('--json', dest='jsonstring',
                        help='Comma-separated list of bbox detector output JSON files',
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

    detect_visualize(
        tfrecords=args.tfrecords,
        cfg=cfg,
        save_dir = args.save_dir,
        jsonstring = args.jsonstring
    )


if __name__ == '__main__':
    main()
