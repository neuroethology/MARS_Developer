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

def detect_visualize(tfrecords, bbox_priors, checkpoint_path, cfg, save_res, save_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    graph = tf.Graph()
    
    if not os.path.exists(save_res):
        os.makedirs(save_res)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Force all Variables to reside on the CPU.
    with graph.as_default():

        batched_images, batched_offsets, batched_dims, batched_is_flipped, batched_bbox_restrictions, batched_max_to_keep, batched_heights_widths, batched_image_ids, batched_image_filenames, batched_image_paths = input_nodes(
            tfrecords=tfrecords,
            num_epochs=1,
            batch_size=cfg.BATCH_SIZE,
            num_threads=cfg.NUM_INPUT_THREADS,
            capacity=cfg.QUEUE_CAPACITY,
            cfg=cfg
        )

        batch_norm_params = {
            # Decay for the batch_norm moving averages.
            'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            'variables_collections': [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
            'is_training': False
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            biases_regularizer=slim.l2_regularizer(0.00004)):

            locations, confidences, inception_vars = model.build(
                inputs=batched_images,
                num_bboxes_per_cell=cfg.NUM_BBOXES_PER_CELL,
                reuse=False,
                scope=''
            )

        ema = tf.train.ExponentialMovingAverage(
            decay=cfg.MOVING_AVERAGE_DECAY
        )
        shadow_vars = {
            ema.average_name(var): var
            for var in slim.get_model_variables()
            }

        # Restore the parameters
        saver = tf.train.Saver(shadow_vars, reshape=True)

        fetches = [locations, confidences, batched_offsets, batched_dims, batched_is_flipped,
                   batched_bbox_restrictions, batched_max_to_keep, batched_heights_widths,
                   batched_image_ids, batched_images, batched_image_filenames, batched_image_paths]

        coord = tf.train.Coordinator()

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

        #utility to measure luminance
        crop = tf.placeholder(tf.float32,[None, None, 3])
        lum_crop = tf.div(tf.reduce_sum(crop, [0,1,2]), 3.0)

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

                plt.ioff()
                fig = plt.figure(frameon=False)

                step = 0
                while not coord.should_stop():

                    t = time.time()
                    outputs = sess.run(fetches)
                    dt = time.time() - t

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
                    image_filenames = outputs[10]
                    image_paths = outputs[11]

                    for b in range(cfg.BATCH_SIZE):

                        img_id = int(np.asscalar(image_ids[b]))
                        img_filename = image_filenames[b][0]
                        img_path  = image_paths[b][0]

                        predicted_bboxes = locs[b] + bbox_priors
                        predicted_bboxes = np.clip(predicted_bboxes, 0., 1.)
                        predicted_confs = confs[b]

                        # Keep only the predictions that are completely contained in the [0.1, 0.1, 0.9, 0.9] square
                        # for this patch
                        filtered_bboxes, filtered_confs = filter_proposals(predicted_bboxes, predicted_confs,
                                                                           patch_bbox_restrictions[b])

                        # No valid predictions?
                        if filtered_bboxes.shape[0] == 0:
                            continue

                        # Lets get rid of some of the predictions
                        num_preds_to_keep = np.asscalar(patch_max_to_keep[b])
                        sorted_idxs = np.argsort(filtered_confs.ravel())[::-1]
                        sorted_idxs = sorted_idxs[:num_preds_to_keep]
                        filtered_bboxes = filtered_bboxes[sorted_idxs]
                        filtered_confs = filtered_confs[sorted_idxs]                       

                        # Convert the bounding boxes to the original image dimensions
                        converted_bboxes = convert_proposals(
                            bboxes=filtered_bboxes,
                            offset=patch_offsets[b],
                            patch_dims=patch_dims[b],
                            image_dims=image_height_widths[b],
                            is_flipped=patch_is_flipped[b]
                        )

                        #to reshape to original size
                        uint8_image = sess.run(convert_image_to_uint8, {image_to_resize: images[b]})
                        ax = plt.axes([0.0, 0.0, 1.0, 1.0])
                        ax.set_axis_off()
                        ax.axes.get_yaxis().set_visible(False)
                        ax.axes.get_xaxis().set_visible(False)
                        fig.add_axes(ax)
                        plt.imshow(uint8_image)
                        plt.text(0,-10,'ID = ' + str(img_id), fontsize = 12, color='r')
                        num_detections_to_render = min(converted_bboxes.shape[0], 4)
                        xdt=35

                        for i in range(num_detections_to_render):
                            loc = converted_bboxes[i].ravel()
                            conf = filtered_confs[i]

                            if conf > 0.2:
                                # Plot the predicted location in red
                                xmin, xmax = loc[[0,2]]*width
                                ymin, ymax = loc[[1,3]]*height

                                plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
                                plt.text(10, -10-xdt, 'conf bbox = %d: %f' % (i,conf) , fontsize=12, color='r')
                                plt.text(xmin-10,ymin-10, str(i), fontsize = 12, color='r')
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

    parser.add_argument('--priors', dest='priors',
                        help='path to the bounding box priors pickle file', type=str,
                        required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                        help='Either a path to a specific model, or a path to a directory where checkpoint files are stored. If a directory, the latest model will be tested against.',
                        type=str,
                        required=True, default=None)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
   	
    parser.add_argument('--save_res', dest='save_res',
                  help='Path to the save the detection results',
                  required=True, type=str)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='Path to the save the proposed detection on test image',
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
        cfg=cfg,
        save_res = args.save_res,
        save_dir = args.save_dir,
    )


if __name__ == '__main__':
    main()
