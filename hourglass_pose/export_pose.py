from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import model
from config import parse_config_file

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib
slim = tf.contrib.slim


def export(checkpoint_path, export_dir, savename, cfg):

    graph = tf.Graph()

    input_node_name = "images"
    output_node_name = None

    with graph.as_default():

        input_height = cfg.INPUT_SIZE
        input_width = cfg.INPUT_SIZE
        input_depth = 3

        # We assume that we have already preprocessed the images (i.e. extracted them from their bboxes.)
        images_bboxes = tf.placeholder(tf.float32,[None, input_height, input_width, input_depth], name=input_node_name)

        # Set activation_fn and parameters for batch_norm.
        batch_norm_params = {
            # Decay for the batch_norm moving averages.
            'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
            # epsilon to prevent 0s in variance.
            'epsilon': cfg.BATCHNORM_EPSILON,
            'variables_collections': [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
            'is_training': False
        }

        # Set various network parameters, then build the network.
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(cfg.WEIGHT_DECAY),
                            biases_regularizer=slim.l2_regularizer(cfg.WEIGHT_DECAY)) as scope:

            predicted_heatmaps = model.build(
                input=images_bboxes,
                num_parts=cfg.PARTS.NUM_PARTS
            )

        variable_averages = tf.train.ExponentialMovingAverage(cfg.MOVING_AVERAGE_DECAY)
        variables_to_restore = {
            variable_averages.average_name(var): var
            for var in slim.get_model_variables()
            }

        # Restore the checkpoint we want to export.
        if os.path.isdir(checkpoint_path):
            checkpoint_dir = checkpoint_path
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

            if checkpoint_path is None:
                raise ValueError("Unable to find a model checkpoint in the directory %s" % (checkpoint_dir,))

        tf.logging.info('Exporting model: %s' % checkpoint_path)

        saver = tf.train.Saver(variables_to_restore, reshape=True)

        # Get the graph definition.
        input_graph_def = graph.as_graph_def()  # graph used to retrieve the nodes

        # Set the session parameters.
        sess_config = tf.ConfigProto(
                log_device_placement= False,
                allow_soft_placement = True,
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
                )
            )
        sess = tf.Session(graph=graph, config=sess_config)

        # Start the session and restore the graph weights.
        with sess.as_default():
            # Initialize all variables.
            tf.global_variables_initializer().run()

            # Restore the variables from the checkpoint
            saver.restore(sess, checkpoint_path)

            # Assuming the model_checkpoint_path looks something like:
            #   /my-favorite-path/model.ckpt-0,
            # extract the global_step from it.
            global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
            print("Found model for global step: %d" % (global_step))
            # This is useful for making sure you're restoring the proper checkpoint.

            # Convert variables to constants.
            constant_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=[predicted_heatmaps[-1].name[:-2]])

            # Now optimize that constant graph so that it runs quicker.
            optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def=constant_graph_def,
                input_node_names=[input_node_name],
                output_node_names=[predicted_heatmaps[-1].name[:-2]],
                placeholder_type_enum=dtypes.float32.as_datatype_enum)

            # Make sure your saving spot actually exists.
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

            # Create the fullpath to your model.
            save_path = os.path.join(export_dir, savename + ".pb")

            # Serialize the model and write it.
            with tf.gfile.GFile(save_path, 'wb') as f:
                f.write(optimized_graph_def.SerializeToString())

            print("Saved optimized model for mobile devices at: %s." % (save_path,))
            print("Input node name: %s" % (input_node_name,))
            print("Output node name: %s" % (predicted_heatmaps[-1].name[:-2],))
            print("%d ops in the final graph." % len(optimized_graph_def.node))


def parse_args():

    parser = argparse.ArgumentParser(description='Export a trained Hourglass network.')

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to the specific model you want to export.',
                          required=True, type=str)

    parser.add_argument('--export_dir', dest='export_dir',
                          help='Path to a directory where the exported model will be saved.',
                          required=True, type=str)

    parser.add_argument('--savename', dest='savename',
                        help='filename you''d like to save the exported model to',
                        required=True, type=str)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    args = parse_args()
    cfg = parse_config_file(args.config_file)

    export(checkpoint_path=args.checkpoint_path,
           export_dir=args.export_dir,
           savename=args.savename,
           cfg=cfg)
