from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.util import deprecation

slim = tf.contrib.slim
import pdb
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath('..'))
import hourglass_pose.model as model_pose

deprecation._PRINT_DEPRECATION_WARNINGS = False


def export(checkpoint_path, export_dir, export_version, view , num_parts, num_stacks):

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    graph = tf.Graph()

    input_node_name = "images"
    output_node_name = "output"

    with graph.as_default():

        input_height = 256
        input_width = 256
        input_depth = 3

        #we assume that we have already preprocessed the image bboxes and bboxes
        images_bboxes = tf.compat.v1.placeholder(tf.float32,[None, input_height, input_width, input_depth], name=input_node_name)

        #build model detection
        batch_norm_params = {
            # Decay for the batch_norm moving averages.
            'decay': 0.9997,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            'variables_collections': [tf.compat.v1.GraphKeys.MOVING_AVERAGE_VARIABLES],
            'is_training': False
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            biases_regularizer=slim.l2_regularizer(0.00004)) as scope:

            predicted_heatmaps = model_pose.build(
                input = images_bboxes,
                num_parts = num_parts,
                num_stacks = num_stacks
            )
            output_node = tf.identity(predicted_heatmaps[-1], output_node_name)

        variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        # variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables())
        variables_to_restore = {
            variable_averages.average_name(var): var
            for var in slim.get_model_variables()
            }

        # retrieve checkpoint
        if os.path.isdir(checkpoint_path):
            checkpoint_dir = checkpoint_path
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

            if checkpoint_path is None:
                raise ValueError("Unable to find a model checkpoint in the directory %s" % (checkpoint_dir,))

        tf.logging.info('Exporting model: %s' % checkpoint_path)

        # we import the meta graph and retrieve the saver
        saver = tf.train.Saver(variables_to_restore, reshape=True)

        #reteieve the protobuf graph definition
        input_graph_def = graph.as_graph_def()  # graph used to retrieve the nodes

        #configure the session
        sess_config = tf.compat.v1.ConfigProto(
                log_device_placement= False,
                allow_soft_placement = True,
                gpu_options = tf.compat.v1.GPUOptions(
                    per_process_gpu_memory_fraction=0.9
                )
            )
        sess = tf.compat.v1.Session(graph=graph, config=sess_config)

        #start the session and restore the graph weights
        with sess.as_default():

            tf.compat.v1.global_variables_initializer().run()

            saver.restore(sess, checkpoint_path)
            #export varibales to constants
            constant_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=[output_node.name[:-2]])

            optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def=constant_graph_def,
                input_node_names=[input_node_name],
                output_node_names=[output_node.name[:-2]],
                placeholder_type_enum=dtypes.float32.as_datatype_enum)

            # serialize and dump the putput graph to fs
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            save_path = os.path.join(export_dir, 'optimized_model_%s_pose_%d.pb' % (view, export_version,))
            with tf.io.gfile.GFile(save_path, 'wb') as f:
                f.write(optimized_graph_def.SerializeToString())

            print("Saved optimized model for mobile devices at: %s." % (save_path,))
            print("Input node name: %s" % (input_node_name,))
            print("Output node name: %s" % (output_node.name[:-2],))
            print("%d ops in the final graph." % len(optimized_graph_def.node))


def parse_args():

    parser = argparse.ArgumentParser(description='Test an Hourglass export network')

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to the specific model you want to export.',
                          required=True, type=str)

    parser.add_argument('--export_dir', dest='export_dir',
                          help='Path to a directory where the exported model will be saved.',
                          required=True, type=str)

    parser.add_argument('--export_version', dest='export_version',
                        help='Version number of the model.',
                        required=True, type=int)

    parser.add_argument('--view', dest='view',
                        help='Top or front.',
                        required=True, type=str)

    parser.add_argument('--num_parts', dest='num_parts',
                        help='Number of parts.',
                        required=True, type=int)

    parser.add_argument('--num_stacks', dest='num_stacks',
                        help='Number of stacks in the model.',
                        required=True, type=int)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    export(args.checkpoint_path, args.export_dir, args.export_version, args.view, args.num_parts, args.num_stacks)

