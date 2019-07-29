from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib

slim = tf.contrib.slim
import pdb
import cPickle as pickle
import numpy as np

import model_detection


def export(checkpoint_path, export_dir, export_version, prior_path, strain, view):

    graph = tf.Graph()

    input_node_name = "images"
    output_node_name = None

    with graph.as_default():

        input_height = 299
        input_width = 299
        input_depth = 3

        #we assume that we have already preprocessed the image
        input_placeholder = tf.placeholder(tf.float32, [None, input_height * input_width * input_depth], name=input_node_name)
        images = tf.reshape( input_placeholder, [-1, input_height,input_width,input_depth])

        with open(prior_path,'r') as f:
            priors_bbox  =pickle.load(f)
        priors_bbox = np.array(priors_bbox).astype(np.float32)


        #build model detection
        batch_norm_params = {
            # Decay for the batch_norm moving averages.
            'decay': 0.9997,
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

            locations, confidences, inception_vars = model_detection.build(
                inputs=images,
                num_bboxes_per_cell=11,
                reuse=False,
                scope=''
            )

            priors_bboxes = tf.constant(priors_bbox, name='priors_bboxes')
            predicted_loc = tf.add(locations,priors_bboxes,name='predicted_locations')


        variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables())

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
        sess_config = tf.ConfigProto(
                log_device_placement= False,
                allow_soft_placement = True,
                gpu_options = tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.9
                )
            )
        sess = tf.Session(graph=graph, config=sess_config)

        #start the session and restore the graph weights
        with sess.as_default():

            tf.global_variables_initializer().run()

            saver.restore(sess, checkpoint_path)

            #export varibales to constants
            constant_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=[predicted_loc.name[:-2],confidences.name[:-2]])

            optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def=constant_graph_def,
                input_node_names=input_node_name,
                output_node_names=[predicted_loc.name[:-2],confidences.name[:-2]],
                placeholder_type_enum=dtypes.float32.as_datatype_enum)

            # serialize and dump the putput graph to fs
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            save_path = os.path.join(export_dir, 'optimized_model_%s_detection_%s_%d.pb' % (view,strain,export_version,))
            with tf.gfile.GFile(save_path, 'wb') as f:
                f.write(optimized_graph_def.SerializeToString())

            print("Saved optimized model for mobile devices at: %s." % (save_path,))
            print("Input node name: %s" % (input_node_name,))
            print("Output node name: %s %s" % (predicted_loc.name[:-2],confidences.name[:-2]))
            print("%d ops in the final graph." % len(optimized_graph_def.node))





def parse_args():

    parser = argparse.ArgumentParser(description='Test an export network')

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='Path to the specific model you want to export.',
                          required=True, type=str)

    parser.add_argument('--export_dir', dest='export_dir',
                          help='Path to a directory where the exported model will be saved.',
                          required=True, type=str)

    parser.add_argument('--export_version', dest='export_version',
                        help='Version number of the model.',
                        required=True, type=int)

    parser.add_argument('--prior_path', dest='prior_path',
                        help='Path to prior bboxes.',
                        required=True, type=str)

    parser.add_argument('--strain', dest='strain',
                        help='Mouse color.',
                        required=True, type=str)

    parser.add_argument('--view', dest='view',
                        help='Top or Front.',
                        required=True, type=str)


    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()

    export(args.checkpoint_path, args.export_dir, args.export_version, args.prior_path,args.strain, args.view)