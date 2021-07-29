from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import yaml
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import importer
# from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.util import deprecation

slim = tf.contrib.slim

sys.path.insert(0, os.path.abspath('..'))
from multibox_detection import model_detection
from hourglass_pose import model_pose
from hourglass_pose.config import parse_config_file

deprecation._PRINT_DEPRECATION_WARNINGS = False


def export_detector(checkpoint_path, export_dir, model_name, prior_path):

    graph = tf.Graph()
    input_node_name = "images"
    output_node_name = None

    with graph.as_default():

        input_height = 299
        input_width = 299
        input_depth = 3

        #we assume that we have already preprocessed the image
        input_placeholder = tf.compat.v1.placeholder(tf.float32, [None, input_height * input_width * input_depth], name=input_node_name)
        images = tf.reshape(input_placeholder, [-1, input_height,input_width,input_depth])

        with open(prior_path, 'rb') as f:
            priors_bbox = pickle.load(f, encoding='latin1')
        priors_bbox = np.array(priors_bbox).astype(np.float32)

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
                            biases_regularizer=slim.l2_regularizer(0.00004)):

            locations, confidences, inception_vars = model_detection.build(
                inputs=images,
                num_bboxes_per_cell=11,
                reuse=False,
                scope=''
            )

            priors_bboxes = tf.constant(priors_bbox, name='priors_bboxes')
            predicted_loc = tf.add(locations, priors_bboxes, name='predicted_locations')

        variable_averages = tf.train.ExponentialMovingAverage(0.9999)
        variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables())

        # retrieve checkpoint
        if os.path.isdir(checkpoint_path):
            checkpoint_dir = checkpoint_path
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

            if checkpoint_path is None:
                raise ValueError("Unable to find a model checkpoint in the directory %s" % (checkpoint_dir,))

        tf.compat.v1.logging.info('Exporting model: %s' % checkpoint_path)

        # we import the meta graph and retrieve the saver
        saver = tf.compat.v1.train.Saver(variables_to_restore, reshape=True)

        #reteieve the protobuf graph definition
        input_graph_def = graph.as_graph_def()  # graph used to retrieve the nodes

        #configure the session
        sess_config = tf.compat.v1.ConfigProto(
                log_device_placement=False,
                allow_soft_placement=True,
                gpu_options=tf.compat.v1.GPUOptions(
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
                output_node_names=[predicted_loc.name[:-2],confidences.name[:-2]])

            optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
                input_graph_def=constant_graph_def,
                input_node_names=[input_node_name],
                output_node_names=[predicted_loc.name[:-2],confidences.name[:-2]],
                placeholder_type_enum=dtypes.float32.as_datatype_enum)

            # serialize and dump the putput graph to fs
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            save_path = os.path.join(export_dir, model_name)
            with tf.io.gfile.GFile(save_path, 'wb') as f:
                f.write(optimized_graph_def.SerializeToString())

            print("Saved optimized detection model at: %s" % (save_path,))
            print("Input node name: %s" % (input_node_name,))
            print("Output node names: %s %s" % (predicted_loc.name, confidences.name))
            print("%d ops in the final graph." % len(optimized_graph_def.node))


def export_pose(checkpoint_path, export_dir, model_name, num_parts, num_stacks):

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    graph = tf.Graph()

    input_node_name = "images"
    output_node_name = "output_heatmaps"

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

        tf.compat.v1.logging.info('Exporting model: %s' % checkpoint_path)

        # we import the meta graph and retrieve the saver
        saver = tf.compat.v1.train.Saver(variables_to_restore, reshape=True)

        #reteieve the protobuf graph definition
        input_graph_def = graph.as_graph_def()  # graph used to retrieve the nodes

        #configure the session
        sess_config = tf.compat.v1.ConfigProto(
                log_device_placement=False,
                allow_soft_placement=True,
                gpu_options=tf.compat.v1.GPUOptions(
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

            # serialize and dump the output graph to fs
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            save_path = os.path.join(export_dir, model_name)
            with tf.io.gfile.GFile(save_path, 'wb') as f:
                f.write(optimized_graph_def.SerializeToString())

            print("Saved optimized pose model at: %s" % save_path)
            print("Input node name: %s" % input_node_name)
            print("Output node name: %s" % output_node.name)
            print("%d ops in the final graph." % len(optimized_graph_def.node))


def export(project, pose_model_names=None, detector_names=None):
    # load project config and model config
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    mdl_cfg = parse_config_file(os.path.join(project, 'pose', 'config_train.yaml'))

    if not detector_names:
        detector_list = cfg['detection']
        detector_names = detector_list.keys()

    num_parts = len(cfg['keypoints'])
    num_stacks = mdl_cfg.NUM_STACKS
    if not pose_model_names:
        pose_model_list = cfg['pose']
        pose_model_names = pose_model_list.keys()

    for model in detector_names:
        checkpoint_path = os.path.join(project, 'detection', model + '_model')
        model_name = cfg['project_name'] + '_' + model + '_detector.pb'
        prior_path = os.path.join(project, 'detection', 'priors_' + model + '.pkl')
        export_detector(checkpoint_path, project, model_name, prior_path)
        print('--')

    for model in pose_model_names:
        checkpoint_path = os.path.join(project, 'pose', model + '_model')
        model_name = cfg['project_name'] + '_' + model + '_pose.pb'
        export_pose(checkpoint_path, project, model_name, num_parts, num_stacks)
        print('--')
