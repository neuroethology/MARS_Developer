import argparse
import copy
import os, sys
import yaml
import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.util import deprecation

from hourglass_pose.config import parse_config_file
from hourglass_pose import training_input
from hourglass_pose import loss
from hourglass_pose import model_pose as model
import pdb

deprecation._PRINT_DEPRECATION_WARNINGS = False


def build_model(input_imgs, cfg):
    # Set up the batch normalization parameters.
    batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections': [tf.compat.v1.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training': True
    }

    # Define a model_scope --everything within this scope has these parameters associated with it.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
        # Build the Stacked Hourglass model.
        predicted_heatmaps = model.build(
            input=input_imgs,
            num_parts=cfg.PARTS.NUM_PARTS,
            num_stacks=cfg.NUM_STACKS,
            reuse=tf.compat.v1.AUTO_REUSE
        )
    return predicted_heatmaps


def build_model_finetuning(input_imgs, cfg, n_train):
    # start with the fixed units:
    # Set up the batch normalization parameters.
    batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections': [tf.compat.v1.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training': False
    }
    # Define a model_scope --everything within this scope has these parameters associated with it.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
        # Build the Stacked Hourglass model.
        hg_inputs = model.build_hg_head(
            input=input_imgs,
            reuse=tf.compat.v1.AUTO_REUSE
        )
        pdb.set_trace()
        predicted_heatmaps = model.build_hg(
            input=hg_inputs,
            num_parts=cfg.PARTS.NUM_PARTS,
            stack_range=range(cfg.NUM_STACKS - n_train),
            num_stacks=cfg.NUM_STACKS,
            reuse=tf.compat.v1.AUTO_REUSE
        )
    # now add the trainable units:
    batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections': [tf.compat.v1.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training': True
    }
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
        # Build the Stacked Hourglass model.
        predicted_heatmaps = model.build_hg(
            input=predicted_heatmaps,
            num_parts=cfg.PARTS.NUM_PARTS,
            stack_range=range(cfg.NUM_STACKS-n_train, cfg.NUM_STACKS),
            num_stacks=n_train,
            reuse=tf.compat.v1.AUTO_REUSE
        )
    return predicted_heatmaps


def train(tfrecords_train, tfrecords_val, logdir, cfg, debug_output=False, fine_tune=0):
    """
    Args:
    tfrecords (list of strings):
      Each string in the list contains the path to a tfrecord file --these are a wrapper for
      all the information about the keypoints we're training on.
    logdir (str)
      This string is the path to the place where we dump our training summaries (i.e. metrics).
    cfg (EasyDict)
      This is a cfg file that's already been parsed into EasyDict form from a yaml file.
    debug_output (bool)
      Turn on debug-level output during training.
    """

    if debug_output:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    # Make a graph
    graph = tf.Graph()

    with graph.as_default():
        # Create a variable to count the number of train() calls.
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Set the learning rate here --we use a constant learning rate.
        lr = tf.constant(cfg.INITIAL_LEARNING_RATE, name='fixed_learning_rate')

        # Create an optimizer that performs gradient descent.
        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=lr,
            decay=cfg.RMSPROP_DECAY,
            momentum=cfg.RMSPROP_MOMENTUM,
            epsilon=cfg.RMSPROP_EPSILON
        )

        # Get all the input nodes for this graph.
        batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids,\
        batched_background_heatmaps = training_input.input_nodes(
            tfrecords_train,
            num_epochs=None,
            shuffle_batch=True,
            add_summaries=True,
            cfg=cfg
        )

        batched_images_v, batched_heatmaps_v, batched_parts_v, batched_part_visibilities_v, batched_image_ids_v,\
        batched_background_heatmaps_v = training_input.input_nodes(
            tfrecords_val,
            num_epochs=None,
            shuffle_batch=True,
            add_summaries=True,
            cfg=cfg
        )

        # Copy the input summaries here, so they don't get overwritten or anything.
        input_summaries = copy.copy(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

        if fine_tune:
            predicted_heatmaps = build_model_finetuning(batched_images, cfg, fine_tune)
            predicted_heatmaps_v = build_model_finetuning(batched_images_v, cfg, fine_tune)
        else:
            predicted_heatmaps = build_model(batched_images, cfg)
            predicted_heatmaps_v = build_model(batched_images_v, cfg)

        # Add the loss functions to the graph tab of losses.
        heatmap_loss, hmloss_summaries = loss.add_heatmaps_loss(batched_heatmaps, predicted_heatmaps,
                                                                True, cfg)
        heatmap_val_loss = loss.compute_heatmaps_loss(batched_heatmaps_v, predicted_heatmaps_v, cfg)

        # Pool all the losses we've added together into one readout.
        total_loss = tf.compat.v1.losses.get_total_loss()

        tf.compat.v1.losses.add_loss(heatmap_val_loss, loss_collection='validation')

        # Track the moving averages of all trainable variables.
        #   At test time we'll restore all variables with the average value.
        #   Note that we maintain a "double-average" of the BatchNormalization
        #   global statistics. This is more complicated then need be but we employ
        #   this for backward-compatibility with our previous models.
        ema = tf.train.ExponentialMovingAverage(
            decay=cfg.MOVING_AVERAGE_DECAY,
            num_updates=global_step
        )

        # Get the variables to average (all of the variables).
        variables_to_average = (slim.get_model_variables())

        # Wrap up the ema into an operation and add it to the collection of update operations, which are called regularly.
        maintain_averages_op = ema.apply(variables_to_average)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, maintain_averages_op)

        # Make a train operation.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Create summary operations
        summary_op = tf.compat.v1.summary.merge([
                                                    tf.compat.v1.summary.scalar('total_loss', total_loss),
                                                    tf.compat.v1.summary.scalar('total_heatmap_loss', heatmap_loss),
                                                    tf.compat.v1.summary.scalar('validation_loss', heatmap_val_loss),
                                                    tf.compat.v1.summary.scalar('learning_rate', lr)
                                                ] + input_summaries + hmloss_summaries)

        # Training from scratch, so don't need initial assignment ops or an initial feed dict.
        init_assign_op = tf.no_op()
        init_feed_dict = {}

        # Create an initial assignment function.
        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)

        sess_config = tf.compat.v1.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
            )
        )

        # Set up a saver to save a checkpoint every so often.
        saver = tf.compat.v1.train.Saver(
            max_to_keep=cfg.MAX_TO_KEEP,
            keep_checkpoint_every_n_hours=cfg.KEEP_CHECKPOINT_EVERY_N_HOURS
        )

        # Run training.
        slim.learning.train(train_op, logdir,
                            init_fn=InitAssignFn,
                            number_of_steps=cfg.NUM_TRAIN_ITERATIONS,
                            save_summaries_secs=cfg.SAVE_SUMMARY_SECS,
                            save_interval_secs=cfg.SAVE_INTERVAL_SECS,
                            saver=saver,
                            session_config=sess_config,
                            summary_op=summary_op,
                            log_every_n_steps=cfg.LOG_EVERY_N_STEPS
                            )


def run_training(project, pose_model_names=[], max_training_steps=None, debug_output=False, fine_tune=0):
    # load project config
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not pose_model_names:
        pose_model_list = cfg['pose']
        pose_model_names = pose_model_list.keys()

    train_cfg = parse_config_file(os.path.join(project, 'pose', 'config_train.yaml'))

    # allow some command-line override of training epochs/batch size, for troubleshooting:
    if max_training_steps is not None:
        train_cfg.NUM_TRAIN_ITERATIONS = max_training_steps

    for model in pose_model_names:
        logdir = os.path.join(project, 'pose', model + '_log')
        if not os.path.isdir(logdir):
            os.mkdir(logdir)

        tf_dir = os.path.join(project, 'pose', model + '_tfrecords_pose')
        tfrecords_train = glob.glob(os.path.join(tf_dir, 'train_dataset-*'))
        tfrecords_val = glob.glob(os.path.join(tf_dir, 'val_dataset-*'))

        train(
            tfrecords_train=tfrecords_train,
            tfrecords_val=tfrecords_val,
            logdir=logdir,
            cfg=train_cfg,
            debug_output=debug_output,
            fine_tune=fine_tune
        )


if __name__ == '__main__':
    """
    hourglass_pose training command line entry point
    Arguments:
        project 	        The absolute path to the project directory.
        models 	            Subset of pose models to train (optional, defaults to all.)
        max_training_steps  Max number of training epochs, for troubleshooting (optional.)
        batch_size          Training batch size, for troubleshooting (optional.)
    """

    parser = argparse.ArgumentParser(description='hourglass_pose training command line entry point',
                                     prog='run_training')
    parser.add_argument('project', type=str, help="absolute path to project folder.")
    parser.add_argument('models', type=list, required=False, default=[],
                        help="(optional) list subset of pose models to train.")
    parser.add_argument('max_training_steps', type=int, required=False, default=None,
                        help="(optional) set a max number of training epochs (for troubleshooting.)")
    parser.add_argument('debug_output', type=int, required=False, default=False,
                        help="(optional) turns on 'debug' outputs (show loss and sec/step during training.)")
    args = parser.parse_args(sys.argv[1:])

    if not os.path.isdir(args.project):
        print('Project directory not found: ' + args.project)
    if isinstance(args.models, str):
        args.models = [args.models]

    run_training(args.project,
                 pose_model_names=args.models,
                 max_training_steps=args.max_training_steps,
                 debug_output=args.debug_output)
