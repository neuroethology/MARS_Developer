import argparse
import pickle
import copy
import os, sys
import yaml
import glob

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.util import deprecation

from multibox_detection import inputs
from multibox_detection import loss
from multibox_detection import model_detection as model
from multibox_detection.config import parse_config_file

# import inputs
# import loss
# import model_detection as model
# from config import parse_config_file

deprecation._PRINT_DEPRECATION_WARNINGS = False


def get_init_function(logdir, pretrained_model_path, fine_tune, original_inception_vars, use_moving_averages=False,
                      restore_moving_averages=False, ema=None):
    """
    Args:
    logdir : location of where we will be storing checkpoint files.
    pretrained_model_path : a path to a specific model, or a directory with a checkpoint file. The latest model will be used.
    fine_tune : If True, then the detection heads will not be restored.
    original_inception_vars : A list of variables that do not include the detection heads.
    use_moving_averages : If True, then the moving average values of the variables will be restored.
    restore_moving_averages : If True, then the moving average values will also be restored.
    ema : The exponential moving average object
    """

    if pretrained_model_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(logdir):
        tf.logging.info(
            'Ignoring --pretrained_model_path because a checkpoint already exists in %s'
            % logdir)
        return None

    if tf.gfile.IsDirectory(pretrained_model_path):
        checkpoint_path = tf.train.latest_checkpoint(pretrained_model_path)
    else:
        checkpoint_path = pretrained_model_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    # Do we need to restore the detection heads?
    if fine_tune:
        variables_to_restore = original_inception_vars
    else:
        variables_to_restore = slim.get_model_variables()

    # Load in the moving average value for a variable, rather than the variable itself
    if use_moving_averages:

        if type(variables_to_restore) == dict:
            vars = variables_to_restore.values()
        else:
            vars = variables_to_restore

        variables_to_restore = {
            ema.average_name(var): var
            for var in vars
        }

    # Do we want to restore the moving average variables? Otherwise they will be reinitialized
    if restore_moving_averages:

        # If we are already using the moving averages to restore the variables, then we will need
        # two Saver() objects (since the names in the dictionaries will clash)
        if use_moving_averages:

            normal_saver = tf.train.Saver(variables_to_restore, reshape=False)
            ema_saver = tf.train.Saver({
                ema.average_name(var): ema.average(var)
                for var in variables_to_restore.values()
            }, reshape=False)

            def callback(session):
                normal_saver.restore(session, checkpoint_path)
                ema_saver.restore(session, checkpoint_path)

            return callback

        else:
            # GVH: Need to check for dict
            variables_to_restore += [ema.average(var) for var in variables_to_restore]

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=False)


def build_fully_trainable_model(inputs, cfg):
    batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections': [tf.compat.v1.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training': True
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
        locs, confs, inception_vars = model.build(
            inputs=inputs,
            num_bboxes_per_cell=cfg.NUM_BBOXES_PER_CELL,
            reuse=False,
            scope=''
        )

    return locs, confs, inception_vars


def build_finetunable_model(inputs, cfg):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
        batch_norm_params = {
            'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
            'epsilon': 0.001,
            'variables_collections': [],
            'is_training': False
        }
        with slim.arg_scope([slim.conv2d], normalizer_params=batch_norm_params):
            features, _ = model.inception_resnet_v2(inputs, reuse=False, scope='InceptionResnetV2')

        # Save off the original variables (for ease of restoring)
        model_variables = slim.get_model_variables()
        inception_vars = {var.op.name: var for var in model_variables}

        batch_norm_params = {
            'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
            'epsilon': 0.001,
            'variables_collections': [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
            'is_training': True
        }
        with slim.arg_scope([slim.conv2d], normalizer_params=batch_norm_params):
            # Add on the detection heads
            locs, confs, _ = model.build_detection_heads(features, cfg.NUM_BBOXES_PER_CELL)
            model_variables = slim.get_model_variables()
            detection_vars = {var.op.name: var for var in model_variables if var.op.name not in inception_vars}

    return locs, confs, inception_vars, detection_vars


def filter_trainable_variables(trainable_vars, trainable_scopes):
    """Allow the user to further restrict which variables should be trained.
  """
    if trainable_scopes is None:
        return trainable_vars
    else:
        scopes = [scope.strip() for scope in trainable_scopes]

    trainable_var_set = set(trainable_vars)

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend([var for var in variables if var in trainable_var_set])

    print("Trainable Variables")
    for variable in variables_to_train:
        print(variable.name)

    return variables_to_train


def train(tfrecords, bbox_priors, logdir, cfg, pretrained_model_path=None, fine_tune=False, trainable_scopes=None,
          use_moving_averages=False, restore_moving_averages=False, debug_output=False):
    """
    Args:
    tfrecords (list)
    bbox_priors (np.array)
    logdir (str)
    cfg (EasyDict)
    pretrained_model_path (str) : path to a pretrained Inception Network
    """
    if not debug_output:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    graph = tf.Graph()

    # Force all Variables to reside on the CPU.
    with graph.as_default():

        # Create a variable to count the number of train() calls.
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (cfg.NUM_TRAIN_EXAMPLES /
                                 cfg.BATCH_SIZE)
        decay_steps = int(num_batches_per_epoch * cfg.NUM_EPOCHS_PER_DELAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.compat.v1.train.exponential_decay(
            learning_rate=cfg.INITIAL_LEARNING_RATE,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=cfg.LEARNING_RATE_DECAY_FACTOR,
            staircase=cfg.LEARNING_RATE_STAIRCASE
        )

        # Create an optimizer that performs gradient descent.
        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=lr,
            decay=cfg.RMSPROP_DECAY,
            momentum=cfg.RMSPROP_MOMENTUM,
            epsilon=cfg.RMSPROP_EPSILON
        )

        batched_images, batched_bboxes, batched_num_bboxes, image_ids = inputs.input_nodes(
            tfrecords=tfrecords,
            max_num_bboxes=cfg.MAX_NUM_BBOXES,
            num_epochs=None,
            batch_size=cfg.BATCH_SIZE,
            num_threads=cfg.NUM_INPUT_THREADS,
            capacity=cfg.QUEUE_CAPACITY,
            min_after_dequeue=cfg.QUEUE_MIN,
            add_summaries=True,
            shuffle_batch=True,
            cfg=cfg
        )

        input_summaries = copy.copy(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES))

        if fine_tune:
            locs, confs, inception_vars, detection_vars = build_finetunable_model(batched_images, cfg)
            all_trainable_var_names = [v.op.name for v in tf.trainable_variables()]
            trainable_vars = [v for v_name, v in detection_vars.items() if v_name in all_trainable_var_names]
        else:
            locs, confs, inception_vars = build_fully_trainable_model(batched_images, cfg)
            trainable_vars = tf.compat.v1.trainable_variables()

        location_loss, confidence_loss = loss.add_loss(
            locations=locs,
            confidences=confs,
            batched_bboxes=batched_bboxes,
            batched_num_bboxes=batched_num_bboxes,
            bbox_priors=bbox_priors,
            location_loss_alpha=cfg.LOCATION_LOSS_ALPHA
        )

        total_loss = tf.compat.v1.losses.get_total_loss()

        # Track the moving averages of all trainable variables.
        # At test time we'll restore all variables with the average value
        # Note that we maintain a "double-average" of the BatchNormalization
        # global statistics. This is more complicated then need be but we employ
        # this for backward-compatibility with our previous models.
        ema = tf.train.ExponentialMovingAverage(
            decay=cfg.MOVING_AVERAGE_DECAY,
            num_updates=global_step
        )
        variables_to_average = (
            slim.get_model_variables())  # Makes it easier to restore for eval and detect purposes (whether you use the fine_tune flag or not)
        maintain_averages_op = ema.apply(variables_to_average)
        tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, maintain_averages_op)

        trainable_vars = filter_trainable_variables(trainable_vars, trainable_scopes)

        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=trainable_vars)

        # Summary operations
        summary_op = tf.compat.v1.summary.merge([
                                                    tf.compat.v1.summary.scalar('total_loss', total_loss),
                                                    tf.compat.v1.summary.scalar('location_loss', location_loss),
                                                    tf.compat.v1.summary.scalar('confidence_loss', confidence_loss),
                                                    tf.compat.v1.summary.scalar('learning_rate', lr)
                                                ] + input_summaries)

        sess_config = tf.compat.v1.ConfigProto(
            log_device_placement=False,
            # device_filters = device_filters,
            allow_soft_placement=True,
            gpu_options=tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
            )
        )

        saver = tf.compat.v1.train.Saver(
            # Save all variables
            max_to_keep=cfg.MAX_TO_KEEP,
            keep_checkpoint_every_n_hours=cfg.KEEP_CHECKPOINT_EVERY_N_HOURS
        )

        # Run training.
        slim.learning.train(train_op, logdir,
                            init_fn=get_init_function(logdir, pretrained_model_path, fine_tune, inception_vars,
                                                      use_moving_averages, restore_moving_averages, ema),
                            number_of_steps=cfg.NUM_TRAIN_ITERATIONS,
                            save_summaries_secs=cfg.SAVE_SUMMARY_SECS,
                            save_interval_secs=cfg.SAVE_INTERVAL_SECS,
                            saver=saver,
                            session_config=sess_config,
                            summary_op=summary_op,
                            log_every_n_steps=cfg.LOG_EVERY_N_STEPS
                            )


def run_training(project, detector_names=[], max_training_steps=None, debug_output=False):
    # load project config
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not detector_names:
        detector_list = cfg['detection']
        detector_names = detector_list.keys()

    train_cfg = parse_config_file(os.path.join(project, 'detection', 'config_train.yaml'))

    # allow some command-line override of training epochs/batch size, for troubleshooting:
    if max_training_steps is not None:
        train_cfg.NUM_TRAIN_ITERATIONS = max_training_steps

    for detector in detector_names:
        logdir = os.path.join(project, 'detection', detector + '_log')
        if not os.path.isdir(logdir):
            os.mkdir(logdir)

        tf_dir = os.path.join(project, 'detection', detector + '_tfrecords_detection')
        tfrecords = glob.glob(os.path.join(tf_dir, 'train_dataset-*'))

        priors_fid = os.path.join(project, 'detection', 'priors_' + detector + '.pkl')
        with open(priors_fid, 'rb') as f:
            bbox_priors = pickle.load(f, encoding='latin1')
        bbox_priors = np.array(bbox_priors).astype(np.float32)

        train(
            tfrecords=tfrecords,
            bbox_priors=bbox_priors,
            logdir=logdir,
            cfg=train_cfg,
            pretrained_model_path=None if cfg['pretrained_model_path'] == 'None' else cfg['pretrained_model_path'],
            fine_tune=cfg['fine_tune'],
            trainable_scopes=None if cfg['trainable_scopes'] == 'None' else cfg['trainable_scopes'],
            use_moving_averages=cfg['use_moving_averages'],
            restore_moving_averages=cfg['restore_moving_averages'],
            debug_output=debug_output
        )


if __name__ ==  '__main__':
    """
    multibox_detection training command line entry point
    Arguments:
        project 	        The absolute path to the project directory.
        models 	            Subset of detectors to train (optional, defaults to all.)
        max_training_steps  Max number of training epochs, for troubleshooting (optional.)
        batch_size          Training batch size, for troubleshooting (optional.)
    """

    parser = argparse.ArgumentParser(description='multibox_detection training command line entry point',
                                     prog='run_training')
    parser.add_argument('project', type=str, help="absolute path to project folder.")
    parser.add_argument('models', type=list, required=False, default=[],
                        help="(optional) list subset of detectors to train.")
    parser.add_argument('max_training_steps', type=int, required=False, default=None,
                        help="(optional) set a max number of training epochs (for troubleshooting.)")
    parser.add_argument('debug_output', type=int, required=False, default=False,
                        help="(optional) display debugger-level output during training.")
    args = parser.parse_args(sys.argv[1:])

    if not os.path.isdir(args.project):
        print('Project directory not found: ' + args.project)
    if isinstance(args.models, str):
        args.models = [args.models]

    run_training(args.project,
                 detector_names=args.models,
                 max_training_steps=args.max_training_steps,
                 debug_output=args.debug_output)
