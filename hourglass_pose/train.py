import argparse
import copy
import logging
import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import parse_config_file
import train_inputs
import loss
import model

def train(tfrecords, logdir, cfg):
  """
  Args:
    tfrecords (list of strings):
      Each string in the list contains the path to a tfrecord file --these are a wrapper for
      all the information about the keypoints we're training on.
    logdir (str)
      This string is the path to the place where we dump our training summaries (i.e. metrics).
    cfg (EasyDict)
      This is a cfg file that's already been parsed into EasyDict form from a yaml file.
  """

  # TODO: If you wanted to get rid of all the dubious warnings, you'd do that here.
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)

  # Make a graph
  graph = tf.Graph()

  with graph.as_default():
    
    # Create a variable to count the number of train() calls. 
    global_step = slim.get_or_create_global_step()

    # TODO: If you wanted to make the learning rate non-constant, you'd do it here.
    # Set the learning rate here --we use a constant learning rate.
    lr = tf.constant(cfg.INITIAL_LEARNING_RATE, name='fixed_learning_rate')

    # TODO: If you wanted to change the optimizer, you'd do it here.
    # Create an optimizer that performs gradient descent.
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate=lr,
      decay=cfg.RMSPROP_DECAY,
      momentum=cfg.RMSPROP_MOMENTUM,
      epsilon=cfg.RMSPROP_EPSILON
    )

    # Get all the input nodes for this graph.
    batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps = train_inputs.input_nodes(
      tfrecords,
      num_epochs=None,
      shuffle_batch = True,
      add_summaries = True,
      cfg=cfg
    )

    # Copy the input summaries here, so they don't get overwritten or anything.
    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
    
    # Set up the batch normalization parameters.
    batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections' : [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training' : True
    }


    # Define a model_scope --everything within this scope has these parameters associated with it.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:

      # Build the Stacked Hourglass model.
      predicted_heatmaps = model.build(
        input = batched_images, 
        num_parts = cfg.PARTS.NUM_PARTS
      )
      # Add the loss functions to the graph tab of losses.
      heatmap_loss, hmloss_summaries = loss.add_heatmaps_loss(batched_heatmaps, predicted_heatmaps,
                                                              True, cfg)

    # Pool all the losses we've added together into one readout.
    total_loss = tf.losses.get_total_loss()

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
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_averages_op)

    # Make a train operation.
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Create summary operations
    summary_op = tf.summary.merge([
      tf.summary.scalar('total_loss', total_loss),
      tf.summary.scalar('total_heatmap_loss', heatmap_loss),
      tf.summary.scalar('learning_rate', lr)
    ] + input_summaries + hmloss_summaries)

    # Training from scratch, so don't need initial assignment ops or an initial feed dict.
    init_assign_op = tf.no_op()
    init_feed_dict = {}
    # TODO: If you were gonna use a pretrained model, you'd set that up here.

    # Create an initial assignment function.
    def InitAssignFn(sess):
        sess.run(init_assign_op, init_feed_dict)

    sess_config = tf.ConfigProto(
      log_device_placement=False,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
          per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
      )
    )

    # Set up a saver to save a checkpoint every so often.
    saver = tf.train.Saver(
      max_to_keep = cfg.MAX_TO_KEEP,
      keep_checkpoint_every_n_hours = cfg.KEEP_CHECKPOINT_EVERY_N_HOURS
    )

    # Run training.
    slim.learning.train(train_op, logdir, 
      init_fn=InitAssignFn,
      number_of_steps=cfg.NUM_TRAIN_ITERATIONS,
      save_summaries_secs=cfg.SAVE_SUMMARY_SECS,
      save_interval_secs=cfg.SAVE_INTERVAL_SECS,
      saver=saver,
      session_config=sess_config,
      summary_op = summary_op,
      log_every_n_steps = cfg.LOG_EVERY_N_STEPS
    )


def parse_args():
  """Simple argparser for the program."""
  parser = argparse.ArgumentParser(description='Train a Stacked Hourglass pose estimator.')

  parser.add_argument('--tfrecords', dest='tfrecords',
                      help='paths to tfrecords files which contain the training data', type=str,
                      nargs='+', required=True)

  parser.add_argument('--logdir', dest='logdir',
                        help='path to directory to store summary files and checkpoint files', type=str,
                        required=True)

  parser.add_argument('--config', dest='config_file',
                      help='Path to the configuration file',
                      required=True, type=str)

  args = parser.parse_args()
  return args


def main():
  # Parse the command-line arguments.
  args = parse_args()

  # Display the command-line arguments.
  print "Command line arguments:"
  pprint.pprint(vars(args))
  print

  # Display the internals of the config_file.
  cfg = parse_config_file(args.config_file)
  print "Configurations:"
  pprint.pprint(cfg)
  print

  # Actually run the training.
  train(
    tfrecords=args.tfrecords,
    logdir=args.logdir,
    cfg=cfg
  )

# Run the main function.
if __name__ == '__main__':
  main()