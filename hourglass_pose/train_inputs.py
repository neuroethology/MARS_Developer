import numpy as np
import tensorflow as tf
import math
import random
from inputs import distorted_shifted_bounding_box, distort_color, apply_with_random_selector, flip_parts_left_right, build_heatmaps_etc

def input_nodes(tfrecords, cfg, num_epochs=None, shuffle_batch = True, add_summaries = True, do_augmentations=True):

  # Get the number of parts.
  num_parts = cfg.PARTS.NUM_PARTS

  # Set up Data Queue parameters.
  batch_size = cfg.BATCH_SIZE
  num_threads = cfg.NUM_INPUT_THREADS
  capacity = cfg.QUEUE_CAPACITY
  min_after_dequeue = cfg.QUEUE_MIN

  # Set up the input queue to read from.
  with tf.name_scope('inputs'):
    # A producer to generate tfrecord file paths
    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=num_epochs,
      shuffle=shuffle_batch
    )

    # Construct a Reader to read examples from the tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Parse an Example to access the Features
    features = tf.parse_single_example(
      serialized_example,
      features = {
        'image/id' : tf.FixedLenFeature([], tf.string),
        'image/encoded'  : tf.FixedLenFeature([], tf.string),
        'image/height' : tf.FixedLenFeature([], tf.int64),
        'image/width' : tf.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/count' : tf.FixedLenFeature([], tf.int64),
        'image/object/parts/x' : tf.VarLenFeature(dtype=tf.float32), # x coord for all parts and all objects
        'image/object/parts/y' : tf.VarLenFeature(dtype=tf.float32), # y coord for all parts and all objects
        'image/object/parts/v' : tf.VarLenFeature(dtype=tf.int64),   # part visibility for all parts and all objects
        'image/object/area' : tf.VarLenFeature(dtype=tf.float32), # the area of the object, based on segmentation mask or bounding box mask
      }
    )

    # Read in a jpeg image
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    
    # Convert the pixel values to be in the range [0,1]
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Assign each item in the Example we parsed to a variable in this namespace.

    # Just the height and width of the image.
    image_height = tf.cast(features['image/height'], tf.float32)
    image_width = tf.cast(features['image/width'], tf.float32)

    # The id associated with the image.
    image_id = features['image/id']

    # The bounding box parameters.
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Get the number of bounding boxes.
    num_bboxes = tf.cast(features['image/object/bbox/count'], tf.int32)
    no_bboxes = tf.equal(num_bboxes, 0)

    # Get the part locations (x and y coordinates) as well as their visibilities.
    parts_x = tf.expand_dims(features['image/object/parts/x'].values, 0)
    parts_y = tf.expand_dims(features['image/object/parts/y'].values, 0)
    parts_v = tf.cast(tf.expand_dims(features['image/object/parts/v'].values, 0), tf.int32)

    # Get the area occupied by each object.
    areas = features['image/object/area'].values
    areas = tf.reshape(areas, [num_bboxes])

    # Add a summary of the original data
    if add_summaries:
      # If we have no bboxes, then just give a default one, otherwise use the bbox from this example.
      bboxes_to_draw = tf.cond(no_bboxes,
                               lambda:  tf.constant([[0, 0, 1, 1]], tf.float32),
                               lambda: tf.transpose(tf.concat(axis=0, values=[ymin, xmin, ymax, xmax]), [1, 0]))

      # Reshape the bboxes to have the batching in the second dimension.
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])

      # Draw the bbox on the image.
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bboxes_to_draw)

      # Add this image with a bbox draw on it to the tab of summaries.
      tf.summary.image('original_image', image_with_bboxes)

    if do_augmentations:

      # TODO: We need to ensure that the perturbed bbox still contains the parts...
      with tf.name_scope('bbox_perturbation'):
        # Perturb the bounding box coordinates
        r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
        do_perturb = tf.logical_and(tf.less(r, cfg.DO_RANDOM_BBOX_SHIFT), tf.greater(num_bboxes, 0))
        xmin, ymin, xmax, ymax = tf.cond(do_perturb,
          lambda: distorted_shifted_bounding_box(xmin, ymin, xmax, ymax, num_bboxes, image_height, image_width, cfg.RANDOM_BBOX_SHIFT_EXTENT),
          lambda: tf.tuple([xmin, ymin, xmax, ymax])
        )

      # Randomly flip the image:
      with tf.name_scope('random_flip'):
        # Sample randomly.
        r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)

        # Only flip if config allows it, and the sample is less than our threshold.
        do_flip = tf.logical_and(tf.less(r, 0.5),
                                 cfg.DO_RANDOM_FLIP_LEFT_RIGHT
                                 )

        # If we're doing a flip, then flip it --otherwise, do nothing.
        image = tf.cond(do_flip,
                        lambda: tf.image.flip_left_right(image),
                        lambda: tf.identity(image))

        # If we're doing a flip, switch around the location of the bounding box, so we still get the right section.
        xmin, xmax = tf.cond(do_flip,
                             lambda: tf.tuple([1. - xmax, 1. - xmin]),
                             lambda: tf.tuple([xmin, xmax])
                             )
        # If we're doing a flip, flip the part locations around as well.
        parts_x, parts_y, parts_v = tf.cond(do_flip,
          lambda: tf.py_func(flip_parts_left_right, [parts_x, parts_y, parts_v, cfg.PARTS.LEFT_RIGHT_PAIRS, num_parts], [tf.float32, tf.float32, tf.int32]),
          lambda: tf.tuple([parts_x, parts_y, parts_v])
        )
        print( '\n')
        part_visibilities = tf.reshape(parts_v, tf.stack([num_bboxes, num_parts]))

      with tf.name_scope('distort_color'):
        # Distort the colors
        r = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
        do_color_distortion = tf.less(r, cfg.DO_COLOR_DISTORTION)
        num_color_cases = 1 if cfg.COLOR_DISTORT_FAST else 4
        distorted_image = apply_with_random_selector(
          image,
          lambda x, ordering: distort_color(x, ordering, fast_mode=cfg.COLOR_DISTORT_FAST),
          num_cases=num_color_cases)
        image = tf.cond(do_color_distortion, lambda: tf.identity(distorted_image), lambda: tf.identity(image))

    # Change the image's shape.
    image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])

    # Add a summary
    if add_summaries:
      bboxes_to_draw = tf.cond(no_bboxes, lambda:  tf.constant([[0, 0, 1, 1]], tf.float32), lambda: tf.transpose(tf.concat(axis=0, values=[ymin, xmin, ymax, xmax]), [1, 0]))
      bboxes_to_draw = tf.reshape(bboxes_to_draw, [1, -1, 4])
      image_with_bboxes = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bboxes_to_draw)
      tf.summary.image('processed_image', image_with_bboxes)


    # Create the crops, the bounding boxes, the parts and heatmaps
    bboxes = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
    bboxes = tf.transpose(bboxes, [1, 0])
    parts = tf.concat(axis=0, values=[parts_x, parts_y])
    parts = tf.transpose(parts, [1, 0])
    parts = tf.reshape(parts, [-1, num_parts * 2])

    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)

    with tf.name_scope('build_heatmaps'):
    # An explanation for each of these parameters can be found in the documentation.
    # TODO: This stuff is in one long list to enable us to run the function using py_func --tf.py_func doesnt support dicts.
      params = [
        image, bboxes, parts, part_visibilities,
        cfg.PARTS.SIGMAS, areas, cfg.PARTS.SCALE_SIGMAS_BY_AREA,
        cfg.INPUT_SIZE, cfg.HEATMAP_SIZE,
        cfg.LOOSE_BBOX_CROP, cfg.LOOSE_BBOX_PAD_FACTOR,
        cfg.PARTS.LEFT_RIGHT_PAIRS,
        cfg.BACKGROUND_HEATMAPS.ADD_TARGET_LEFT_RIGHT_PAIRS, cfg.BACKGROUND_HEATMAPS.ADD_NON_TARGET_PARTS,
        cfg.BACKGROUND_HEATMAPS.NON_TARGET_INCLUDE_OCCLUDED, cfg.BACKGROUND_HEATMAPS.ADD_NON_TARGET_LEFT_RIGHT_PAIRS,
        cfg.DO_RANDOM_PADDING, cfg.RANDOM_PADDING_FREQ,
        cfg.RANDOM_PADDING_MIN, cfg.RANDOM_PADDING_MAX,
        cfg.DO_RANDOM_BLURRING, cfg.RANDOM_BLUR_FREQ, cfg.MAX_BLUR,
        cfg.DO_RANDOM_NOISE, cfg.RANDOM_NOISE_FREQ, cfg.RANDOM_NOISE_SCALE,
        cfg.DO_JPEG_ARTIFACTS, cfg.RANDOM_JPEG_FREQ, cfg.RANDOM_JPEG_QUALITY_MIN, cfg.RANDOM_JPEG_QUALITY_MAX
      ]
      cropped_images, heatmaps, parts, background_heatmaps = tf.py_func(build_heatmaps_etc, params, [tf.uint8, tf.float32, tf.float32, tf.float32])
      cropped_images = tf.image.convert_image_dtype(cropped_images, dtype=tf.float32)


    # Add a summary of the final crops
    if add_summaries:
      tf.summary.image('cropped_images', cropped_images)

    # Get the images in the range [-1, 1]
    cropped_images = tf.subtract(cropped_images, 0.5)
    cropped_images = tf.multiply(cropped_images, 2.0)

    # Set the shape of everything for the queue --None means that's the batch dimension.
    cropped_images.set_shape([None, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    image_ids = tf.tile([[image_id]], [num_bboxes, 1])
    image_ids.set_shape([None, 1])

    heatmaps.set_shape([None, cfg.HEATMAP_SIZE, cfg.HEATMAP_SIZE, num_parts])
    bboxes.set_shape([None, 4])
    parts.set_shape([None, num_parts * 2])
    part_visibilities.set_shape([None, num_parts])
    background_heatmaps.set_shape([None, cfg.HEATMAP_SIZE, cfg.HEATMAP_SIZE, num_parts])

    if shuffle_batch:
      batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps = tf.train.shuffle_batch(
        [cropped_images, heatmaps, parts, part_visibilities, image_ids, background_heatmaps],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue= min_after_dequeue, # 3 * batch_size,
        seed = cfg.RANDOM_SEED,
        enqueue_many=True,
        name="shuffle_batch_queue"
      )
    else:
      batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps = tf.train.batch(
        [cropped_images, heatmaps, parts, part_visibilities, image_ids, background_heatmaps],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        enqueue_many=True
      )

    if do_augmentations:
      if cfg.DO_RANDOM_ROTATION:
        with tf.name_scope('random_rotation'):
          if random.random() < cfg.RANDOM_ROTATION_FREQ:
            # Generate random angles to rotate by.
            delta = math.pi*(cfg.RANDOM_ROTATION_DELTA/180.)
            angles = tf.random_uniform(tf.constant([batch_size]), -delta, delta)

            # Rotate the image-like stuff for the training set.
            batched_images =  tf.contrib.image.rotate(batched_images, angles)
            batched_heatmaps = tf.contrib.image.rotate(batched_heatmaps, angles)
            batched_background_heatmaps = tf.contrib.image.rotate(batched_heatmaps, angles)

            # Rotate the parts.
            batched_parts = rotate_parts(batched_parts, angles, batch_size, cfg.INPUT_SIZE)



  # return a batch of images and their labels
  return batched_images, batched_heatmaps, batched_parts, batched_part_visibilities, batched_image_ids, batched_background_heatmaps


def rotate_parts(batched_parts, angles, batch_size, input_size):
  ### Rotates a batch of parts about a batch of angles.
  batched_new_parts = []

  # The part coordinates are normalized to the input image, which is of a designated size.
  image_width = input_size
  image_height = input_size

  for b in xrange(batch_size):
    # Break everything up batch-wise.
    parts = batched_parts[b]
    angle = angles[b]

    # Scale the x and y coordinates.
    x = parts[0::2]*image_width
    y = parts[1::2]*image_height

    # Rotate the x and y be specified angles
    new_x, new_y = rotate(x,y,angle, input_size)

    # Renormalize.
    new_x = new_x/image_width
    new_y = new_y/image_height

    # Prepare the new coordinates for stacking.
    new_x = tf.expand_dims(new_x,0)
    new_y = tf.expand_dims(new_y,0)

    # Get the coordinates back into their original ordering.
    new_parts = tf.stack([new_x, new_y], 0)
    new_parts = tf.transpose(new_parts)
    new_parts = tf.reshape(new_parts, [-1])

    # Add to the list of new parts.
    batched_new_parts.append(new_parts)

  return tf.stack(batched_new_parts)


def rotate(xs, ys, angle, input_size):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    # Set the origin.
    ox = input_size/2
    oy = input_size/2

    # Correct our sign conventions.
    angle = -angle

    # Do the actual rotation.
    qx = ox + tf.cos(angle) * (xs - ox) - tf.sin(angle) * (ys - oy)
    qy = oy + tf.sin(angle) * (xs - ox) + tf.cos(angle) * (ys - oy)
    return qx, qy
