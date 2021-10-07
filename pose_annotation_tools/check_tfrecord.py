import tensorflow.compat.v1 as tf
from tensorflow.python.util import deprecation
from hourglass_pose.inputs import flip_parts_left_right

deprecation._PRINT_DEPRECATION_WARNINGS = False


def restore(tfrecords_filenames):
    # this is a script to extract data from tfrecord files, for sanity-checking.
    f = tfrecords_filenames
    totalFiles = 0

    tf.reset_default_graph()

    # get the number of records in the tfrecord file
    c = 0
    for file in tfrecords_filenames:
        for record in tf.python_io.tf_record_iterator(file):
            c += 1
    totalFiles += c

    tf.reset_default_graph()

    # tfrecords_filenames should be paths to tfrecords files as a list
    fq = tf.train.string_input_producer(tfrecords_filenames, num_epochs=totalFiles)
    reader = tf.TFRecordReader()
    _, v = reader.read(fq)
    # the fields we want to extract
    fx = {
        'image/id' : tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin' : tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin' : tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax' : tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax' : tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/count' : tf.io.FixedLenFeature([], tf.int64),
        'image/object/parts/x' : tf.io.VarLenFeature(dtype=tf.float32), # x coord for all parts and all objects
        'image/object/parts/y' : tf.io.VarLenFeature(dtype=tf.float32), # y coord for all parts and all objects
        'image/object/parts/v' : tf.io.VarLenFeature(dtype=tf.int64),   # part visibility for all parts and all objects
      }

    # parse the tfrecord, then unpack values
    features = tf.parse_single_example(v, fx)

    # The id associated with the image.
    image_id = features['image/id']

    # The bounding box parameters.
    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Get the number of bounding boxes.
    num_bboxes = tf.cast(features['image/object/bbox/count'], tf.int32)

    # Get the part locations (x and y coordinates) as well as their visibilities.
    parts_x = tf.expand_dims(features['image/object/parts/x'].values, 0)
    parts_y = tf.expand_dims(features['image/object/parts/y'].values, 0)
    parts_v = tf.cast(tf.expand_dims(features['image/object/parts/v'].values, 0), tf.int32)

    num_parts = 8  # hacks!

    #  flip the image:
    parts_x, parts_y, parts_v = tf.numpy_function(flip_parts_left_right, [parts_x, parts_y, parts_v, [[2, 1], [5, 4]], num_parts], [tf.float32, tf.float32, tf.int32])

    part_visibilities = tf.reshape(parts_v, tf.stack([num_bboxes, num_parts]))

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        # set the number of images in your tfrecords file
        num_images = c
        print("restoring {} files from {}".format(num_images, f))
        for i in range(num_images):

            fields = ['image id', 'bbox xmin', 'bbox ymin', 'bbox xmax', 'bbox ymax', 'num bboxes', 'parts x', 'parts y', 'vis raw', 'vis reshape']
            outputs = sess.run([image_id, xmin, ymin, xmax, ymax, num_bboxes, parts_x, parts_y, parts_v, part_visibilities])

            for fieldname, value in zip(fields, outputs):
                print(fieldname.ljust(12) + ': ' + str(type(value)).ljust(25) + ': ' + str(value))
            print('---')

        coord.request_stop()
        coord.join(threads)