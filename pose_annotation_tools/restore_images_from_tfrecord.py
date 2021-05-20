import tensorflow.compat.v1 as tf
import cv2
import os
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False


def restore(tfrecords_filenames, output_path):
    # this is a script to extract images from tfrecord files, for sanity-checking.
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

    # here a path to tfrecords file as list
    fq = tf.train.string_input_producer(tfrecords_filenames, num_epochs=totalFiles)
    reader = tf.TFRecordReader()
    _, v = reader.read(fq)
    fk = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/synset': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], tf.string, default_value='')
        }

    ex = tf.parse_single_example(v, fk)
    image = tf.image.decode_jpeg(ex['image/encoded'], dct_method='INTEGER_ACCURATE')
    label = tf.cast(ex['image/class/synset'], tf.string)
    fileName = tf.cast(ex['image/filename'], tf.string)
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

            im_, lbl, fName = sess.run([image, label, fileName])

            lbl_ = lbl.decode("utf-8")

            savePath = os.path.join(output_path, lbl_)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            fName_ = os.path.join(savePath, 'image' + f'{i:07d}' + '_' + fName.decode("utf-8"))

            # change the image save path here
            cv2.imwrite(fName_, im_)

        coord.request_stop()
        coord.join(threads)