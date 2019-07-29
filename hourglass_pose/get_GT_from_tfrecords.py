import numpy as np
import tensorflow as tf
import argparse
import json
import os


def get_ground_truth(
        tfrecords,
        num_parts=11,
        num_examples=1500
    ):
    with tf.Session() as sess:

        # A producer to generate tfrecord file paths.
        filename_queue = tf.train.string_input_producer(
            tfrecords
        )

        # Construct a Reader to read examples from the tfrecords file
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Parse an Example to access the Features
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/id': tf.FixedLenFeature([], tf.string),
                'image/filename': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/count': tf.FixedLenFeature([], tf.int64),
                'image/object/bbox/score': tf.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
                'image/object/parts/x': tf.VarLenFeature(dtype=tf.float32),  # x coord for all parts and all objects
                'image/object/parts/y': tf.VarLenFeature(dtype=tf.float32)
            }
        )

        # Get the image id, and the filename
        image_id = features['image/id']
        filename = features['image/filename']

        print(filename)

        num_bboxes = tf.cast(features['image/object/bbox/count'], tf.int32)

        parts_x = tf.expand_dims(features['image/object/parts/x'].values, 0)
        parts_y = tf.expand_dims(features['image/object/parts/y'].values, 0)

        parts = tf.concat(axis=0, values=[parts_x, parts_y])
        parts = tf.transpose(parts, [1, 0])
        parts = tf.reshape(parts, [-1, num_parts * 2])

        # Copies the image_id, [num_bboxes] times
        image_ids = tf.tile([[image_id]], [num_bboxes, 1])
        image_ids.set_shape([None, 1])

        # Initialize the session.
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        # Start the coordinator and queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Initialize the dictionary where we'll store everything.
        results = {"image_id":[], "filename":[], "keypoints":[]}

        # Initialize the list of things we're trying to store.
        id_list = []
        keypoints = []
        fname_list = []

        # Parse each example into the relevant parts.
        for i in xrange(num_examples):
            [example_id, example_fname, example_parts] = sess.run([image_id, filename , parts])

            # Keep track of the id's and the filenames.
            id_list.append(int(example_id))
            fname_list.append(example_fname)

            # print example_parts
            B = example_parts[0]
            W = example_parts[1]

            # Get the X and Y values of the Black and White mice.
            xB = B[0::2]
            yB = B[1::2]
            xW = W[0::2]
            yW = W[1::2]

            # Expand dims so we can stack them.
            xB = np.expand_dims(xB, axis= 0)
            xW = np.expand_dims(xW, axis= 0)
            yB = np.expand_dims(yB, axis= 0)
            yW = np.expand_dims(yW, axis= 0)

            # Stack the x's and y's.
            black_m = np.concatenate((xB,yB), axis=0)
            white_m = np.concatenate((xW,yW), axis=0)

            # Expand dims again so we can stack them.
            black_m = np.expand_dims(black_m, axis=0)
            white_m = np.expand_dims(white_m, axis=0)

            # Stack the black and white mouse together.
            both_m = np.concatenate((black_m, white_m), axis=0)

            # Expand the dims of the frame so we can stack them.
            both_m = np.expand_dims(both_m, axis=0)

            # Stack everything together.
            if i == 0:
                keypoints = both_m
            else:
                keypoints = np.concatenate((keypoints, both_m), axis=0)

        # Store everything as a list of dicts.
        results['keypoints'] = keypoints.tolist()
        results['image_id'] = id_list
        results['filename'] = fname_list

        coord.request_stop()
        coord.join(threads)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--num_parts', dest='num_parts',
                        help='the number of parts in the model', type=int,
                        required=True)

    parser.add_argument('--num_examples', dest='num_examples',
                        help='the number of examples to parse', type=int,
                        required=True)

    parser.add_argument('--savename', dest='savename',
                        help='the fullpath (with extension) to save the file to', type=str,
                        required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    results= get_ground_truth(
        tfrecords=args.tfrecords,
        num_parts=args.num_parts,
        num_examples=args.num_examples
    )
    # If we get something, dump it.
    if len(results) > 0:
        with open(args.savename, 'wb') as fp:
            json.dump(results, fp)
    else:
        print("Ground truth generation failed.")

