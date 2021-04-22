
"""
generate priors for the bbs
You can hand defined them or use prior.py utilties to use your training dataset to cluster the aspect ratios 
of the bounding boxes
"""

#clustering for generation of priors
import os
import sys
import argparse

from priors import *
import pickle

def _parse_function(example_proto):
	features = {
		'image/id' : tf.io.FixedLenFeature([], tf.string),
		'image/encoded'  : tf.io.FixedLenFeature([], tf.string),
		'image/height' : tf.io.FixedLenFeature([], tf.int64),
		'image/width' : tf.io.FixedLenFeature([], tf.int64),
		'image/object/bbox/xmin' : tf.io.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymin' : tf.io.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/xmax' : tf.io.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymax' : tf.io.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/count' : tf.io.FixedLenFeature([], tf.int64)
	}
	return tf.io.parse_single_example(example_proto, features)

def len_util(tf_record_dataset):
    """
    Counts the number of entries in a TFRecordDataset
    """
    c = 0
    for record in tf.python_io.tf_record_iterator(tf_record_dataset):
        c += 1
    return c

def convert(tf_r):
    """
    INPUTS:
        tf_r (str): path to tfrecords file
    OUTPUTS:
        dataset (list): a list of dictionaries to be used in
            generate_aspect_ratios
    """
    # generate_aspect_ratios takes as input a list of image distionaries
    dataset = []

    # Loop through all the tfrecords provided
    for r in tf_r:
        # Make TFRecord Dataset object and add mapping 
        raw_dataset = tf.compat.v1.data.TFRecordDataset(r)
        parsed_dataset = raw_dataset.map(_parse_function)

        # Make an iterator for our dataset 
        iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)
        next_element = iterator.get_next()

        # Count the number of entries in our dataset
        num_records = len_util(r)

        # Enumerate over our parsed dataset
        with tf.Session() as sess:
            for i in range(num_records):
                # Get the current record
                record = sess.run(next_element)

                # Make new dict
                image_data = {}

                # Extract and create the bbox dict
                image_data['object'] = {
                        'bbox': {
                            'xmax': record['image/object/bbox/xmax'].values,
                            'xmin': record['image/object/bbox/xmin'].values,
                            'ymax': record['image/object/bbox/ymax'].values,
                            'ymin': record['image/object/bbox/ymin'].values}
                        }

                # Retrieve the id, width, and height of the image
                image_data['width'] = record['image/width']
                image_data['height'] = record['image/height']
                image_data['id'] = record['image/id']

                # Append to dataset list
                dataset.append(image_data)

    return dataset

def generate_priors_from_data(dataset=None, aspect_ratios=None):
    """
    INPUTS:
        dataset (str): path to tfrecords file(s) to create priors from
        aspect_ratios (str): a list of hand-defined aspect ratios to construct the priors from
    OUTPUTS:
        priors (list): list of the priors
    """
    # Generate aspect ratios from the dataset
    if (aspect_ratios is None) and (dataset is not None):
        # Convert dataset from tfrecords format to list of dicts 
        dataset = convert(dataset)

        # Generate aspect ratios using list of dicts
        aspect_ratios = generate_aspect_ratios(dataset, num_aspect_ratios=11, visualize=False, warp_bboxes=True)

    # Need to convert aspect ratios to proper format
    elif (aspect_ratios is not None) and (dataset is None):
        aspect_ratios = [float(aspect_ratio) for aspect_ratio in aspect_ratios]

    else:
        print("Only provide either the hand-defined aspect ratios or tfrecords file(s) to construct the priors from")
        sys.exit()
    # Generate the priors with either hand-defined priors or priors generated from data
    priors = generate_priors(aspect_ratios, min_scale=0.1, max_scale=0.95, restrict_to_image_bounds=True)

    return priors

def parse_args():
    parser = argparse.ArgumentParser(description='Generate aspect ratios and priors from training data or hand-defined aspect ratios.')

    parser.add_argument('--dataset', dest='dataset',
						help='binary tfrecords file that contains the training data to generate priors',
                        type=str,
						required=False,
                        nargs='+')

    parser.add_argument('--aspect_ratios',
                        dest='aspect_ratios',
                        help='[list of aspect ratios used to hand generate the priors]',
                        type=str,
                        nargs='+',
                        required=False)

    parser.add_argument('--save_path',
                        dest='save_path',
                        help='Path to save priors to',
                        type=str,
                        required=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Generate priors from either data or hand-defined set of aspect ratios
    p = generate_priors_from_data(
            dataset=args.dataset,
            aspect_ratios=args.aspect_ratios
            )

    # Save priors
    with open(args.save_path, 'wb') as f:
        pickle.dump(p, f)

if __name__ == "__main__":
	main()
