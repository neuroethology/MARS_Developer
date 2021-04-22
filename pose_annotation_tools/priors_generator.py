import sys, os
import yaml
import pickle
import glob
from pose_annotation_tools.priors import *


def _parse_function(example_proto):
    features = {
        'image/id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/count': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example_proto, features)


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
        raw_dataset = tf.data.TFRecordDataset(r)
        parsed_dataset = raw_dataset.map(_parse_function)

        # Enumerate over our parsed dataset
        for _, record in enumerate(parsed_dataset):
            # Make new dict
            image_data = {}

            # Extract and create the bbox dict
            image_data['object'] = {
                'bbox': {
                    'xmax': record['image/object/bbox/xmax'],
                    'xmin': record['image/object/bbox/xmin'],
                    'ymax': record['image/object/bbox/ymax'],
                    'ymin': record['image/object/bbox/ymin']}
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


def make_project_priors(project):

    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    detector_list = config['detection']
    detector_names = detector_list.keys()

    for detector in detector_names:
        if config['verbose']:
            print('Generating ' + detector + ' priors...')

        output_dir = os.path.join(project, 'detection', 'tfrecords_detection_' + detector)
        record_list = glob.glob(os.path.join(output_dir, 'train_dataset-*'))
        priors = generate_priors_from_data(dataset=record_list)

        with open(os.path.join(project, 'detection', 'priors_' + detector + '.pkl'), 'wb') as fp:
            pickle.dump(priors, fp)

        if config['verbose']:
            print('done.')
