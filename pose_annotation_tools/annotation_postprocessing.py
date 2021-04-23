from pose_annotation_tools.tfrecord_util import *
from pose_annotation_tools.json_util import *
from pose_annotation_tools.priors_generator import *
import random
import yaml
import json
import os
import glob
import argparse


def make_annot_dict(project):
    """"
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), creates a
    cleaned-up json intermediate file, for use in generating tfrecords and analyzing annotator performance.

    project : string
        The absolute path to your annotation project

    correct_flips : bool
        Defaults to True
        Whether to attempt to correct left/right errors in raw annotations.

    Example
    --------
    make_annot_dict('/path/to/savedir/my_project')
    --------
    """
    # check that everything is where we want it to be
    config_fid = os.path.join(project,'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    annotation_file = os.path.join(project, 'annotation_data', config['manifest_name'])
    if not os.path.exists(annotation_file):  # annotations file isn't where we expected it to be.
        raise SystemExit("I couldn't find an annotation file at " + annotation_file)

    _, ext = os.path.splitext(annotation_file)
    if ext == '.csv':
        csv_to_dict(project)
    elif ext == '.manifest':
        manifest_to_dict(project)


def make_clean_dir(output_dir):
    if os.path.exists(output_dir):
        # remove all old tfrecords and priors for safety
        oldrecords = []
        for filetype in ['tfrecord','prior']:
            oldrecords.extend(glob.glob(os.path.join(output_dir, '*' + filetype + '*')))
        for f in oldrecords:
            try:
                os.remove(f)
            except OSError as e:
                print('Error: %s : %s' % (f, e.strerror))

    else:
        os.makedirs(output_dir)


def prepare_detector_training_data(project):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    and prior files that are used to train MARS's detectors and pose estimators for black and white mice.
    """
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    detector_list = config['detection']
    detector_names = detector_list.keys()

    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)
    random.shuffle(D)

    for detector in detector_names:
        if config['verbose']:
            print('Generating ' + detector + ' detection training files...')

        output_dir = os.path.join(project, 'detection', 'tfrecords_detection_' + detector)
        make_clean_dir(output_dir)
        v_info = prep_records_detection(D, detector_list[detector])
        write_to_tfrecord(v_info, output_dir)

        if config['verbose']:
            print('done.')


def prepare_pose_training_data(project):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    files that are used to train MARS's pose estimator.
    """
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    pose_list = config['pose']
    pose_names = pose_list.keys()

    # extract info from annotations
    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)
    random.shuffle(D)

    for pose in pose_names:
        if config['verbose']:
            print('Generating ' + pose + ' pose training files...')

        output_dir = os.path.join(project, 'pose', 'tfrecords_pose_' + pose)
        make_clean_dir(output_dir)
        v_info = prep_records_pose(D, pose_list[pose])
        write_to_tfrecord(v_info, output_dir)

        if config['verbose']:
            print('done.')



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


def annotation_postprocessing(project):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    files that are used to train MARS's detectors and pose estimators.
    project : string
        The absolute path to the project directory.
    Example
    --------
    process_annotations('D:\\my_project')
    --------
    """
    # extract info from annotations into an intermediate dictionary file
    make_annot_dict(project)

    # save tfrecords
    prepare_detector_training_data(project)
    prepare_pose_training_data(project)

    # make priors
    make_project_priors(project)


if __name__ ==  '__main__':
    """
    annotation_postprocessing command line entry point
    Arguments:
        project 	The absolute path to the project directory.
    """

    parser = argparse.ArgumentParser(description='postprocess and package manual pose annotations', prog='annotation_postprocessing')
    parser.add_argument('project', type=str, help="absolute path to project folder.")
    args = parser.parse_args(sys.argv[1:])

    annotation_postprocessing(args.project)
