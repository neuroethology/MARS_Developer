# from pose_annotation_tools.tfrecord_util import *
from pose_annotation_tools.json_util import *
# from pose_annotation_tools.priors_generator import *
import random
import yaml
import json
import os
import glob
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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


def plot_frame(project, fr, markersize=8, figsize=[15, 20]):
    # plots annotations from all workers plus the worker median for an example frame.

    config_fid = os.path.join(project,'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)

    plt.rcParams['figure.figsize'] = figsize
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    markers = 'vosd*p'

    im = mpimg.imread(D[fr]['image'])
    plt.imshow(im, cmap='gray')

    # plot the labels from each individual worker:
    for mouse in config['animal_names']:
        for w, [x, y] in enumerate(zip(D[fr]['ann_' + mouse]['X'], D[fr]['ann_' + mouse]['Y'])):
            for i, [px, py] in enumerate(zip(x, y)):
                plt.plot(px * D[fr]['width'], py * D[fr]['height'],
                         colors[i % 9], marker=markers[w % 6], markersize=markersize)

        for i, [px, py] in enumerate(zip(D[fr]['ann_' + mouse]['med'][1], D[fr]['ann_' + mouse]['med'][0])):
            plt.plot(np.array(px) * D[fr]['width'], np.array(py) * D[fr]['height'],
                     'k', marker='o', markeredgecolor='w', markeredgewidth=math.sqrt(markersize), markersize=markersize)
    plt.show()

def plot_frame(project, fr, markersize=8, figsize=[15, 20]):
    # plots annotations from all workers plus the worker median for an example frame.

    config_fid = os.path.join(project,'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)

    plt.rcParams['figure.figsize'] = figsize
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    markers = 'vosd*p'

    im = mpimg.imread(D[fr]['image'])
    plt.imshow(im, cmap='gray')

    # plot the labels from each individual worker:
    for mouse in config['animal_names']:
        for w, [x, y] in enumerate(zip(D[fr]['ann_' + mouse]['X'], D[fr]['ann_' + mouse]['Y'])):
            for i, [px, py] in enumerate(zip(x, y)):
                plt.plot(px * D[fr]['width'], py * D[fr]['height'],
                         colors[i % 9], marker=markers[w % 6], markersize=markersize)

        for i, [px, py] in enumerate(zip(D[fr]['ann_' + mouse]['med'][1], D[fr]['ann_' + mouse]['med'][0])):
            plt.plot(np.array(px) * D[fr]['width'], np.array(py) * D[fr]['height'],
                     'k', marker='o', markeredgecolor='w', markeredgewidth=math.sqrt(markersize), markersize=markersize)
    plt.show()


def plot_summary(project, xlim=[0, 50], pixels_per_cm=None):
    # pixels_per_cm = 37.795
    # project = '/media/storage/CRIM13_sample_project'
    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)

    nSamp = len(D)
    nKpts = len(D[0]['ann_label'])

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    dashes = {'white': 'dotted', 'black': 'dashed', 'both': 'solid'}
    # colors = 'rbg'
    for mouse in ['white', 'black']:
        dMean   = np.zeros((nKpts, nSamp))  # average worker-gt distance
        dMedian = np.zeros((nKpts, nSamp))  # median worker-gt distance
        dMin    = np.zeros((nKpts, nSamp))  # performance of best worker on a given frame
        dMax    = np.zeros((nKpts, nSamp))  # performance of worst worker on a given frame

        for fr, frame in enumerate(D):
            X = np.array(frame['ann_' + mouse]['X']) * D[0]['width']
            Y = np.array(frame['ann_' + mouse]['Y']) * D[0]['height']
            trial_dists = []
            for i, [pX, pY] in enumerate(zip(X, Y)):
                mX = np.median(np.delete(X, i, axis=0), axis=0)
                mY = np.median(np.delete(Y, i, axis=0), axis=0)
                trial_dists.append(np.sqrt(np.square(mX - pX) + np.square(mY - pY)))
            trial_dists = np.array(trial_dists)

            dMean[:, fr]   = np.mean(trial_dists, axis=0)
            dMedian[:, fr] = np.median(trial_dists, axis=0)
            dMin[:, fr]    = np.min(trial_dists, axis=0)
            dMax[:, fr]    = np.max(trial_dists, axis=0)

        bins = 10000
        binrange = [-1 / bins, np.max(dMax) + 1 / bins]

        for c, use in enumerate([dMin, dMean, dMedian, dMax]):
            for p, pt in enumerate(use):
                counts, usedbins = np.histogram(pt, bins, range=binrange, density=True)
                ax[int(p / 4), p % 4].plot(usedbins[1:], counts.cumsum() / bins * binrange[1], ls=dashes[mouse])
        for p,label in enumerate(D[0]['ann_label']):
            ax[int(p / 4), p % 4].set_title(label)
            ax[int(p / 4), p % 4].set_xlim(xlim)


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

    # # save tfrecords
    # prepare_detector_training_data(project)
    # prepare_pose_training_data(project)
    #
    # # make priors
    # make_project_priors(project)


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
