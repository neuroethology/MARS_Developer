from tfrecord_util import *
from json_util import *


def create_tfrecords(**kwargs):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    files that are used to train MARS's detectors and pose estimators for black and white mice.

    annotation_file : string
        The absolute path to the file containing keypoint annotations (either a .manifest or a .csv file.)

    im_path : string
        The absolute path to the folder containing all images that were annotated.

    keypoints : list of strings
        The name of the keypoints that were annotated. If empty, will attempt to extract this from the annotation file.

    save_path : string
        The absolute path to the folde where the tfrecords should be saved. If empty, will save with annotation_file.

    overwrite_json : bool
        If true, overwrites intermediate json files during tfrecord generation.

    manifest_key : string
        For manifest files (from Amazon Ground Truth), this is a key set in run_labeling_job.ipynb;
        if you used this script and modified the key name, pass the updated name here.

    Example
    --------
    kpts = ['Nose','EarL','EarR','Neck','HipL','HipR','Tail']
    prepare_training_data('/media/tracking/human_annotations.manifest', /media/tracking/movies', kpts)
    --------
    """
    prepare_detector_training_data(**kwargs)
    prepare_pose_training_data(**kwargs)


def make_annot_dict(annotation_file, im_path='', save_file='', keypoints=[], manifest_key='annotatedResult'):
    """"
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), creates a
    cleaned-up json intermediate file, for use in generating tfrecords.

    annotation_file : string
        The absolute path to the file containing keypoint annotations (either a .manifest or a .csv file.)

    im_path : string
        The absolute path to the folder containing all images that were annotated.

    save_file : string
        The absolute path to the folder where the json will be saved. If empty, will save in the same directory
        as annotation_file.

    keypoints : list of strings
        The name of the keypoints that were annotated. If empty, will attempt to extract this from the annotation file.

    manifest_key : string
        For manifest files (from Amazon Ground Truth), this is a key set in run_labeling_job.ipynb;
        if you used this script and modified the key name, pass the updated name here.

    Example
    --------
    kpts = ['Nose','EarL','EarR','Neck','HipL','HipR','Tail']
    make_annot_dict('/media/tracking/human_annotations.manifest', /media/tracking/movies', keypoints=kpts)
    --------
    """
    if not im_path:
        im_path, _ = os.path.split(annotation_file)

    if not save_file:
        save_path, _ = os.path.split(annotation_file)
        save_file = os.path.join(save_path, 'processed_keypoints.json')

    _,ext = os.path.splitext(annotation_file)
    if ext == '.csv':
        csv_to_dict(annotation_file, im_path, save_file, keypoints)
    elif ext == '.manifest':
        manifest_to_dict(annotation_file, im_path, save_file, keypoints, manifest_key=manifest_key)
