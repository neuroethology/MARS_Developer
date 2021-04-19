from tfrecord_util import *
from json_util import *
import yaml

# detection and pose preparation functions -----------------------------------------------------------------------------
def prepare_detector_training_data(project, correct_flips = True, split_train_val_test = False, overwrite_json = False, manifest_key='annotatedResult'):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    files that are used to train MARS's detectors and pose estimators for black and white mice.
    """
    im_path = os.path.join(project,'annotation_data','raw_images')

    # extract info from annotations
    dictionary_file_path = os.path.join(project,'annotation_data','processed_keypoints.json')  # Path to save the intermediate dictionary file.
    if overwrite_json or not os.path.exists(dictionary_file_path):
        make_annot_dict(project, correct_flips=correct_flips)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)

    # TODO: don't hard-code black/white
    detection_black_tfrecord_output_name = os.path.join(project,'detection','detection_black_tfrecords')
    detection_white_tfrecord_output_name = os.path.join(project,'detection','detection_white_tfrecords')
    if not os.path.exists(detection_black_tfrecord_output_name):
        os.makedirs(detection_black_tfrecord_output_name)
    if not os.path.exists(detection_white_tfrecord_output_name):
        os.makedirs(detection_white_tfrecord_output_name)

    v_infob = make_bbox_dict(D, im_path, 'black')
    v_infow = make_bbox_dict(D, im_path, 'white')
    v_info = list(zip(v_infob, v_infow))
    random.shuffle(v_info)
    v_infob, v_infow = zip(*v_info)

    write_to_tfrecord(v_infob, detection_black_tfrecord_output_name, split_train_val_test=split_train_val_test)
    write_to_tfrecord(v_infow, detection_white_tfrecord_output_name, split_train_val_test=split_train_val_test)


def prepare_pose_training_data(project, correct_flips = True, split_train_val_test = False, overwrite_json = False):
    """
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), create the tfrecord
    files that are used to train MARS's pose estimator.
    """

    im_path = os.path.join(project,'annotation_data','raw_images')

    # extract info from annotations
    dictionary_file_path = os.path.join(project,'annotation_data','processed_keypoints.json')  # Path to save the intermediate dictionary file.
    if overwrite_json or not os.path.exists(dictionary_file_path):
        make_annot_dict(project, correct_flips=correct_flips)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)

    pose_estimation_tfrecord_output_name = os.path.join(project,'pose','pose_estimation_tfrecords')  # Path to save the pose estimation tfrecord.
    if not os.path.exists(pose_estimation_tfrecord_output_name):
        os.makedirs(pose_estimation_tfrecord_output_name)
    v_info = make_pose_dict(D, im_path)
    random.shuffle(v_info)
    write_to_tfrecord(v_info, pose_estimation_tfrecord_output_name, split_train_val_test=split_train_val_test)


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
    create_tfrecords('/media/tracking/human_annotations.manifest', '/media/tracking/movies', kpts)
    --------
    """
    prepare_detector_training_data(**kwargs)
    prepare_pose_training_data(**kwargs)


def make_annot_dict(project, correct_flips=True):
    """"
    Given human annotations (from Amazon Ground Truth or from the DeepLabCut annotation interface), creates a
    cleaned-up json intermediate file, for use in generating tfrecords and analyzing annotator performance.

    project : string
        The absolute path to your annotation project

    manifest_name : string
        Defaults to output.manifest
        The name of the file containing raw annotations. Can be either a .manifest from Amazon or a .csv from DeepLabCut.

    correct_flips : bool
        Defaults to True
        Whether to attempt to correct left/right errors in raw annotations.

    Example
    --------
    make_annot_dict('/path/to/savedir/my_project')
    --------
    """

    # read the annotation config file
    config_fid = os.path.join(project,'project_config.yml')
    with open(config_fid) as f:
        cfg = yaml.load(f)

    im_path = os.path.join(project,'annotation_data','raw_images')
    save_file = os.path.join(project,'annotation_data','processed_keypoints.json')
    annotation_file = os.path.join(project,'annotation_data',cfg['manifest_name'])
    keypoints    = cfg['keypoints']

    if not os.path.exists(annotation_file): # annotations file isn't where we expected it to be.
        raise SystemExit("I couldn't find an annotation file at " + annotation_file)

    _,ext = os.path.splitext(annotation_file)
    if ext == '.csv':
        csv_to_dict(annotation_file, im_path, save_file, cfg)
    elif ext == '.manifest':
        manifest_to_dict(annotation_file, im_path, save_file, cfg)
