import numpy as np
import os,sys
import pandas as pd
from PIL import Image
import json
from scipy.spatial.distance import cdist
import yaml


def count_workers(data):
    nWorkers = []
    for f,frame in enumerate(data):
        if 'annotatedResult' in frame.keys():  # check if this frame has at least one set of annotations
            nWorkers.append(len(frame['annotatedResult']['annotationsFromAllWorkers']))
    return nWorkers


def apply_flip_correction(frame, meds, keypoints, pair):
    if not pair[0] in keypoints or not pair[1] in keypoints:
        raise SystemExit('annot_config error: one or more entries of check_pairs is not a member of keypoints. Please check project_config.yaml')

    i1 = keypoints.index(pair[0])
    i2 = keypoints.index(pair[1])

    for rep in range(3):  # repeat 3 times for stability
        for w, worker in enumerate(frame.swapaxes(0, 2).swapaxes(1, 2)):
            d1 = cdist(worker[[i1, i2], :], [meds[i1, :]])
            d2 = cdist(worker[[i1, i2], :], [meds[i2, :]])
            if (d1[0] > d1[1]) and (d2[1] > d2[0]):
                frame[[i1, i2], :, w] = frame[[i2, i1], :, w]

            # re-compute the medians:
            meds = np.nanmedian(frame, axis=2)

    return frame, meds

def create_mask_bad_wks(data, workers2del, nKpts, nWorkers):
    null_mask_arr = np.ones((2, nWorkers))

    workers_list = data['annotatedResult']['annotationsFromAllWorkers']
    valid_idx = [i for i, x in enumerate([w['workerId'] for w in workers_list])
            if x not in workers2del]
   
    for i in valid_idx:
        null_mask_arr[:,i] = np.zeros((null_mask_arr.shape[0],1)).ravel()
   
    null_mask_arr = np.repeat(null_mask_arr[np.newaxis, :, :], nKpts, axis=0)
   
    return null_mask_arr

def hold_one_out_dist(fr_df, frame_num):
    cols = ['x', 'y']
    all_wkrs_dist = pd.DataFrame()
    for wk in fr_df['worker ID'].unique():
        df_worker = fr_df[fr_df['worker ID'].eq(wk)] \
                                 .loc[:,'x':'color'].set_index(['color']).sort_index()
                                 
        dist_df = fr_df[~fr_df['worker ID'].eq(wk)] \
                                           .loc[:,'x':'color'] \
                                           .groupby(['color']) \
                                           .median().sort_index() \
                                           .assign(
                                                distance = lambda df_:
                                                    np.linalg.norm(df_[cols] - df_worker[cols], axis=1)) \
                                           .loc[:,'distance'].reset_index() \
                                           .assign(
                                                worker = [wk for i in range(2)],
                                                frame = [frame_num for i in range(2)])
        all_wkrs_dist = pd.concat([dist_df, all_wkrs_dist], ignore_index=True)
   
    return all_wkrs_dist

def detect_bad_workers(project, manifest_file, thr_dist = 300, per_del = 0.15):
    
    f = open(os.path.join(project, 'annotation_data', manifest_file), 'r')
    st = f.read()
    st = "[\n" + st + "\n]"
    st = st.replace("\\","").replace("\"{","{") \
            .replace("}\"}","}}").replace("{\"source",",{\"source") \
            .replace(',', '', 1)
    while True:
        try:
            output_manifest = json.loads(st)
            break
        except Exception as e:
            print(str(e))

    f.close()
    
    frames_dist_df = pd.DataFrame()
    for fr_id, data in enumerate(output_manifest):

        frame = data['annotatedResult']['annotationsFromAllWorkers']
        fr_df = pd.DataFrame()

        for wk_id, worker_content in enumerate(frame):
            worker = worker_content['annotationData']['content']['annotatedResult']
            k_pts = worker['keypoints']
            pos_blk_nose = [(k_pts[i]['x'],k_pts[i]['y']) for i, x in enumerate (k_pts) if x['label'] == 'black mouse nose']
            pos_wht_nose = [(k_pts[i]['x'],k_pts[i]['y']) for i, x in enumerate (k_pts) if x['label'] == 'white mouse nose']

            wht_info = pd.DataFrame([{'x':pos_wht_nose[0][0],
                                      'y':pos_wht_nose[0][1],
                                      'color':'white',
                                      'worker ID': frame[wk_id]['workerId']}])
            blk_info = pd.DataFrame([{'x':pos_blk_nose[0][0],
                                      'y':pos_blk_nose[0][1],
                                      'color':'black',
                                      'worker ID': frame[wk_id]['workerId']}])

            fr_df = pd.concat([fr_df, wht_info], ignore_index=True)
            fr_df = pd.concat([fr_df, blk_info], ignore_index=True)

        frames_dist_df = pd.concat([frames_dist_df, hold_one_out_dist(fr_df, fr_id)], ignore_index=True)

    
    workers_total_annotations = frames_dist_df.worker.value_counts()
    bad_workers_list = frames_dist_df[frames_dist_df['distance']>thr_dist].groupby(['worker']) \
                                                                .size() \
                                                                .div(workers_total_annotations) \
                                                                .loc[lambda df: df > per_del] \
                                                                .index.tolist()
    
    return bad_workers_list
                                                           


def manifest_to_dict(project):
    """
    Converts an annotation file generated by AMT/Ground Truth into the MARS json format.
    """
    config_fid = os.path.join(project,'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    manifest_file  = config['manifest_name']
    save_file      = os.path.join(project,'annotation_data','processed_keypoints.json')
    animal_names   = config['animal_names'] if config['animal_names'] else config['species']
    species        = config['species']
    keypoint_names = config['keypoints']
    check_pairs    = config['check_pairs']
    verbose        = config['verbose']
    nKpts          = len(keypoint_names)

    fid = open(os.path.join(project, 'annotation_data', manifest_file), 'r')
    data = []
    for line in fid:
        data.append(json.loads(line))
    nWorkers = count_workers(data)
    nSamp = len(data)

    # loop over frames in the manifest file
    images = ['']*nSamp      # store local paths to labeled images
    hits = [False]*nSamp     # track which images have annotations (hopefully all of them)

    sourceStr = os.path.dirname(data[0]['source-ref'])  # image path on AWS
    localStr = os.path.join(project, 'annotation_data', 'raw_images')  # image path locally

    D = []
    if verbose:
        print('Processing manifest file...')
    print(len(data))
    
    print('Detecting bad workers')
    bad_workers_list = detect_bad_workers(project, manifest_file, thr_dist = 300, per_del = 0.15)
    
    for f, sample in enumerate(data):
        if f and not f % 1000 and verbose:
            print('  frame '+str(f))
        if 'annotatedResult' in sample.keys():  # check if this frame has at least one set of annotations
            hits[f] = True
            images[f] = sample['source-ref']
            images[f] = images[f].replace(sourceStr, localStr).replace('/', os.path.sep)

            # Use the path to the image data to open the image.
            im = Image.open(images[f])
            im = (np.asarray(im)).astype(float)

            frame_dict = {
                'image': images[f],
                'height': im.shape[0],
                'width': im.shape[1],
                'frame_id': images[f],
                'ann_label': keypoint_names,
                'ann': []
            }

            rawPts = {n: np.zeros((nKpts, 2, nWorkers[f])) for n in animal_names}
            for w, worker in enumerate(sample['annotatedResult']['annotationsFromAllWorkers']):
                # the json of annotations from each worker is stored as a string for security reasons.
                # we'll use eval to convert it into a dict:
                annot = eval(worker['annotationData']['content'])

                # now we can unpack this worker's annotations for each keypoint:
                for pt in annot['annotatedResult']['keypoints']:
                    animal = next((n for n in animal_names if n in pt['label']),species)
                    kpt_clean = pt['label'].replace(animal, '').replace(species, '').strip()
                    if kpt_clean not in keypoint_names:
                        continue
                    part = keypoint_names.index(kpt_clean)

                    rawPts[animal][part, 0, w] = pt['x']/im.shape[1]
                    rawPts[animal][part, 1, w] = pt['y']/im.shape[0]
            
            mask_bad_wks = create_mask_bad_wks(sample, bad_workers_list, nKpts, nWorkers[f])            
            for animal in animal_names:
                rawPts[animal] = np.ma.array(rawPts[animal], mask = mask_bad_wks).filled(np.nan)
                
                if check_pairs:  # adjust L/R assignments to try to find better median estimates.
                    meds = np.nanmedian(rawPts[animal], axis=2)
                    for pair in check_pairs:
                        rawPts[animal], meds = apply_flip_correction(rawPts[animal], meds, keypoint_names, pair)

                # Compute some statistics for the tfrecord and append.
                frame_dict['ann_'+animal] = make_animal_dict(rawPts[animal], im.shape)

            D.append(frame_dict)

    # save info to file
    with open(save_file, 'w') as fp:
        json.dump(D, fp)
    if verbose:
        print('Ground-truth keypoint locations extracted!')


def csv_to_dict(project):
    """
    Converts manual annotations created by DeepLabCut from csv into the format created by MARS from
    AMT or Ground Truth manifest files. MARS can then use that json to create tfrecords for training
    the detection and pose models.
    """
    config_fid = os.path.join(project,'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    csv_file       = config['manifest_name']
    save_file      = os.path.join(project,'annotation_data','processed_keypoint_annotations.json')
    animal_names   = config['animal_names'] if config['animal_names'] else config['species']
    species        = config['species']
    keypoint_names = config['keypoints']
    check_pairs    = config['check_pairs']
    verbose        = config['verbose']
    nKpts          = len(keypoint_names)
    im_path        = os.path.join(project, 'annotation_data', 'raw_images')  # image path locally

    with open(csv_file) as datafile:
        next(datafile)
        if "individuals" in next(datafile):
            header = list(range(4))
            multianimal = True
        else:
            header = list(range(3))
            multianimal = False
    data = pd.read_csv(csv_file, index_col=0, header=header)

    worker_names = list(set(list(data.columns.get_level_values(0))))  # find all the annotators.
    nWorkers = len(set(worker_names))

    # loop over frames in the manifest file
    D = []
    print('Processing csv file...')
    for f, image in enumerate(data.index):
        if f and not f % 1000 and verbose:
            print('  frame '+str(f))

        # Use the path to the image data to open the image.
        im_name = os.path.split(str(image))[-1]
        im = Image.open(os.path.join(im_path, im_name))
        im = (np.asarray(im)).astype(float)

        frame_dict = {
            'image': str(image),
            'height': im.shape[0],
            'width': im.shape[1],
            'frame_id': im_name,
            'ann_label': keypoint_names,
            'ann': []
        }

        # convert our dataframe to a dict
        annot = data.loc[image]
        # TODO: update this to handle the multianimal case.
        annot = {level0: {level1: annot.xs([level0, level1]).to_dict() for level1 in annot.index.levels[1]}
                 for level0 in annot.index.levels[0]}

        # extract info for each worker
        rawPts = {n:np.zeros((nKpts, 2, nWorkers)) for n in animal_names}
        for w, worker in annot.items():

            for label,pt in worker.items():
                # todo: get color if none was provided!
                animal = next((n for n in animal_names if n in label), species)
                idx = keypoint_names.index(label.replace(animal ,'').replace(species ,'').strip())

                rawPts[animal][idx, 0, w] = pt['x'].im.shape[1]
                rawPts[animal][idx, 1, w] = pt['y'].im.shape[0]

        for animal in animal_names:
            if check_pairs and nWorkers>1:  # adjust L/R assignments to try to find better median estimates.
                meds = np.median(rawPts[animal], axis=2)
                for pair in check_pairs:
                    rawPts[animal], meds = apply_flip_correction(rawPts[animal], meds, keypoint_names, pair)

            # Compute some statistics for the tfrecord and append.
            frame_dict['ann_'+animal] = make_animal_dict(rawPts[animal], im.shape)

        D.append(frame_dict)

    # save info to file
    with open(save_file, 'w') as fp:
        json.dump(D, fp)
    print('Ground-truth keypoint locations extracted!')


def make_animal_dict(pts, im_shape):
    X = pts[:, 0, :].T
    Y = pts[:, 1, :].T

    mX = np.nanmedian(X, axis=0)
    mY = np.nanmedian(Y, axis=0)

    muX = np.nanmean(X, axis=0)
    muY = np.nanmean(Y, axis=0)

    stdX = np.nanstd(Y, axis=0)
    stdY = np.nanstd(Y, axis=0)

    # bounding box
    Bxmin = min(mX)
    Bxmax = max(mX)
    Bymin = min(mY)
    Bymax = max(mY)
    Bxmin, Bxmax, Bymin, Bymax = correct_box(Bxmin, Bxmax, Bymin, Bymax)
    Barea = abs(Bxmax - Bxmin) * abs(Bymax - Bymin) * im_shape[0] * im_shape[1]

    # store info for this mouse
    animal_dict = {'X': X.tolist(),
                   'Y': Y.tolist(),
                   'bbox': np.array([Bxmin, Bxmax, Bymin, Bymax]).tolist(),
                   'med': np.array([mY, mX]).tolist(),
                   'mu': np.array([muY, muX]).tolist(),
                   'std': np.array([stdY, stdX]).tolist(),
                   'area': Barea.tolist()
                   }
    return animal_dict


def correct_box(xmin, xmax, ymin, ymax, stretch_const=0.04, stretch_factor=0.30, useConstant=True):
    # Code to modify bounding boxes to be a bit larger than the keypoints.

    if useConstant:
        stretch_constx = stretch_const
        stretch_consty = stretch_const
    else:
        stretch_constx = (xmax - xmin) * stretch_factor  # of the width
        stretch_consty = (ymax - ymin) * stretch_factor

    # Calculate the amount to stretch the x by.
    x_stretch = np.minimum(xmin, abs(1 - xmax))
    x_stretch = np.minimum(x_stretch, stretch_constx)

    # Calculate the amount to stretch the y by.
    y_stretch = np.minimum(ymin, abs(1 - ymax))
    y_stretch = np.minimum(y_stretch, stretch_consty)

    # Adjust the bounding box accordingly.
    xmin -= x_stretch
    xmax += x_stretch
    ymin -= y_stretch
    ymax += y_stretch
    return xmin,xmax,ymin,ymax


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

