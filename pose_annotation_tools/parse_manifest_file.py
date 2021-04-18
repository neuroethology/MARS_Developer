import json
import numpy as np
from scipy.spatial.distance import cdist, euclidean
import copy
import pickle
import os,sys
import yaml


def apply_flip_correction(frame, meds, keypoints, pair):
    if not pair[0] in keypoints or not pair[1] in keypoints:
        raise SystemExit('annot_config error: one or more entries of check_pairs is not a member of keypoints. Please check annot_config.yml')

    i1 = keypoints.index(pair[0])
    i2 = keypoints.index(pair[1])

    for rep in range(3): #repeat 3 times for stability
        for w,worker in enumerate(frame.swapaxes(0,2).swapaxes(1,2)):
            d1 = cdist(worker[[i1,i2],:],[meds[:,i1]])
            d2 = cdist(worker[[i1,i2],:],[meds[:,i2]])
            if (d1[0]>d2[1]) and (d2[1]>d1[0]):
                frame[[i1, i2],:,w] = frame[[i2, i1],:,w]

        # re-compute the medians:
        meds = np.median(np.squeeze(frame.T),axis=0)

    return frame, meds


def parse_manifest_file(project, nWorkers=5, manifest_name='output.manifest', correct_flips=True)
    # read the output manifest
    manifest_fid = os.path.join(project,'annotation_data',manifest_name) + ('.manifest' if not manifest_name.endswith('.manifest') else '')
    data = []
    for line in open(fid,'r'):
        data.append(json.loads(line))

    # read the annotation config file
    config_fid = os.path.join(project,'annotation_data','annot_config.yml')
    with open(config_fid) as f:
        cfg = yaml.load(f)

    keypoints    = cfg['keypoints']
    animal_names = cfg['animal_names'] if cfg['animal_names'] else cfg['species']
    check_pairs   = cfg['check_pairs']

    # next we unpack the raw data from all individual annotators
    nSamp   = len(data)
    nKpts   = len(keypoints)

    # for each image: nKpts keypoints, 2 dimensions, nWorkers workers/keypoint
    rawPts = {n:np.zeros((nSamp,nKpts,2,nWorkers)) for n in animal_names}

    images = ['']*nSamp        # store paths to labeled images, replacing s3 paths with local versions
    hits = [False]*nSamp       # track which images have annotations (hopefully all of them)
    workerCount = [0]*nSamp    # track the number of workers who labeled each image

    sourceStr = os.path.dirname(data[0]['source-ref']) # image path on AWS
    localStr = os.path.join(project,'annotation_data','raw_images') # image path locally

    for f,frame in enumerate(data):
        if ('annotatedResult' in frame.keys()): #check if this frame has at least one set of annotations
            hits[f] = True
            images[f] = frame['source-ref']
            images[f].replace(sourceStr,localStr)

            for w,worker in enumerate(frame['annotatedResult']['annotationsFromAllWorkers']):
                workerCount[f] = workerCount[f] + 1

                # the json of annotations from each worker is stored as a string for security reasons.
                # we'll use eval to convert it into a dict:
                annot = eval(worker['annotationData']['content'])

                # now we can unpack this worker's annotations for each keypoint:
                for pt in annot['annotatedResult']['keypoints']:
                    animal = next((n for n in animal_names if n in pt['label']),cfg['species'])
                    idx = keypoints.index(pt['label'].replace(animal,'').replace(species,'').strip())

                    rawPts[animal][f,idx,0,w] = pt['x']
                    rawPts[animal][f,idx,1,w] = pt['y']


    # now let's get our ground-truth keypoints by taking the median across annotators.
    gtPts  = {n:[] for n in animal_names}
    allPts = copy.deepcopy(rawPts)
    for animal in animal_names:
        #first we'll take medians of the raw keypoints:
        gtPts[animal] = np.zeros((nSamp,nKpts,2))
        for f,frame in enumerate(rawPts[animal]):

            meds = np.median(np.squeeze(frame.T),axis=0)

            if correct_flips: # adjust L/R assignments to try to find better median estimates.
                for pair in check_pairs:
                    frame,meds = apply_flip_correction(frame, meds, keypoints, pair)
                    allPts[animal][f,:,:,:] = frame   # corrected copy of rawPts

            gtPts[animal][f,:,:] = meds # median keypoint locations
