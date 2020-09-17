###########################################################################################
#                PARSE AMT CSV and add actions as labels for each frame
############################################################################################
# header
#  'run_id', 'filename_hash', 'frame_number', 'annotation_id', 'annotation_time',
#  'annotation_version', 'hit_id', 'worker_id', 'assignment_id', 'click_id',
#  'annotation_type',   'x',  'y',  'dt',  'frame_to_annotate_id',  'filename'
"""
{'frame_name':
'height':
'width':
'frame_id':
'version':
'ann_label':
'action':
'action_idx':
'ann': {'worker_id':
        'time':
        'B':{'type':
             'deltas':
             'xy':
             'xy_head':
             'xy_body':
             'xy_tail'}
        'W':{'type':
             'deltas':
             'xy':
             'xy_head':
             'xy_body':
             'xy_tail'}}
'ann_B':{'X':
         'Y':
         'bbox':
         'bbox_medoid':
         'med':
         'mu':
         'std':
         'medoid':
         'ellipse_param' }
'ann_W':{'X':
         'Y':
         'bbox':
         'bbox_medoid':
         'med':
         'mu':
         'std':
         'medoid':
         'ellipse_param'}}
"""

import os
import sys
import csv
from PIL import Image
import numpy as np
import cPickle as pickle
import math
import scipy.io as sp
import pdb


# from get_param_ellipse import get_param_ellipse
if not os.path.exists('../tf_dataset_detection/front/'):    os.makedirs('../tf_dataset_detection/front/')

path = '/PATH_TO_FRAMES_IMGS/'
frames10k = sp.loadmat('frames10K_labels.mat')
print('load')   #get actions saved from matlab used to extract the 10k frames and actions
labels = np.array([str(x[0]) for x in frames10k['labels'][0]])
# labels = frames10k['labels']
labels_idx  =  np.array([int(x[0]) for x in frames10k['labels_idx'][0]])
# labels_idx = frames10k['type']
fname  = np.array([str(x[0]) for x in frames10k['saved'][0]])
# fname = frames10k['images']

#parse csv file to extract data
csv_file = 'front_annotations.csv'
f = open(csv_file)
csv_f = csv.reader(f,delimiter=';')
T = list(csv_f)
print('load done')

headers = T[0]
T = np.array(T[1:])
filename = np.array([row[-1] for row in T])
unique_filename= list(set(filename))
N_frames = len(unique_filename)
N_keypoints = 13
D = []

# for each frame in the csv file
errors = []
for f in range(N_frames):
    print(f)
    idx_t=[]; tmp_t=[]
    #assign the name of the frame to the entry and find all entries with the same name of the current frame name
    idx_t = np.where(filename== unique_filename[f])
    tmp_t =  T[idx_t]

    #get action of the frame
    find_frame  = [i for i in range(len(fname)) if fname[i][22:] == tmp_t[0][15][26:]]
    # better to store the frames not found and return them at the end
    if find_frame is None:
        print('frame None')
        errors.append(tmp_t[0][15])
    action = labels[find_frame[0]]
    action_idx = labels_idx[find_frame[0]]

    #basic info of the frame
    im = Image.open(path + tmp_t[0][15])
    im = (np.asarray(im)).astype(float)
    frame_dict = {
        'frame_name': tmp_t[0][15],
        'height': im.shape[0],
        'width':im.shape[1],
        'frame_id':tmp_t[0][14],
        'version': tmp_t[0][0],
        'ann_label':['nose tip','right ear','left ear','neck','right side body','left side body','tail base','middle tail','end tail',
                     'right front paw','left front paw','right rear paw','left rear paw'],
        'ann':[],
        'action': action,
        'action_idx': action_idx
    }

    # extract info for each worker
    workers = tmp_t[:,3]
    unique_workers = list(set(workers))
    N_workers = len(unique_workers)
    XB=[];YB=[];XW=[];YW=[]
    for w in range(N_workers):
        idx_w=[]; tmp_w=[]
        idx_w = np.where(workers==unique_workers[w])
        tmp_w = tmp_t[idx_w]

        ann = {
            'worker_id':tmp_w[0][7],
            'time': tmp_w[0][4],
            'B':{'type': list([t[10] for t in tmp_w[:13]]),
                 'deltas': np.array([t[13] for t in tmp_w[:13]]).astype(float),
                 'xy': np.array([t[11:13] for t in tmp_w[:13]]).astype(float),
                 'xy_head': np.array([t[11:13] for t in tmp_w[:13]]).astype(float)[:3],
                 'xy_body': np.array([t[11:13] for t in tmp_w[:13]]).astype(float)[3:7],
                 'xy_tail': np.array([t[11:13] for t in tmp_w[:13]]).astype(float)[7:9],
                 'xy_paws': np.array([t[11:13] for t in tmp_w[:13]]).astype(float)[9:]},
            'W':{'type': list([t[10] for t in tmp_w[13:]]),
                 'deltas': np.array([t[13] for t in tmp_w[13:]]).astype(float),
                 'xy': np.array([t[11:13] for t in tmp_w[13:]]).astype(float),
                 'xy_head': np.array([t[11:13] for t in tmp_w[13:]]).astype(float)[:3],
                 'xy_body': np.array([t[11:13] for t in tmp_w[13:]]).astype(float)[3:7],
                 'xy_tail': np.array([t[11:13] for t in tmp_w[13:]]).astype(float)[7:9],
                 'xy_paws': np.array([t[11:13] for t in tmp_w[13:]]).astype(float)[9:]}}
        frame_dict['ann'].append(ann)

        if len(np.array([t[11] for t in tmp_w[:13]]).astype(float))==13:
            XB.append(np.array([t[11] for t in tmp_w[:13]]).astype(float))
        if len(np.array([t[12] for t in tmp_w[:13]]).astype(float))==13:
            YB.append(np.array([t[12] for t in tmp_w[:13]]).astype(float))
        if len(np.array([t[11] for t in tmp_w[13:]]).astype(float))==13:
            XW.append(np.array([t[11] for t in tmp_w[13:]]).astype(float))
        if len(np.array([t[12] for t in tmp_w[13:]]).astype(float))==13:
            YW.append(np.array([t[12] for t in tmp_w[13:]]).astype(float))

    #some statistics
    XB = np.array(XB);    YB = np.array(YB)
    XW = np.array(XW);    YW = np.array(YW)

    mXB = np.median(XB, axis=0);    mYB = np.median(YB, axis=0)
    mXW = np.median(XW, axis=0);    mYW = np.median(YW, axis=0)

    muXB = np.mean(XB,axis=0);     muYB = np.mean(YB, axis=0)
    muXW = np.mean(XW, axis=0);    muYW = np.mean(YW, axis=0)

    stdXB = np.std(YB, axis=0);  stdYB = np.std(YB, axis=0)
    stdXW = np.std(XW, axis=0);  stdYW = np.std(YW, axis=0)

    #bboxes
    ind = np.hstack([np.arange(7),np.arange(9,13)])
    Bxmin = min(mYB[ind]);   Bxmax = max(mYB[ind])
    Bymin = min(mXB[ind]);   Bymax = max(mXB[ind])
    Wxmin = min(mYW[ind]);   Wxmax = max(mYW[ind])
    Wymin = min(mXW[ind]);   Wymax = max(mXW[ind])

    if Bxmin - .06>0.: Bxmin -= .06
    if Bxmax + .06<1.: Bxmax += .06
    if Bymin - .06>0.: Bymin -= .06
    if Bymax + .06<1.: Bymax += .06
    if Wxmin - .06>0.: Wxmin -= .06
    if Wxmax + .06<1.: Wxmax += .06
    if Wymin - .06>0.: Wymin -= .06
    if Wymax + .06<1.: Wymax += .06

    # area bboxes
    Barea = abs(Bxmax-Bxmin)*abs(Bymax-Bymin)*im.shape[0]*im.shape[1]
    Warea = abs(Wxmax-Wxmin)*abs(Wymax-Wymin)*im.shape[0]*im.shape[1]

    # store info for Black and White mouse
    frame_dict['ann_B']={'X':XB,
                         'Y':YB,
                         'bbox': np.array([Bxmin, Bxmax, Bymin, Bymax]),
                         'med':np.array([mXB,mYB]),
                         'mu':np.array([muXB,muYB]),
                         'std':np.array([stdXB,stdYB]),
                         'area':Barea,
                         }
    frame_dict['ann_W'] = {'X':XW,
                           'Y':YW,
                           'bbox': np.array([Wxmin, Wxmax, Wymin, Wymax]),
                           'med': np.array([mXW, mYW]),
                           'mu': np.array([muXW, muYW]),
                           'std': np.array([stdXW,stdYW]),
                           'area':Warea,
                           }

    D.append(frame_dict)

# save info into pickle file
with open('../tf_dataset_detection/front/AMT10K_csv.pkl','w') as fp:
    pickle.dump(D,fp)
print('saved')

if errors:
    print(errors)
