###############################################################################
############ FORMAT TO TF_RECORDS
################################################################################

import sys
import cPickle as pickle
import math
import numpy as np
import pdb
import os
import random

sys.path.append('./')
from create_tfrecords import create

open_file = '../tf_dataset_keypoints/top_allset/dict15k_tfrecords.pkl'
save_file = '../tf_dataset_keypoints/top_allset/dict15k_tfrecords.pkl'
tfrec_dest = '../tf_dataset_keypoints/top_allset/'
if not os.path.exists(tfrec_dest):
    os.makedirs(tfrec_dest)

im_path = '/PATH_TO_FRAMES_IMG/'

#load extracted info from annotations
open_file = 'PATH_TO/AMT10K_csv_top.pkl'
with open(open_file,'rb') as fp:    D = pickle.load(fp)
open_file = 'PATH_TO/AMT5K_csv_top.pkl'
with open(open_file,'rb') as fp:    Dm = pickle.load(fp)
D = D + Dm
print 'load'


#prepare a dict with the info needed for the next step of preparing the tf records
idg = 1
v_info = []
areas =[]
for i in range(len(D)):
    print i
    # image name and id
    # bbox/label allows to separate between black or white mouse
    # from the annotation 0 is the black mouse, 1 is the white mouse

    B = D[i]['ann_B']['bbox']
    W = D[i]['ann_W']['bbox']
    Bp = D[i]['ann_B']['med']
    Wp = D[i]['ann_W']['med']
    i_frame = {'filename': im_path + D[i]['frame_name'],
               'id': format(idg, '06d'),
               "class": {
                   "label": 0,
                   "text": '',
               },
               'width': D[i]['width'],
               'height': D[i]['height'],
               'object': {
                   'id':[0,1],
                   'area':[ D[i]['ann_B']['area'], D[i]['ann_W']['area']],
                   'bbox': {
                       'xmin': [B[0],W[0]],
                       'xmax': [B[1],W[1]],
                       'ymin': [B[2],W[2]],
                       'ymax': [B[3],W[3]],
                       'label': [0,0],
                       'count': 2,
                       'score':[1,1]},
                   'parts':{
                        'x':Bp[1][:7].tolist() + Wp[1][:7].tolist(),
                        'y':Bp[0][:7].tolist() + Wp[0][:7].tolist(),
                        'v':[2]*(len(Bp[0][:7]) + len(Wp[0][:7])),
                        'count': [len(Bp[0][:7]),len(Wp[0][:7])]
                    }}}

    v_info.append(i_frame)
    idg += 1

random.shuffle(v_info)

#save the dict
with open(save_file,'wb') as fp:
    pickle.dump(v_info,fp)
print ('done ')

# v_info[0]

#prepare tf records dataset
# split train val test
n = len(v_info)
ntrain =int(math.floor(n * .85))
nval = int(round(n * .05))
ntest = int(round(n * .10))
print(n, ntrain, nval, ntest)

train = v_info[:ntrain]
val = v_info[ntrain:ntrain + nval]
test = v_info[ntrain + nval:]

for i in range(len(v_info)):
    if not v_info[i]['object']['area'][0]>0 or not v_info[i]['object']['area'][1]>0:
        print i


# create tf records
create(
    dataset=train,
    dataset_name="train_dataset",
    output_directory=tfrec_dest,
    num_shards=10,
    num_threads=5,
    shuffle=False
)

create(
    dataset=val,
    dataset_name="val_dataset",
    output_directory=tfrec_dest,
    num_shards=1,
    num_threads=1,
    shuffle=False
)

create(
    dataset=test,
    dataset_name="test_dataset",
    output_directory=tfrec_dest,
    num_shards=1,
    num_threads=1,
    shuffle=False
)
