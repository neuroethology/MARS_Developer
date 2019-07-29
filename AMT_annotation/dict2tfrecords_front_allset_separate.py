###############################################################################
############ FORMAT TO TF_RECORDS for single mouse
################################################################################

import sys, os
import cPickle as pickle
import math
import random
sys.path.append('./')
from create_tfrecords import create

#load extracted info from annotations
with open('PATH_TO/AMT15K_csv_front.pkl','rb') as fp:
    D = pickle.load(fp)
print 'load'

im_path = '/PATH_TO_FRAMES/'

"""
{
  "filename" : "the full path to the image",
  "id" : "an identifier for this image",
  'width':
  'height':
  "class" : {
    "label" : "integer in the range [0, num_classes)",
    "text" : "a human readable string for this class"
  },
  "object" : {
    "bbox" : {
      "xmin" : "an array of float values",
      "xmax" : "an array of float values",
      "ymin" : "an array of float values",
      "ymax" : "an array of float values",
      "label" : "an array of integer values, in the range [0, num_classes)",
      "count" : "an integer, the number of bounding boxes"
    }
  }
}
"""
#prepare a dict with the info needed for the next step of preparing the tf records
#black mouse
idg = 1
v_infob = []
for i in range(len(D)):
    print i
    # image name and id
    # bbox/label allows to separate between black or white mouse
    # from the annotation 0 is the black mouse, 1 is the white mouse

    B = D[i]['ann_B']['bbox']
    W = D[i]['ann_W']['bbox']
    i_frame = {'filename': im_path + D[i]['frame_name'],
               "class": {
                   "label": 0,
                   "text": '',
               },
               'id':  format(idg, '06d'),
               'width': D[i]['width'],
               'height': D[i]['height'],
               'object': {'area':[ D[i]['ann_B']['area']],
                   'bbox': {
                       'xmin': [B[0]],
                       'xmax': [B[1]],
                       'ymin': [B[2]],
                       'ymax': [B[3]],
                       'label':[0],
                       'count': 1}}}
    v_infob.append(i_frame)
    idg += 1

#white mouse
idg = 1
v_infow = []
for i in range(len(D)):
    print i
    # image name and id
    # bbox/label allows to separate between black or white mouse
    # from the annotation 0 is the black mouse, 1 is the white mouse

    B = D[i]['ann_B']['bbox']
    W = D[i]['ann_W']['bbox']
    i_frame = {'filename': im_path + D[i]['frame_name'],
               "class": {
                   "label": 0,
                   "text": '',
               },
               'id':  format(idg, '06d'),
               'width': D[i]['width'],
               'height': D[i]['height'],
               'object': {'area':[ D[i]['ann_W']['area']],
                   'bbox': {
                       'xmin': [W[0]],
                       'xmax': [W[1]],
                       'ymin': [W[2]],
                       'ymax': [W[3]],
                       'label':[0],
                       'count': 1}}}
    v_infow.append(i_frame)
    idg += 1


v_info = list(zip(v_infob, v_infow))
random.shuffle(v_info)
v_infob, v_infow = zip(*v_info)

#save the dict
if not os.path.exists('../tf_dataset_detection/front_separate_allset'):
    os.makedirs('../tf_dataset_detection/front_separate_allset')
if not os.path.exists('../tf_dataset_detection/front_separate_allset/black'):
    os.makedirs('../tf_dataset_detection/front_separate_allset/black')
if not os.path.exists('../tf_dataset_detection/front_separate_allset/white'):
    os.makedirs('../tf_dataset_detection/front_separate_allset/white')
with open('../tf_dataset_detection/front_separate_allset/dict15k_tfrecords_black.pkl','wb') as fp:
    pickle.dump(v_infob,fp)
with open('../tf_dataset_detection/front_separate_allset/dict15k_tfrecords_white.pkl','wb') as fp:
    pickle.dump(v_infow,fp)
print ('done ')

v_info[0]

#prepare tf records dataset
# split train val test
n = len(v_infob)
ntrain =int(math.floor(n * .85))
nval = int(round(n * .05))
ntest = int(round(n * .10))
print(n, ntrain, nval, ntest)

train = v_infob[:ntrain]
val = v_infob[ntrain:ntrain + nval]
test = v_infob[ntrain + nval:]

# where to save the files
tfrec_dest = '../tf_dataset_detection/front_separate_allset/black/'
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

n = len(v_infow)
ntrain =int(math.floor(n * .85))
nval = int(round(n * .05))
ntest = int(round(n * .10))
print(n, ntrain, nval, ntest)

train = v_infow[:ntrain]
val = v_infow[ntrain:ntrain + nval]
test = v_infow[ntrain + nval:]

# where to save the files
tfrec_dest = '../tf_dataset_detection/front_separate_allset/white/'
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
