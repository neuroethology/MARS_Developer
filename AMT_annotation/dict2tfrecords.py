###############################################################################
############ SCRIPT FOR MAKING TFRECORDS
################################################################################

import cPickle as pickle
import math
import os
import random
from create_tfrecords import create
# TODO: Set the number of parts.
num_parts = 7

# TODO: Set the destination for your tfrecords
tfrec_dest = '/YOUR/TFRECORD/PATH/HERE'
# We also dump the info into the same directory.
save_file = os.path.join(tfrec_dest, 'dict_tfrecords.pkl')

# Make the directory if it doesn't exist.
if not os.path.exists(tfrec_dest):
    os.makedirs(tfrec_dest)

# TODO: Set the location for your images -- this will be prepended on each example, which when combined will form
# the fullpath to the file.
im_path = '/YOUR/IMAGE/PATH/HERE'

# TODO: Set the path to the annotation files.
AMT_file1 = '/PATH/TO/AMT/FILE1.pkl'
AMT_file2 = '/PATH/TO/AMT/FILE2.pkl'
AMT_files = [AMT_file1, AMT_file2]

# Load the annotations from the AMT annotators.
D = []
for AMT_file in AMT_files:
    with open(AMT_file, 'rb') as fp:
        D += pickle.load(fp)

# Let them know the files have been loaded.
load_msg = 'Loaded all files.'
print(load_msg)


# Prepare a list of dicts, with each dict containing the info needed for the next step of preparing the tf records
idg = 1
v_info = []
areas =[]
for i in range(len(D)):
    # bbox/label allows to separate between black or white mouse
    # from the annotation 0 is the black mouse, 1 is the white mouse

    # Get the bounding box coordinates of each mouse.
    B = D[i]['ann_B']['bbox']
    W = D[i]['ann_W']['bbox']

    # Take the median of each point's set of annotated locations as its ground truth location.
    Bp = D[i]['ann_B']['med']
    Wp = D[i]['ann_W']['med']

    # Put all the information into the proper format.
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
                        'x':Bp[1][:num_parts].tolist() + Wp[1][:num_parts].tolist(),
                        'y':Bp[0][:num_parts].tolist() + Wp[0][:num_parts].tolist(),
                        'v':[2]*(len(Bp[0][:num_parts]) + len(Wp[0][:num_parts])),
                        'count': [len(Bp[0][:num_parts]),len(Wp[0][:num_parts])]
                    }}}

    v_info.append(i_frame)
    idg += 1

random.shuffle(v_info)

# Save the dict containing this compiled info.
with open(save_file,'wb') as fp:
    pickle.dump(v_info,fp)



# Actually prepare the tfrecords.
# TODO: These decimals determine your testing, training, and validation splits.
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


# Actually create the tfrecords.
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


