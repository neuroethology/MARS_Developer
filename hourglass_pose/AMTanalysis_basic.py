###############################################################################
############ script to calculate annotator error.
################################################################################

import sys
import cPickle as pickle
import math
import numpy as np
import os
import json

# TODO: Change this to match your data.
sys.path.append('/YOUR/PATH/HERE')

# TODO: Change this to match your data.
savename = 'YOUR/PATH/TO/SAVE/TO/HERE.json'

# TODO: Change this to match your data.
# Load from the non-miniscope data.
open_file = '/PATH/TO/NONMINISCOPE/ANNOTATIONS/AMT10K_top_csv.pkl'
with open(open_file,'rb') as fp:
    D = pickle.load(fp)

# TODO: Change this to match your data.
# Load from the miniscope data. 
open_file = '/PATH/TO/MINISCOPE/ANNOTATIONS/AMT5K_top_miniscope_csv.pkl'
with open(open_file,'rb') as fp:
    Dm = pickle.load(fp)

# Append them together.
D = D + Dm
print 'loaded'


def unpack_raw_annotations(annotations_dict):
    """
    Unpacks the raw annotations of a pickled annotation file and produces the X's and Y's, with each annotator in a row.
    """
    raw_X = annotations_dict['Y']
    raw_Y = annotations_dict['X']
    return raw_X, raw_Y


def unpack_gt_annotations(annotation_dict):
    medians = annotation_dict['med']
    gt_X = medians[1,:]
    gt_Y = medians[0,:]
    return gt_X, gt_Y


def calc_pixel_error(gt_X, gt_Y, raw_X, raw_Y, x_scale=1024., y_scale=570.):
    """Calculates the pixel error for a given example."""
    print(raw_X)
    print(gt_X)
    err_x = raw_X - gt_X
    err_y = raw_Y - gt_Y
    print(err_x)
    err_x *= x_scale
    err_y *= y_scale

    # print(err_x)
    # raw_input()
    pixelwise_error = np.sqrt(err_x**2 + err_y**2)
    return pixelwise_error

def annotations_to_error(annotation_dict):
    gt_x, gt_y = unpack_gt_annotations(annotation_dict)
    raw_x, raw_y = unpack_raw_annotations(annotation_dict)
    px_error = calc_pixel_error(gt_x,gt_y, raw_x, raw_y)
    return px_error

# Initialize the lists to store errors.
blk_error = []
white_error = []

# Loop over the size of the annotation set.
for i in range(len(D)):
    # Give an update on how far we are.
    print i

    # Parse out the annotations.
    black_annotations = D[i]['ann_B']
    white_annotations = D[i]['ann_W']

    # Convert annotations to error.
    px_error_black = annotations_to_error(black_annotations)
    px_error_white = annotations_to_error(white_annotations)

    # Add the error to the list.
    blk_error.append(px_error_black.tolist())
    white_error.append(px_error_white.tolist())

# Bind the errors together.
errors = [blk_error,white_error]
# Save them as a json.
with open(savename, 'wb') as fp:
    json.dump(errors, fp)





