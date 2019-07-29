
"""
generate priors for the bbs
You can hand defined them or use prior.py utilties to use your training dataset to cluster the aspect ratios 
of the bounding boxes
"""

#clustering for generation of priors
import os
import sys

sys.path.append('./')

from priors import *
import cPickle as pickle

with open('../tf_dataset_detection/top_separate_allset/dict15k_tfrecords_black.pkl','rb') as fp:
    dataset = pickle.load(fp)


aspect_ratios = generate_aspect_ratios(dataset, num_aspect_ratios=11, visualize=False, warp_bboxes=True)
p = generate_priors(aspect_ratios, min_scale=0.1, max_scale=0.95, restrict_to_image_bounds=True)

# save generated priors
with open('../tf_dataset_detection/top_separate_allset/priors_black.pkl', 'w') as f:
    pickle.dump(p, f)

#visualize priors
# visualize_priors(p, num_priors_per_cell=5, image_height=570, image_width=1024)


##hand generation of priors
# import os, sys
#
# import cPickle as pickle
# import priors
#
# aspect_ratios = [1, 2, 3, 1./2, 1./3]
# p = priors.generate_priors(aspect_ratios, min_scale=0.1, max_scale=0.95, restrict_to_image_bounds=True)
# with open('priors_hand.pkl', 'w') as f:
#     pickle.dump(p, f)

# Clusters:
# Aspect ratio 0.1759, membership count 7180
# Aspect ratio 0.3326, membership count 5084
# Aspect ratio 0.5167, membership count 3956
# Aspect ratio 0.7161, membership count 2674
# Aspect ratio 0.9625, membership count 1106