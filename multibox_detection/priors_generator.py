
"""
generate priors for the bbs
You can hand defined them or use prior.py utilties to use your training dataset to cluster the aspect ratios 
of the bounding boxes
"""

#clustering for generation of priors
import os
import sys
import argparse

from priors import *
import pickle

def _parse_function(example_proto):
	features = {
		'image/id' : tf.io.FixedLenFeature([], tf.string),
		'image/encoded'  : tf.io.FixedLenFeature([], tf.string),
		'image/height' : tf.io.FixedLenFeature([], tf.int64),
		'image/width' : tf.io.FixedLenFeature([], tf.int64),
		'image/object/bbox/xmin' : tf.io.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymin' : tf.io.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/xmax' : tf.io.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/ymax' : tf.io.VarLenFeature(dtype=tf.float32),
		'image/object/bbox/count' : tf.io.FixedLenFeature([], tf.int64)
	}
	return tf.io.parse_single_example(example_proto, features)

def convert(tf_r):
	raw_dataset = tf.data.TFRecordDataset(tf_r)
	parsed_dataset = raw_dataset.map(_parse_function)
	imgs = []
	img = {'object': {'bbox': {}}}
	for parsed_record in parsed_dataset.take(len(list(tf.python_io.tf_record_iterator(tf_r[0])))):
		for k in list(parsed_record.keys()):
			list_ver = k.split('/')
			if len(list_ver) == 4:
				img['object']['bbox'][list_ver[-1]] = parsed_record[k]
			if len(list_ver) == 2:
				img[list_ver[-1]] = parsed_record[k]
		imgs.append(img)
	return imgs	

def parse_args():
	parser = argparse.ArgumentParser(description='Generate aspect ratios and priors from training data or hand-defined aspect ratios.')
	parser.add_argument('--dataset', dest='dataset',
						help='pkl or binary tfrecords file that contains the training data to generate priors', type=str,
						required=False)
	parser.add_argument('--aspect_ratios', dest='aspect_ratios',
						help='[list of aspect ratios used to hand generate the priors]', type=str, nargs='+', required=False)

	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	# Generate the apspect ratios from the training data
	if args.aspect_ratios == None:
		if not args.dataset.endswith(".pkl"):
			dataset = convert([args.dataset])
		else:
			with open(args.dataset,'rb') as fp:
				dataset = pickle.load(fp, encoding='latin1')
		aspect_ratios = generate_aspect_ratios(dataset, num_aspect_ratios=11, visualize=False, warp_bboxes=True)
		p = generate_priors(aspect_ratios, min_scale=0.1, max_scale=0.95, restrict_to_image_bounds=True)
		with open(args.dataset.split('/')[-1]+'_priors.pkl', 'wb') as f:
			pickle.dump(p, f)
	# aspect_ratios are hand-defined
	else:
		aspect_ratios = [float(aspect_ratio) for aspect_ratio in args.aspect_ratios]	
		p = generate_priors(aspect_ratios, min_scale=0.1, max_scale=0.95, restrict_to_image_bounds=True)
		with open('priors_hand_gen.pkl', 'wb') as f:
			pickle.dump(p, f)

if __name__ == "__main__":
	tf.compat.v1.enable_eager_execution()
	main()


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
