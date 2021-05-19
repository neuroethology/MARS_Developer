import argparse
import pickle
import logging
import pprint
import sys, os, inspect
import yaml
import time
import glob
import json
import math
from io import StringIO
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.util import deprecation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from MARSeval.coco import COCO
from MARSeval.cocoeval import COCOeval

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from pose_annotation_tools import restore_images_from_tfrecord
import multibox_detection.model_detection as model
from multibox_detection.config import parse_config_file
from multibox_detection import eval_inputs as inputs
from tensorboard.backend.event_processing import event_accumulator

deprecation._PRINT_DEPRECATION_WARNINGS = False


def coco_eval(project, detector_names=None):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not detector_names:
        pose_model_list = cfg['detection']
        detector_names = pose_model_list.keys()

    # Parse things for COCO evaluation.
    savedEvals = {n: {} for n in detector_names}
    for model in detector_names:

        infile = os.path.join(project, 'detection', model + '_evaluation', 'performance_detection.json')
        with open(infile) as jsonfile:
            cocodata = json.load(jsonfile)
        gt_keypoints = cocodata['gt_bbox']
        pred_keypoints = cocodata['pred_bbox']

        gt_coco = COCO()
        gt_coco.dataset = gt_keypoints
        gt_coco.createIndex()
        pred_coco = gt_coco.loadRes(pred_keypoints)

        # Actually perform the evaluation.
        cocoEval = COCOeval(gt_coco, pred_coco, iouType='bbox')

        cocoEval.params.useCats = 0
        cocoEval.evaluate()
        cocoEval.accumulate()
        savedEvals[model] = cocoEval

    return savedEvals


def plot_frame(project, frame_num, detector_names=None, markersize=8, figsize=[15, 10]):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not detector_names:
        pose_model_list = cfg['detection']
        detector_names = pose_model_list.keys()

    legend_flag = [False, False]
    for model in detector_names:
        animals_per_image = len(cfg['detection'][model])
        test_images = os.path.join(project, 'annotation_data', 'test_sets', model + '_detection')
        if not os.path.exists(test_images):
            tfrecords = glob.glob(os.path.join(project, 'detection', model + '_tfrecords_detection', 'test*'))
            for record in tfrecords:
                restore_images_from_tfrecord.restore(record, test_images)  # pull all the images so we can look at them

        image = glob.glob(os.path.join(test_images, 'image' + f'{frame_num:07d}' + '*'))
        if not image:
            print("I couldn't fine image " + str(frame_num))
            return

        infile = os.path.join(project, 'detection', model + '_evaluation', 'performance_detection.json')
        with open(infile) as jsonfile:
            cocodata = json.load(jsonfile)
        pred = [i for i in cocodata['pred_bbox'] if
                any([i['image_id'] == frame_num * animals_per_image + 1 + a for a in range(animals_per_image)])]
        gt = [i for i in cocodata['gt_bbox']['annotations'] if
              any([i['image_id'] == frame_num * animals_per_image + 1 + a for a in range(animals_per_image)])]

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

        im = mpimg.imread(image[0])
        plt.figure(figsize=figsize)
        plt.imshow(im, cmap='gray')

        for pt in gt:
            x = 0
            y = 0
            plt.plot(x, y, color=colors[0], label='ground truth' if not legend_flag[0] else None)
            legend_flag[0] = True
        for pt in pred:
            x = 0
            y = 0
            plt.plot(x, y, color=colors[1], label='predicted' if not legend_flag[1] else None)
            legend_flag[1] = True

        plt.legend(prop={'size': 14})
        plt.show()


def evaluation(tfrecords, bbox_priors, summary_dir, checkpoint_path, num_images, cfg):
    graph = tf.Graph()

    # Force all Variables to reside on the CPU.
    with graph.as_default():

        batched_images, batched_bboxes, batched_num_bboxes, batched_areas, batched_image_ids = inputs.input_nodes(
            tfrecords=tfrecords,
            max_num_bboxes=cfg.MAX_NUM_BBOXES,
            num_epochs=1,
            batch_size=cfg.BATCH_SIZE,
            num_threads=cfg.NUM_INPUT_THREADS,
            capacity=cfg.QUEUE_CAPACITY,
            shuffle_batch=False,
            cfg=cfg
        )

        batch_norm_params = {
            # Decay for the batch_norm moving averages.
            'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            'variables_collections': [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
            'is_training': False
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            biases_regularizer=slim.l2_regularizer(0.00004)):

            locations, confidences, inception_vars = model.build(
                inputs=batched_images,
                num_bboxes_per_cell=cfg.NUM_BBOXES_PER_CELL,
                reuse=False,
                scope=''
            )

        ema = tf.train.ExponentialMovingAverage(
            decay=cfg.MOVING_AVERAGE_DECAY
        )
        shadow_vars = {
            ema.average_name(var): var
            for var in slim.get_model_variables()
        }

        # Restore the parameters
        saver = tf.train.Saver(shadow_vars, reshape=True)

        fetches = [locations,
                   confidences,
                   batched_bboxes,
                   batched_num_bboxes,
                   batched_areas,
                   batched_image_ids]

        # Create a coordinator that will control the different threads.
        coord = tf.train.Coordinator()

        sess_config = tf.ConfigProto(
            log_device_placement=False,
            # device_filters = device_filters,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
            )
        )
        sess = tf.Session(graph=graph, config=sess_config)

        with sess.as_default():

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            dataset_image_ids = set()
            gt_annotations = []
            pred_annotations = []  # {imageID,x1,y1,w,h,score,class}
            gt_annotation_id = 1

            try:
                if tf.io.gfile.isdir(checkpoint_path):
                    print(checkpoint_path)
                    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
                    print(''.join(['is now:', checkpoint_path]))

                if checkpoint_path is None:
                    print("ERROR: No checkpoint file found.")
                    return

                # Restores from checkpoint
                saver.restore(sess, checkpoint_path)

                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
                print("Found model for global step: {:d}".format(global_step))

                step = 0
                print_str = ', '.join(['Step: {:d}', 'Time/image (ms): {:.1f}'])

                while not coord.should_stop():
                    t = time.time()
                    outputs = sess.run(fetches)
                    dt = time.time() - t

                    for b in range(cfg.BATCH_SIZE):
                        locs = outputs[0][b]
                        predicted_confs = outputs[1][b]
                        gt_bboxes = outputs[2][b]
                        gt_num_bboxes = outputs[3][b]
                        gt_areas = outputs[4][b]
                        img_id = int(outputs[5][b])
                        # everything in this file up to here is more or less identical to evaluate_pose---

                        predicted_bboxes = locs + bbox_priors
                        predicted_bboxes = np.clip(predicted_bboxes, 0., 1.)

                        # Scale the predictions and ground truth boxes
                        # GVH: Should check to see if we are preserving aspect ratio or not...
                        im_scale = np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE])
                        predicted_bboxes = predicted_bboxes * im_scale
                        gt_bboxes = gt_bboxes * im_scale

                        # Sort the predictions based on confidences
                        sorted_idxs = np.argsort(predicted_confs.ravel())[::-1]
                        sorted_bboxes = predicted_bboxes[sorted_idxs]
                        sorted_confs = predicted_confs[sorted_idxs]

                        # Store the results
                        for k in range(100):
                            x1, y1, x2, y2 = sorted_bboxes[k]
                            score = sorted_confs[k]
                            pred_annotations.append([
                                img_id,
                                x1, y1, x2 - x1, y2 - y1,
                                score[0],
                                1
                            ])

                        for k in range(gt_num_bboxes):
                            x1, y1, x2, y2 = gt_bboxes[k]
                            w = x2 - x1
                            h = y2 - y1
                            gt_annotations.append({
                                "id": gt_annotation_id,
                                "image_id": img_id,
                                "category_id": 1,
                                "area": gt_areas[k],
                                "bbox": [x1, y1, w, h],
                                "iscrowd": 0,
                            })
                            gt_annotation_id += 1

                        dataset_image_ids.add(img_id)

                    print(print_str.format(step, (dt / cfg.BATCH_SIZE) * 1000))
                    step += 1

                    if num_images > 0 and step == num_images:
                        break

            except tf.errors.OutOfRangeError as e:
                pass

            # When done, ask the threads to stop. Requesting stop twice is fine.
            coord.request_stop()
            # Wait for them to stop before preceding.
            coord.join(threads)

            gt_dataset = {
                'annotations': gt_annotations,
                'images': [{'id': img_id} for img_id in dataset_image_ids],
                'categories': [{'id': 1}]
            }

            cocodata = {'gt_bbox': gt_dataset,
                        'pred_bbox': pred_annotations}
            if os.path.exists(os.path.join(summary_dir, 'performance_detection.json')):
                os.remove(os.path.join(summary_dir, 'performance_detection.json'))
            with open(os.path.join(summary_dir, 'performance_detection.json'), 'w') as jsonfile:
                json.dump(cocodata, jsonfile)


def run_test(project, detector_names=None, num_images=0):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not detector_names:
        pose_model_list = config['detection']
        detector_names = pose_model_list.keys()

    # Set the logging level.
    tf.logging.set_verbosity(tf.logging.DEBUG)

    performance = {n: None for n in detector_names}
    for detector in detector_names:
        checkpoint_path = os.path.join(project, 'detection', detector + '_model')
        if not os.path.isdir(checkpoint_path):
            os.mkdir(os.path.join(project, 'detection', detector + '_model'))
            checkpoint_path = os.path.join(project, 'detection', detector + '_log')
            if not os.path.isdir(checkpoint_path):
                sys.exit("Couldn't find any training output for " + detector + ", you should run train first.")
            else:
                print("Couldn't find a saved model for " + detector + ", using latest training checkpoint instead.")

        tf_dir = os.path.join(project, 'detection', detector + '_tfrecords_detection')
        tfrecords = glob.glob(os.path.join(tf_dir, 'test_dataset-*'))

        priors_fid = os.path.join(project, 'detection', 'priors_' + detector + '.pkl')
        with open(priors_fid, 'rb') as f:
            bbox_priors = pickle.load(f, encoding='latin1')
        bbox_priors = np.array(bbox_priors).astype(np.float32)

        summary_dir = os.path.join(project, 'detection', detector + '_evaluation')
        if not os.path.isdir(summary_dir):
            os.mkdir(summary_dir)

        cfg = parse_config_file(os.path.join(project, 'detection', 'config_test.yaml'))

        evaluation(
            tfrecords=tfrecords,
            bbox_priors=bbox_priors,
            summary_dir=summary_dir,
            checkpoint_path=checkpoint_path,
            num_images=num_images,
            cfg=cfg
        )

        performance[detector] = coco_eval(project, detector_names=detector_names)

    return performance


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate detection model on a tfrecord file')

    parser.add_argument('--project_path', dest='project_path',
                        help='full path to MARS project', type=str, required=True)

    parser.add_argument('--model', dest='model',
                        help='Name of model to evaluate (defaults to all.)',
                        required=False, type=str, default=None)

    parser.add_argument('--num_images', dest='num_images',
                        help='Maximum number of images in test set to analyze. Set to 0 to run on all images.',
                        required=False, type=int, default=0)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    run_test(
        project=args.project,
        detector_names=args.model,
        num_images=args.num_images,
        # show_heatmaps=args.show_heatmaps,
        # show_layer_heatmaps=args.show_layer_heatmaps
    )
