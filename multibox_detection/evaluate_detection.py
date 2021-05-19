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


def coco_eval(project, detector_names=None, view=None, fixedSigma=None):
    # # initialize COCO GT api
    # gt_coco = COCO()
    # gt_coco.dataset = gt_dataset
    # gt_coco.createIndex()
    # # initialize COCO detections api: output creating index and index created
    # pred_coco = gt_coco.loadRes(pred_annotations)
    # # running evaluation
    # cocoEval = COCOeval(gt_coco, pred_coco, iouType='bbox')
    # 
    # cocoEval.params.useCats = 0
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # 
    # old_stdout = sys.stdout
    # sys.stdout = captured_stdout = StringIO()
    # cocoEval.summarize()  # print results
    # sys.stdout = old_stdout
    # 
    # summary_op = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(summary_dir)
    # summary = tf.Summary()
    # summary.ParseFromString(sess.run(summary_op))
    # 
    # with open(summary_dir + 'cocoEval.pkl', 'wb') as fp:
    #     pickle.dump(cocoEval, fp)
    # 
    # for line in captured_stdout.getvalue().split('\n'):
    #     if line != "":
    #         description, score = line.rsplit("=", 1)
    #         description = description.strip()
    #         score = float(score)
    # 
    #         summary.value.add(tag=description, simple_value=score)
    # 
    #         print(f"{description:s}: {score:0.3f}")
    # 
    # summary_writer.add_summary(summary, global_step)
    # summary_writer.flush()
    # summary_writer.close()

    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    parts = cfg['keypoints']

    if not detector_names:
        pose_model_list = cfg['pose']
        detector_names = pose_model_list.keys()

    # Parse things for COCO evaluation.
    savedEvals = {n: {} for n in detector_names}

    for model in detector_names:

        infile = os.path.join(project, 'pose', model + '_evaluation', 'performance_pose.json')
        with open(infile) as jsonfile:
            cocodata = json.load(jsonfile)
        gt_keypoints = cocodata['gt_keypoints']
        pred_keypoints = cocodata['pred_keypoints']

        for partNum in range(len(parts) + 1):

            MARS_gt = copy.deepcopy(gt_keypoints)
            MARS_gt['annotations'] = [d for d in MARS_gt['annotations'] if d['category_id'] == (partNum + 1)]
            MARS_pred = [d for d in pred_keypoints if d['category_id'] == (partNum + 1)]

            gt_coco = COCO()
            gt_coco.dataset = MARS_gt
            gt_coco.createIndex()
            pred_coco = gt_coco.loadRes(MARS_pred)

            # Actually perform the evaluation.
            part = [parts[partNum - 1]] if partNum else parts
            if fixedSigma:
                assert fixedSigma in ['narrow', 'moderate', 'wide', 'ultrawide']
                cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='fixed', useParts=[fixedSigma])
            elif not view:
                # print('warning: camera view not specified, defaulting to evaluation using fixedSigma=narrow')
                cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='fixed', useParts=['narrow'])
            elif view.lower() == 'top':
                cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='MARS_top', useParts=part)
            elif view.lower() == 'front':
                cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='MARS_front', useParts=part)
            else:
                raise ValueError('Something went wrong.')

            cocoEval.evaluate()
            cocoEval.accumulate()
            partstr = part[0] if partNum else 'all'
            savedEvals[model][partstr] = cocoEval

    return savedEvals


def plot_frame(project, frame_num, detector_names=None, markersize=8, figsize=[15, 10]):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not detector_names:
        pose_model_list = cfg['pose']
        detector_names = pose_model_list.keys()

    legend_flag = [False, False]
    for model in detector_names:
        animals_per_image = len(cfg['pose'][model])
        test_images = os.path.join(project, 'annotation_data', 'test_sets', model + '_pose')
        if not os.path.exists(test_images):
            tfrecords = glob.glob(os.path.join(project, 'pose', model + '_tfrecords_pose', 'test*'))
            for record in tfrecords:
                restore_images_from_tfrecord.restore(record, test_images)  # pull all the images so we can look at them

        image = glob.glob(os.path.join(test_images, 'image' + f'{frame_num:07d}' + '*'))
        if not image:
            print("I couldn't fine image " + str(frame_num))
            return

        infile = os.path.join(project, 'pose', model + '_evaluation', 'performance_pose.json')
        with open(infile) as jsonfile:
            cocodata = json.load(jsonfile)
        pred = [i for i in cocodata['pred_keypoints'] if i['category_id'] == 1 and
                any([i['image_id'] == frame_num * animals_per_image + 1 + a for a in range(animals_per_image)])]
        gt = [i for i in cocodata['gt_keypoints']['annotations'] if i['category_id'] == 1 and
              any([i['image_id'] == frame_num * animals_per_image + 1 + a for a in range(animals_per_image)])]

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
        markers = 'vosd*p'

        im = mpimg.imread(image[0])
        plt.figure(figsize=figsize)
        plt.imshow(im, cmap='gray')

        for pt in gt:
            for i, [x, y] in enumerate(zip(pt['keypoints'][::3], pt['keypoints'][1::3])):
                plt.plot(x, y, color=colors[i], marker='o', markeredgecolor='k',
                         markeredgewidth=math.sqrt(markersize) / 4, markersize=markersize, linestyle='None',
                         label='ground truth' if not legend_flag[0] else None)
                legend_flag[0] = True
        for pt in pred:
            for i, [x, y] in enumerate(zip(pt['keypoints'][::3], pt['keypoints'][1::3])):
                plt.plot(x, y, color=colors[i], marker='^', markeredgecolor='w',
                         markeredgewidth=math.sqrt(markersize) / 2, markersize=markersize, linestyle='None',
                         label='predicted' if not legend_flag[1] else None)
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

            pred_annotations = np.array(pred_annotations)
            
            gt_dataset = {
                'annotations': np.array(gt_annotations),
                'images': np.array([{'id': img_id} for img_id in dataset_image_ids]).flatten(),
                'categories': [{'id': 1}]
            }


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
