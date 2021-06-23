import argparse
import pickle
import sys, os, inspect
import yaml
import time
import glob
import json
import re
import math
import shutil
from io import StringIO
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.util import deprecation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import matplotlib.cm as cm
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


def coco_eval(project, detector_names=None, reselect_model=False, rerun_test=False):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not detector_names:
        detector_list = cfg['detection']
        detector_names = detector_list.keys()

    # Parse things for COCO evaluation.
    savedEvals = {n: {} for n in detector_names}
    for model in detector_names:

        model_pth = os.path.join(project, 'detection', model + '_model')
        if not os.path.exists(model_pth) or len(os.listdir(model_pth)) == 0 or reselect_model:
            save_best_checkpoint(project, detector_names=model)
        infile = os.path.join(project, 'detection', model + '_evaluation', 'performance_detection.json')
        if not os.path.exists(infile) or rerun_test or reselect_model:
            run_test(project, detector_names=model)

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


def plot_frame(project, frame_num, detector_names=None, markersize=8, figsize=[15, 10], confidence_thr=0.75):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not detector_names:
        detector_list = cfg['detection']
        detector_names = detector_list.keys()

    for model in detector_names:
        if not os.path.exists(os.path.join(project,'detection',model + '_evaluation')):
            print('Please first call evaluate_detection.run_test for model ' + model)
            continue
        print('Sample frame for ' + model + ' detector:')
        legend_flag = [False, False, False]
        test_images = os.path.join(project, 'annotation_data', 'test_sets', model + '_detection')
        if not os.path.exists(test_images):
            tfrecords = glob.glob(os.path.join(project, 'detection', model + '_tfrecords_detection', 'test*'))
            restore_images_from_tfrecord.restore(tfrecords, test_images)  # pull all the images so we can look at them

        image = glob.glob(os.path.join(test_images, 'test' + f'{frame_num:07d}' + '*'))
        if not image:
            print("I couldn't fine image " + str(frame_num))
            return
        matched_id = int(re.search('(?<=image)\d*', image[0]).group(0))

        infile = os.path.join(project, 'detection', model + '_evaluation', 'performance_detection.json')
        with open(infile) as jsonfile:
            cocodata = json.load(jsonfile)
        pred = [i for i in cocodata['pred_bbox'] if i['image_id'] == matched_id]
        gt = [i for i in cocodata['gt_bbox']['annotations'] if i['image_id'] == matched_id]

        colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

        im = mpimg.imread(image[0])
        dims = np.shape(im)
        plt.figure(figsize=figsize)
        plt.imshow(im, cmap='gray')
        for pt in gt:
            x = np.array([pt['bbox'][0], pt['bbox'][0] + pt['bbox'][2]])/299. * dims[1]
            y = np.array([pt['bbox'][1], pt['bbox'][1] + pt['bbox'][3]])/299. * dims[0]
            plt.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], color=colors[0],
                     label='ground truth' if not legend_flag[0] else None)
            legend_flag[0] = True
        for rank, pt in enumerate(pred):
            x = np.array([pt['bbox'][0], pt['bbox'][0] + pt['bbox'][2]]) / 299. * dims[1]
            y = np.array([pt['bbox'][1], pt['bbox'][1] + pt['bbox'][3]]) / 299. * dims[0]
            if rank == 0:
                plt.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], color=colors[1],
                         label='predicted' if not legend_flag[1] else None)
                legend_flag[1] = True
            elif pt['score'] >= confidence_thr:
                plt.plot([x[0], x[1], x[1], x[0], x[0]], [y[0], y[0], y[1], y[1], y[0]], color=colors[2],
                         label='high confidence' if not legend_flag[2] else None)
                legend_flag[2] = True

        plt.legend(prop={'size': 14})
        plt.show()


def pr_curve(project, detector_names=None):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if not detector_names:
        detector_list = cfg['detection']
        detector_names = detector_list.keys()

    performance = coco_eval(project, detector_names=detector_names)
    for model in detector_names:

        rs_mat = performance[model].params.recThrs  # [0:.01:1] R=101 recall thresholds for evaluation
        ps_mat = performance[model].eval['precision']
        iou_mat = performance[model].params.iouThrs

        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=len(iou_mat))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

        show = [0.5, 0.75, 0.85, 0.9, 0.95]
        fig, ax = plt.subplots(1, figsize=[8, 6])
        for i in range(len(iou_mat)):
            if round(iou_mat[i]*1000.)/1000. in show:
                colorVal = scalarMap.to_rgba(i)
                ax.plot(rs_mat, ps_mat[i, :, :, 0, 1], c=colorVal, ls='-', lw=2, label='IoU >= %s' % np.round(iou_mat[i], 2))
        plt.grid()
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('PR curves for ' + model + ' detector')
        plt.legend(loc='best')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.savefig(os.path.join(project, 'detection', model + '_evaluation',  'PR_curves.pdf'))
        plt.show()


# this section is the same as in evaluate_pose -------------------------------------------------------------------------
def smooth(xran, tran, decay=0.99, burnIn=0):
    # burnIn lowers the smoothing time constant at the start of smoothing. Set to 0 to disable.

    sm_x = np.zeros_like(xran)
    sm_x[0] = xran[0]
    for i, x in enumerate(xran[1:]):
        if burnIn:
            d = decay * (1 - 0.5 * math.exp(-tran[i] / burnIn))
        else:
            d = decay
        delta = sm_x[i]
        for step in range(int(tran[i + 1] - tran[i])):
            delta = float(x) * (1 - d) + delta * d
        sm_x[i + 1] = delta

    return sm_x


def find_best_checkpoint(project, model, decay=0.99975, burnIn=1000):
    event_path = os.path.join(project, 'detection', model + '_log')

    ckptfiles = glob.glob(os.path.join(event_path, 'model.ckpt-*.index'))
    ckptfiles = ['.'.join(f.split('.')[:-1]) for f in ckptfiles]
    ckpt_steps = np.array([int(c) for text in ckptfiles for c in re.compile(r'\d+').findall(os.path.basename(text))])

    event_path = os.path.join(project, 'detection', model + '_log')
    eventfiles = glob.glob(os.path.join(event_path, 'events.out.tfevents.*'))
    eventfiles.sort(key=lambda text: [int(c) for c in re.compile(r'\d+').findall(text)])

    onlyfiles = [f for f in os.listdir(event_path) if os.path.isfile(os.path.join(event_path, f))]
    onlyckpts = ['.'.join(f.split('.')[:-1]) for f in onlyfiles if 'index' in f]
    onlyckpts.sort(key=lambda text: [int(c) for c in re.compile(r'\d+').findall(text)])

    sz = {event_accumulator.IMAGES: 0}
    steps = np.empty(shape=(0))
    vals = np.empty(shape=(0))
    for f in eventfiles:
        ea = event_accumulator.EventAccumulator(f, size_guidance=sz)
        ea.Reload()

        if not ea.Tags()['scalars'] or not 'validation_loss' in ea.Tags()['scalars']: # skip empty event files
            continue

        tr_steps = np.array([step.step for step in ea.Scalars('validation_loss')]).T
        tr_vals = np.array([step.value for step in ea.Scalars('validation_loss')]).T

        steps = np.concatenate((steps, tr_steps))
        vals = np.concatenate((vals, tr_vals))
    inds = np.argsort(steps)
    steps = steps[inds]
    vals = vals[inds]

    sm_vals = smooth(vals, steps, decay, burnIn)
    step_inds = [np.where(steps == k) for k in ckpt_steps]
    step_inds = [x[0][0] for x in step_inds if len(x[0])]
    min_step = steps[np.where(sm_vals == np.amin(sm_vals[step_inds]))[0][0]]

    params = {'burnIn': burnIn, 'decay': decay}
    return steps, vals, ckpt_steps, min_step, params


def save_best_checkpoint(project, detector_names=None):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not detector_names:
        detector_list = config['detection']
        detector_names = detector_list.keys()
    if isinstance(detector_names, str):
        detector_names = [detector_names]

    for model in detector_names:
        log_path = os.path.join(project, 'detection', model + '_log')
        model_path = os.path.join(project, 'detection', model + '_model')
        _, _, _, min_step, _ = find_best_checkpoint(project, model)
        model_str = str(int(min_step))

        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path)

        ckpt_files = glob.glob(os.path.join(log_path, '*' + model_str + '*'))
        for f in ckpt_files:
            shutil.copyfile(f, f.replace('_log', '_model'))

        with open(os.path.join(model_path,'checkpoint'), 'w') as f:
            f.write('model_checkpoint_path: "' + os.path.join(model_path,'model.ckpt-' + model_str) + '"')
        print('Saved best-performing checkpoint for model "' + model + '."')


def plot_training_progress(project, detector_names=None, figsize=(14, 6), logTime=False, omitFirst=0):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not detector_names:
        detector_names = config['detection']
        detector_names = detector_names.keys()

    fix, ax = plt.subplots(len(detector_names), 1, figsize=figsize, squeeze=False)
    for i, model in enumerate(detector_names):

        steps, vals, ckpt_steps, min_step, params = find_best_checkpoint(project, model)
        sm_vals = smooth(vals, steps, params['decay'], params['burnIn'])

        drop = np.argwhere(steps < omitFirst)
        steps = np.delete(steps, drop)
        vals = np.delete(vals, drop)
        sm_vals = np.delete(sm_vals, drop)

        dropckpt = np.argwhere(ckpt_steps < omitFirst)
        ckpt_steps = np.delete(ckpt_steps, dropckpt)
        step_inds = []
        for k in ckpt_steps:
            if len(np.where(steps >= k)[0]):
                step_inds.append(np.where(steps >=k)[0][0])
            else:
                step_inds.append(len(steps) - 1)
        ckpt_vals = [sm_vals[x] for x in step_inds]

        ax[i, 0].plot(steps, vals, color='skyblue', label='raw')
        ax[i, 0].plot(steps, sm_vals, color='darkblue', label='smoothed')
        ax[i, 0].plot(ckpt_steps, ckpt_vals, 'ro', label='saved checkpoints')
        ax[i, 0].plot(min_step, sm_vals[np.where(steps == min_step)], 'o', label='best model', markersize=16,
                      markeredgewidth=4, markeredgecolor='orange', markerfacecolor='None')

        ax[i, 0].set_xlabel('Training step')
        ax[i, 0].set_ylabel('Validation loss')
        if logTime:
            ax[i, 0].set_xscale('log')
        ax[i, 0].set_yscale('log')
        ax[i, 0].set_title('Training progress for detector "' + model + '"')

    ax[0, 0].legend()
    plt.show()
# ----------------------------------------------------------------------------------------------------------------------


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
                img_id = 0
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
                        # img_id = int(outputs[5][b])
                        # everything in this file up to here is more or less identical to evaluate_pose---

                        predicted_bboxes = locs + bbox_priors
                        predicted_bboxes = np.clip(predicted_bboxes, 0., 1.)

                        # Scale the predictions and ground truth boxes
                        # GVH: Should check to see if we are preserving aspect ratio or not...
                        # im_scale = np.array([cfg.INPUT_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE])
                        # predicted_bboxes = predicted_bboxes * im_scale
                        # gt_bboxes = gt_bboxes * im_scale

                        # Sort the predictions based on confidences
                        sorted_idxs = np.argsort(predicted_confs.ravel())[::-1]
                        sorted_bboxes = predicted_bboxes[sorted_idxs]
                        sorted_confs = predicted_confs[sorted_idxs]

                        # Store the results
                        for k in range(100):
                            sorted_bboxes[k].tolist
                            x1, y1, x2, y2 = sorted_bboxes[k].tolist()
                            score = sorted_confs[k].tolist()
                            pred_annotations.append({
                                "image_id": img_id,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": score[0],
                                "category_id": 1
                            })

                        for k in range(gt_num_bboxes):
                            x1, y1, x2, y2 = gt_bboxes[k].tolist()
                            w = x2 - x1
                            h = y2 - y1
                            gt_annotations.append({
                                "id": gt_annotation_id,
                                "image_id": img_id,
                                "category_id": 1,
                                "area": gt_areas[k].tolist(),
                                "bbox": [x1, y1, w, h],
                                "iscrowd": 0,
                            })
                            gt_annotation_id += 1

                        dataset_image_ids.add(img_id)
                        img_id += 1

                    # print(print_str.format(step, (dt / cfg.BATCH_SIZE) * 1000))
                    step += 1

                    if 0 < num_images == step:
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
        detector_list = config['detection']
        detector_names = detector_list.keys()
    elif isinstance(detector_names,str):
        detector_names = [detector_names]

    # Set the logging level.
    tf.logging.set_verbosity(tf.logging.DEBUG)

    performance = {n: None for n in detector_names}
    for detector in detector_names:
        print('detecting using ' + detector)
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

        performance[detector] = coco_eval(project, detector_names=[detector])

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
