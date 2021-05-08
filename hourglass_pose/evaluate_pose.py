import argparse
import yaml
import numpy as np
import sys, os, inspect
import json
import math
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim
from tensorflow.python.util import deprecation
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import interpolate
from MARSeval.coco import COCO
from MARSeval.cocoeval import COCOeval
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import copy
import glob
import re
import shutil

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from pose_annotation_tools.evaluation import compute_human_PCK
from pose_annotation_tools import restore_images_from_tfrecord
from hourglass_pose.config import parse_config_file
from hourglass_pose import eval_inputs as inputs
from hourglass_pose import model_pose
from tensorboard.backend.event_processing import event_accumulator


deprecation._PRINT_DEPRECATION_WARNINGS = False


def coco_eval(project, pose_model_names=None, view=None, fixedSigma=None):
    config_fid = os.path.join(project,'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    parts = cfg['keypoints']

    if not pose_model_names:
        pose_model_list = cfg['pose']
        pose_model_names = pose_model_list.keys()

    # Parse things for COCO evaluation.
    savedEvals = {n: {} for n in pose_model_names}

    for model in pose_model_names:

        infile = os.path.join(project, 'pose', model+'_evaluation', 'performance_pose.json')
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


def plot_frame(project, frame_num, pose_model_names=None, markersize=8, figsize=[15, 10]):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not pose_model_names:
        pose_model_list = cfg['pose']
        pose_model_names = pose_model_list.keys()

    legend_flag=[False,False]
    for model in pose_model_names:
        animals_per_image = len(cfg['pose'][model])
        test_images = os.path.join(project, 'annotation_data', 'test_sets', model + '_pose')
        if not os.path.exists(test_images):
            tfrecords = glob.glob(os.path.join(project,'pose', model + '_tfrecords_pose', 'test*'))
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
                any([i['image_id'] == frame_num*animals_per_image + 1 + a for a in range(animals_per_image)])]
        gt = [i for i in cocodata['gt_keypoints']['annotations'] if i['category_id'] == 1 and
              any([i['image_id'] == frame_num*animals_per_image + 1 + a for a in range(animals_per_image)])]

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
        markers = 'vosd*p'

        im = mpimg.imread(image[0])
        plt.figure(figsize=figsize)
        plt.imshow(im, cmap='gray')

        for pt in gt:
            for i, [x, y] in enumerate(zip(pt['keypoints'][::3], pt['keypoints'][1::3])):
                plt.plot(x, y, color=colors[i], marker='o',  markeredgecolor='k',
                         markeredgewidth=math.sqrt(markersize)/4, markersize=markersize, linestyle='None',
                         label='ground truth' if not legend_flag[0] else None)
                legend_flag[0]=True
        for pt in pred:
            for i,[x,y] in enumerate(zip(pt['keypoints'][::3], pt['keypoints'][1::3])):
                plt.plot(x, y, color=colors[i], marker='^', markeredgecolor='w',
                         markeredgewidth=math.sqrt(markersize)/2, markersize=markersize, linestyle='None',
                         label='predicted' if not legend_flag[1] else None)
                legend_flag[1] = True

        plt.legend(prop={'size': 14})
        plt.show()


def compute_oks_histogram(cocoEval, bins=[]):
    oks = []
    partID = list(cocoEval.cocoGt.catToImgs.keys())[0]  # which body part are we looking at?
    for i in cocoEval.params.imgIds:
        oks.append(cocoEval.computeOks(i, partID)[0][0])
    if not bins:
        counts, bins = np.histogram(oks, 20, (0, 1))
    else:
        counts, bins = np.histogram(oks, bins=bins)
    return counts, bins


def compute_model_pck(cocoEval, lims=None, pixels_per_cm=None, pixel_units=False):
    bins = 10000
    pck = []
    partID = list(cocoEval.cocoGt.catToImgs.keys())[0]  # which body part are we looking at?
    for i in cocoEval.params.imgIds:
        pck.append(cocoEval.computePcks(i, partID)[0][0])
    pck = np.array(pck)

    if not lims:
        lims = [0, max(pck)]

    counts, binedges = np.histogram(pck, bins, range=lims)
    counts = counts / len(pck)
    binctrs = (binedges[:-1] + binedges[1:]) / 2
    if not pixel_units:
        binctrs = binctrs / pixels_per_cm

    return counts, binctrs


def plot_model_PCK(project, performance=None, pose_model_names=None, xlim=None, pixel_units=False,
                   combine_animals=False):

    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    ptNames = ['all']  + cfg['keypoints']
    pix_per_cm = cfg['pixels_per_cm']

    if not pix_per_cm:
        pixel_units=True
    elif xlim and not pixel_units:  # assume xlim was provided in cm; PCK computations are always done in pixels.
        xlim = [x * pix_per_cm for x in xlim]

    if not pose_model_names:
        pose_model_list = cfg['pose']
        pose_model_names = list(pose_model_list.keys())

    if not performance:
        performance = coco_eval(project, pose_model_names=pose_model_names)

    nKpts = len(cfg['keypoints'])
    fig, ax = plt.subplots(math.ceil(nKpts / 4), 4, figsize=(15, 4 * math.ceil(nKpts / 4)))
    thr = 0.85
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    # ax = ax.flatten()
    cnum = -1
    for model in pose_model_names:
        animal_names = pose_model_list[model]
        counts_hu, super_counts_hu, binctrs_hu = compute_human_PCK(project, animal_names=animal_names,
                                                                   xlim=xlim, pixel_units=pixel_units)
        if not xlim:
            delta = (binctrs_hu[1] - binctrs_hu[0]) / 2
            xlim_hu = [binctrs_hu[0] - delta, binctrs_hu[-1] + delta]
        else:
            xlim_hu = xlim

        counts_model = {n: [] for n in ptNames}
        for i, usePart in enumerate(ptNames):
            counts_model[usePart], binctrs_model = compute_model_pck(performance[model][usePart],
                                                                     lims=xlim_hu, pixels_per_cm=pix_per_cm,
                                                                     pixel_units=pixel_units)

        if not pixel_units and xlim:
                xlim = [x / pix_per_cm for x in xlim]

        if combine_animals:
            cutoff = 0
            for p, pt in enumerate(ptNames):
                objs = ax[int(p / 4), p % 4].stackplot(binctrs_hu, super_counts_hu[pt]['max'].cumsum(),
                                                       (super_counts_hu[pt]['min'].cumsum() -
                                                        super_counts_hu[pt]['max'].cumsum()),
                                                       color=colors[0], alpha=0.25)
                objs[0].set_alpha(0)
                ax[int(p / 4), p % 4].plot(binctrs_hu, super_counts_hu[pt]['med'].cumsum(),
                                           '--', color=colors[0], label='median (human)')

                cutoff = max(cutoff, sum((super_counts_hu[pt]['med'].cumsum()) < thr))
        else:
            for animal in animal_names:
                cnum+=1
                cutoff = 0
                for p, pt in enumerate(ptNames):
                    objs = ax[int(p / 4), p % 4].stackplot(binctrs_hu, counts_hu[animal][pt]['max'].cumsum(),
                                                           (counts_hu[animal][pt]['min'].cumsum() -
                                                            counts_hu[animal][pt]['max'].cumsum()),
                                                           color=colors[cnum], alpha=0.25)
                    objs[0].set_alpha(0)
                    ax[int(p / 4), p % 4].plot(binctrs_hu, counts_hu[animal][pt]['med'].cumsum(),
                                               '--', color=colors[cnum], label=animal + ' median (human)')

                    cutoff = max(cutoff, sum((counts_hu[animal][pt]['med'].cumsum()) < thr))
        for p, label in enumerate(ptNames):
            ax[int(p / 4), p % 4].plot(binctrs_model, counts_model[label].cumsum(),
                                       'k-', label='model')
            ax[int(p / 4), p % 4].set_title(label)
            xlim = xlim if xlim is not None else [0, binctrs_hu[cutoff]]
            ax[int(p / 4), p % 4].set_xlim(xlim)
        ax[int(p / 4), p % 4].legend()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel('percent correct keypoints')
    if not pixel_units:
        plt.xlabel('error radius (cm)')
    else:
        plt.xlabel('error radius (pixels)')
    plt.show()


def get_local_maxima(data, x_offset, y_offset, input_width, input_height, image_width, image_height, threshold=0.000002,
                     neighborhood_size=15):
    """ Return the local maxima of the heatmaps
  Args:
    data: the heatmaps
    x_offset : the normalized x_offset coordinate, used to transform the local maxima back to image space
    y_offset : the normalized y_offset coordinate

  """

    keypoints = []

    heatmap_height, heatmap_width, num_parts = data.shape
    heatmap_height = float(heatmap_height)
    heatmap_width = float(heatmap_width)

    input_width = float(input_width)
    input_height = float(input_height)

    image_width = float(image_width)
    image_height = float(image_height)

    for k in range(num_parts):

        data1 = data[:, :, k]
        data_max = filters.maximum_filter(data1, neighborhood_size)
        maxima = (data1 == data_max)
        data_min = filters.minimum_filter(data1, neighborhood_size)
        diff = (data_max - data_min) > (threshold * data1.max())
        maxima[diff == 0] = 0

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        x1, y1, v1 = [], [], []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1) / 2.
            x_center_int = int(np.round(x_center))
            normalized_x_center = x_center * (input_width / heatmap_width) * (1. / image_width) + x_offset

            y_center = (dy.start + dy.stop - 1) / 2.
            y_center_int = int(np.round(y_center))
            normalized_y_center = y_center * (input_height / heatmap_height) * (1. / image_height) + y_offset

            x1.append(float(normalized_x_center))
            y1.append(float(normalized_y_center))
            v1.append(float(data1[y_center_int, x_center_int]))

        keypoints.append({'x': x1, 'y': y1, 'score': v1})

    return keypoints


def show_final_heatmaps(session, outputs, batch, plotcfg, cfg):
    # Extract the outputs
    parts = outputs[1][batch]
    part_visibilities = outputs[2][batch]
    image_height_widths = outputs[4][batch]
    crop_bbox = outputs[5][batch]
    image = outputs[6][batch]
    heatmaps = outputs[-1][batch]

    # Convert the image to uint8
    int8_image = session.run(plotcfg['convert_to_uint8'], {plotcfg['image_to_convert']: image})

    fig = plt.figure("heat maps")
    plt.clf()

    heatmaps = np.clip(heatmaps, 0., 1.)  # Enable double-ended saturation of the image.
    heatmaps = np.expand_dims(heatmaps,
                              0)  # Add another row to the heatmaps array; (Necessary because it has 3 channels?)
    resized_heatmaps = session.run(plotcfg['resize_to_input_size'],
                                   {plotcfg['image_to_resize']: heatmaps})  # Resize the heatmaps
    resized_heatmaps = np.squeeze(resized_heatmaps)  # Get rid of that row we added.

    image_height, image_width = image_height_widths
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    crop_w, crop_h = np.array([crop_x2 - crop_x1, crop_y2 - crop_y1]) * np.array([image_width, image_height],
                                                                                 dtype=np.float32)

    for j in range(cfg.PARTS.NUM_PARTS):
        heatmap = resized_heatmaps[:, :, j]

        fig.add_subplot(plotcfg['num_heatmap_rows'], plotcfg['num_heatmap_cols'], j + 1)
        plt.imshow(int8_image)

        # Rescale the values of the heatmap
        f = interpolate.interp1d([0., 1.], [0, 255])
        int_heatmap = f(heatmap).astype(np.uint8)

        # Add the heatmap as an alpha channel over the image
        blank_image = np.zeros(image.shape, np.uint8)
        blank_image[:] = [255, 0, 0]
        heat_map_alpha = np.dstack((blank_image, int_heatmap))
        plt.imshow(heat_map_alpha)
        plt.axis('off')
        plt.title(cfg.PARTS.NAMES[j])

        # Render the argmax point
        x, y = np.array(np.unravel_index(np.argmax(heatmap), heatmap.shape)[::-1])
        plt.plot(x, y, color=cfg.PARTS.COLORS[j], marker=cfg.PARTS.SYMBOLS[j], label=cfg.PARTS.NAMES[j])

        # Render the ground truth part location
        part_v = part_visibilities[j]
        if part_v:
            indx = j * 2  # There's an x and a y, so we have to go two parts by two parts
            part_x, part_y = parts[indx:indx + 2]

            # We need to transform the ground truth annotations to the crop space
            input_size = cfg.INPUT_SIZE
            part_x = (part_x - crop_x1) * input_size / (crop_x2 - crop_x1)
            part_y = (part_y - crop_y1) * input_size / (crop_y2 - crop_y1)

            plt.plot(part_x, part_y, color=cfg.PARTS.COLORS[j], marker='*', label=cfg.PARTS.NAMES[j])

        plt.show()
        return fig


def show_all_heatmaps(session, outputs, batch, plotcfg, cfg):
    # Extract the outputs
    parts = outputs[1][batch]
    part_visibilities = outputs[2][batch]
    image_height_widths = outputs[4][batch]
    crop_bbox = outputs[5][batch]
    image = outputs[6][batch]

    int8_image = session.run(plotcfg['convert_to_uint8'], {plotcfg['image_to_convert']: image})

    fig = plt.figure('cropped image', figsize=(8, 8))
    plt.clf()
    plt.imshow(int8_image)

    image_height, image_width = image_height_widths
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox

    fig = plt.figure("heatmaps", figsize=(8, 8))
    plt.clf()

    for i in range(8):
        # For each hourglass subunit...
        # Extract out its heatmaps.
        heatmaps = outputs[-8 + i][batch]
        # Constrain the values to 0 and 1.
        heatmaps = np.clip(heatmaps, 0., 1.)
        heatmaps = np.expand_dims(heatmaps, 0)

        resized_heatmaps = session.run(plotcfg['resize_to_input_size'], {plotcfg['image_to_resize']: heatmaps})
        resized_heatmaps = np.squeeze(resized_heatmaps)

        for j in range(cfg.PARTS.NUM_PARTS):
            heatmap = resized_heatmaps[:, :, j]

            ax = fig.add_subplot(plotcfg['num_layer_rows'], plotcfg['num_layer_cols'],
                                 i * plotcfg['num_layer_cols'] + j + 1)
            # Plot the heatmap.
            plt.imshow(heatmap)

            # Rescale the values of the heatmap.
            f = interpolate.interp1d([0., 1.], [0, 255])
            int_heatmap = f(heatmap).astype(np.uint8)

            # Add the heatmap as an alpha channel over the image
            blank_image = np.zeros(image.shape, np.uint8)
            blank_image[:] = [255, 0, 0]
            plt.axis('off')
            if i == 0:
                plt.title(cfg.PARTS.NAMES[j])

            # Render the argmax point.
            x, y = np.array(np.unravel_index(np.argmax(heatmap), heatmap.shape)[::-1])
            plt.plot(x, y, color=cfg.PARTS.COLORS[j], marker=cfg.PARTS.SYMBOLS[j], label=cfg.PARTS.NAMES[j])

            # Render the ground truth part location
            part_v = part_visibilities[j]
            if part_v:
                indx = j * 2
                part_x, part_y = parts[indx:indx + 2]

                # Transform the ground truth annotations to the crop space.
                input_size = cfg.INPUT_SIZE
                w = (crop_x2 - crop_x1)
                h = (crop_y2 - crop_y1)

                part_x = (part_x - crop_x1) / w * input_size
                part_y = (part_y - crop_y1) / h * input_size

                plt.plot(part_x, part_y, color=cfg.PARTS.COLORS[j], marker='*', label=cfg.PARTS.NAMES[j])

                if i == cfg.PARTS.NUM_PARTS:
                    ax.set_xlabel("%0.3f" % (np.linalg.norm(np.array([part_x, part_y]) - np.array([x, y]))), )

            else:
                if i == cfg.PARTS.NUM_PARTS:
                    ax.set_xlabel("Not Visible")

                print("Part not visible")

            print("%s : max %0.3f, min %0.3f" % (cfg.PARTS.NAMES[j], np.max(heatmap), np.min(heatmap)))

    fig.subplots_adjust(wspace=0, hspace=0)

    plt.show()
    return fig


def evaluation(tfrecords, summary_dir, checkpoint_path, cfg,
               num_images=0, show_heatmaps=False, show_layer_heatmaps=False):

    partNames = cfg['PARTS']['NAMES']  # these should match the names of parts we have sigmas for in MARS_pycocotools

    # Initialize the graph.
    graph = tf.Graph()

    with graph.as_default():

        batched_images, batched_bboxes, batched_parts, batched_part_visibilities, batched_image_ids, \
        batched_image_height_widths, batched_crop_bboxes = inputs.input_nodes(
            tfrecords=tfrecords,
            num_parts=cfg.PARTS.NUM_PARTS,
            num_epochs=1,
            batch_size=cfg.BATCH_SIZE,
            num_threads=cfg.NUM_INPUT_THREADS,
            capacity=cfg.QUEUE_CAPACITY,
            shuffle_batch=False,
            cfg=cfg
        )

        # Set batch_norm parameters.
        batch_norm_params = {
            'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
            'epsilon': 0.001,
            'variables_collections': [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
            'is_training': False
        }

        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.00004),
                            biases_regularizer=slim.l2_regularizer(0.00004)) as scope:

            predicted_heatmaps = model_pose.build(
                input=batched_images,
                num_parts=cfg.PARTS.NUM_PARTS,
                num_stacks=cfg.NUM_STACKS
            )

        # Set parameters for the EMA.
        ema = tf.train.ExponentialMovingAverage(
            decay=cfg.MOVING_AVERAGE_DECAY
        )
        shadow_vars = {
            ema.average_name(var): var
            for var in slim.get_model_variables()
        }

        saver = tf.train.Saver(shadow_vars, reshape=True)

        # Set up a node to fetch values from when we run the graph.
        fetches = [batched_bboxes,
                   batched_parts,
                   batched_part_visibilities,
                   batched_image_ids,
                   batched_image_height_widths,
                   batched_crop_bboxes,
                   batched_images] + predicted_heatmaps

        # Create a training coordinator that will control the different threads.
        coord = tf.train.Coordinator()

        # Set up GPU according to config.
        sess_config = tf.ConfigProto(
            log_device_placement=False,
            # device_filters = device_filters,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
            )
        )
        session = tf.Session(graph=graph, config=sess_config)

        with session.as_default():
            # Initialize all the variables.
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()

            # Launch the queue runner threads
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            dataset_image_ids = set()
            gt_annotations = []
            pred_annotations = []
            gt_annotation_id = 1
            gt_image_id = 1
            try:
                if tf.io.gfile.isdir(checkpoint_path):
                    print(checkpoint_path)
                    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
                    print(''.join(['is now:', checkpoint_path]))

                if checkpoint_path is None:
                    print("ERROR: No checkpoint file found.")
                    return

                # Restores from checkpoint
                saver.restore(session, checkpoint_path)

                # Assuming the model_checkpoint_path looks something like:
                #   /my-favorite-path/model.ckpt-0,
                # extract global_step from it.
                global_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
                print("Found model for global step: %d" % (global_step,))

                if show_heatmaps or show_layer_heatmaps:
                    # set up interactive plotting
                    plt.ion()

                    # Set up image converting and image resizing graphs
                    image_to_convert = tf.placeholder(tf.float32)
                    image_to_resize = tf.placeholder(tf.float32)

                    plotcfg = {
                        'image_to_convert': image_to_convert,
                        'convert_to_uint8': tf.image.convert_image_dtype(tf.add(tf.div(image_to_convert, 2.0), 0.5),
                                                                         tf.uint8),
                        'image_to_resize': image_to_resize,
                        'resize_to_input_size': tf.image.resize_bilinear(image_to_resize,
                                                                         size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE]),

                        # for plotting intermediate heatmaps
                        'num_layers_cols': cfg.PARTS.NUM_PARTS,
                        'num_layers_rows': cfg.NUM_STACKS,

                        # for plotting final heatmaps
                        'num_heatmap_cols': 3,
                        'num_heatmap_rows': int(np.ceil(cfg.PARTS.NUM_PARTS / 3.))
                    }
                step = 0
                done = False
                print_str = ', '.join(['Step: %d', 'Time/image network (ms): %.1f'])

                while not coord.should_stop() and not done:
                    t = time.time()
                    outputs = session.run(fetches)
                    dt = time.time() - t

                    for b in range(cfg.BATCH_SIZE):
                        bbox = outputs[0][b]
                        parts = outputs[1][b]
                        part_visibilities = outputs[2][b]
                        image_id = outputs[3][b]
                        image_height_widths = outputs[4][b]
                        crop_bboxes = outputs[5][b]
                        heatmaps = outputs[-1][b]

                        fig_heatmaps = []
                        fig_layers = []
                        if show_heatmaps:
                            fig_heatmaps = show_final_heatmaps(session, outputs, b, plotcfg, cfg)
                        if show_layer_heatmaps:
                            fig_layers = show_all_heatmaps(session, outputs, b, plotcfg, cfg)
                        if show_heatmaps or show_layer_heatmaps:
                            r = input("Press Enter to continue, s to save the current figure, or any other key to exit")
                            if r == 's' or r == 'S':
                                savedir = os.path.join(summary_dir, 'saved_images')
                                if not os.path.exists(savedir):
                                    os.makedirs(savedir)
                                if fig_heatmaps:
                                    fig_heatmaps.savefig(
                                        os.path.join(savedir, str(image_id[0]) + '_' + str(step) + 'heatmaps.png'))
                                    fig_heatmaps.savefig(
                                        os.path.join(savedir, str(image_id[0]) + '_' + str(step) + 'heatmaps.pdf'))
                                if fig_layers:
                                    fig_layers.savefig(
                                        os.path.join(savedir, str(image_id[0]) + '_' + str(step) + 'layers.png'))
                                    fig_layers.savefig(
                                        os.path.join(savedir, str(image_id[0]) + '_' + str(step) + 'layers.pdf'))
                            elif r != "":
                                done = True
                                break

                        # Transform the keypoints back to the original image space.
                        image_height, image_width = image_height_widths
                        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bboxes
                        crop_w, crop_h = np.array([crop_x2 - crop_x1, crop_y2 - crop_y1]) * np.array(
                            [image_width, image_height], dtype=np.float32)

                        if cfg.LOOSE_BBOX_CROP:
                            keypoints = get_local_maxima(heatmaps, crop_x1, crop_y1, crop_w, crop_h, image_width,
                                                         image_height)
                        else:

                            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
                            bbox_w = (bbox_x2 - bbox_x1) * image_width
                            bbox_h = (bbox_y2 - bbox_y1) * image_height

                            if bbox_h > bbox_w:
                                input_size = bbox_h
                            else:
                                input_size = bbox_w

                            keypoints = get_local_maxima(heatmaps, bbox_x1, bbox_y1, input_size, input_size,
                                                         image_width, image_height)

                        selected_scores = []
                        pred_parts = []
                        for count, k in enumerate(keypoints):
                            # Each keypoint prediction may actually have multiple prediction loci
                            # --pick the one with the highest score.
                            s_idx = np.argsort(k['score']).tolist()
                            s_idx.reverse()

                            if len(s_idx) == 0:
                                x = 0
                                y = 0
                                v = 0
                            else:
                                x = k['x'][s_idx[0]] * image_width
                                y = k['y'][s_idx[0]] * image_height
                                v = 1
                                selected_scores.append(k['score'][s_idx[0]])
                            # Store the predicted parts in a list.
                            pred_parts += [x, y, v]
                            # and as separate entries of pred_annotations for part-wise evaluation
                            pred_annotations.append({
                                'image_id': gt_image_id,
                                'keypoints': [x, y, v],
                                'score': selected_scores[-1],
                                'category_id': count + 2
                            })

                        avg_score = np.mean(selected_scores)
                        # Store the results
                        pred_annotations.append({
                            'image_id': gt_image_id,
                            'keypoints': pred_parts,
                            'score': avg_score.item(),
                            'category_id': 1
                        })

                        # Convert the ground-truth parts to unnormalized coords.
                        gt_parts_x = parts[0::2] * image_width
                        gt_parts_y = parts[1::2] * image_height

                        # Put them in the format we need for COCO evaluation.
                        gt_parts = np.transpose(np.vstack([gt_parts_x, gt_parts_y, part_visibilities]), [1, 0])
                        gt_parts = gt_parts.ravel().tolist()

                        # Get the bbox coordinates again.
                        x1, y1, x2, y2 = crop_bboxes * np.array([image_width, image_height, image_width, image_height])
                        w = x2 - x1
                        h = y2 - y1

                        for eval_part in range(int(np.sum(part_visibilities > 0))):
                            gt_annotations.append({
                                "id": gt_annotation_id,
                                "image_id": gt_image_id,
                                "category_id": eval_part + 2,
                                "area": (w * h).item(),
                                "bbox": [x1.item(), y1.item(), w.item(), h.item()],
                                "iscrowd": 0,
                                "keypoints": gt_parts[(eval_part * 3):(eval_part * 3 + 3)],
                                "num_keypoints": 1
                            })
                        gt_annotations.append({
                            "id": gt_annotation_id,
                            "image_id": gt_image_id,
                            "category_id": 1,
                            "area": (w * h).item(),
                            "bbox": [x1.item(), y1.item(), w.item(), h.item()],
                            "iscrowd": 0,
                            "keypoints": gt_parts,
                            "num_keypoints": int(np.sum(part_visibilities > 0))
                        })

                        dataset_image_ids.add(gt_image_id)

                        gt_annotation_id += 1
                        gt_image_id += 1

                    print(print_str % (step, (dt / cfg.BATCH_SIZE) * 1000))
                    step += 1

                    if num_images > 0 and step == num_images:
                        break

            except Exception as e:
                # Report exceptions to the coordinator.
                coord.request_stop(e)

            # When done, ask the threads to stop. Requesting stop twice is fine.
            coord.request_stop()
            # Wait for them to stop before preceding.
            coord.join(threads)

            gt_dataset = {
                'annotations': gt_annotations,
                'images': [{'id': img_id} for img_id in dataset_image_ids],
                'categories': [{'id': id + 1} for id in range(int(np.sum(part_visibilities > 0)) + 1)]
            }

            cocodata = {'gt_keypoints': gt_dataset,
                        'pred_keypoints': pred_annotations,
                        'partNames': partNames}
            if os.path.exists(os.path.join(summary_dir, 'performance_pose.json')):
                os.remove(os.path.join(summary_dir, 'performance_pose.json'))
            with open(os.path.join(summary_dir, 'performance_pose.json'), 'w') as jsonfile:
                json.dump(cocodata, jsonfile)


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
    event_path = os.path.join(project, 'pose', model + '_log')

    ckptfiles = glob.glob(os.path.join(event_path, 'model.ckpt-*.meta'))
    ckptfiles = ['.'.join(f.split('.')[:-1]) for f in ckptfiles]
    ckpt_steps = np.array([int(c) for text in ckptfiles for c in re.compile(r'\d+').findall(os.path.basename(text))])

    event_path = os.path.join(project, 'pose', model + '_log')
    eventfiles = glob.glob(os.path.join(event_path, 'events.out.tfevents.*'))
    eventfiles.sort(key=lambda text: [int(c) for c in re.compile(r'\d+').findall(text)])

    onlyfiles = [f for f in os.listdir(event_path) if os.path.isfile(os.path.join(event_path, f))]
    onlyckpts = ['.'.join(f.split('.')[:-1]) for f in onlyfiles if 'meta' in f]
    onlyckpts.sort(key=lambda text: [int(c) for c in re.compile(r'\d+').findall(text)])

    sz = {event_accumulator.IMAGES: 0}
    steps = np.empty(shape=(0))
    vals = np.empty(shape=(0))
    for f in eventfiles:
        ea = event_accumulator.EventAccumulator(f, size_guidance=sz)
        ea.Reload()

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


def save_best_checkpoint(project, pose_model_names=None):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not pose_model_names:
        pose_model_list = config['pose']
        pose_model_names = pose_model_list.keys()

    for model in pose_model_names:
        log_path = os.path.join(project, 'pose', model + '_log')
        model_path = os.path.join(project, 'pose', model + '_model')
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


def plot_training_progress(project, pose_model_names=None, figsize=(14, 6), logTime=False, omitFirst=0):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not pose_model_names:
        pose_model_list = config['pose']
        pose_model_names = pose_model_list.keys()

    fix, ax = plt.subplots(len(pose_model_names), 1, figsize=figsize, squeeze=False)
    for i, model in enumerate(pose_model_names):

        steps, vals, ckpt_steps, min_step, params = find_best_checkpoint(project, model)
        sm_vals = smooth(vals, steps, params['decay'], params['burnIn'])

        drop = np.argwhere(steps < omitFirst)
        steps = np.delete(steps, drop)
        vals = np.delete(vals, drop)
        sm_vals = np.delete(sm_vals, drop)

        dropckpt = np.argwhere(ckpt_steps < omitFirst)
        ckpt_steps = np.delete(ckpt_steps, dropckpt)
        step_inds = [np.where(steps >= k) for k in ckpt_steps]
        step_inds = [x[0][0] for x in step_inds if len(x[0])]
        ckpt_vals = [sm_vals[x] for x in step_inds]

        ax[i, 0].plot(steps, vals, color='skyblue', label='raw')
        ax[i, 0].plot(steps, sm_vals, color='darkblue', label='smoothed')
        ax[i, 0].plot(ckpt_steps, ckpt_vals, 'ro', label='saved checkpoints')
        ax[i, 0].plot(min_step, sm_vals[np.where(steps == min_step)], 'o', label='best model', markersize=16, markeredgewidth=4,
                      markeredgecolor='orange', markerfacecolor='None')

        ax[i, 0].set_xlabel('Training step')
        ax[i, 0].set_ylabel('Validation loss')
        if logTime:
            ax[i, 0].set_xscale('log')
        ax[i, 0].set_yscale('log')
        ax[i, 0].set_title('Training progress for pose model "' + model + '"')

    ax[0, 0].legend()
    plt.show()


def run_test(project, pose_model_names=None, num_images=0, show_heatmaps=False, show_layer_heatmaps=False):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # get the names of the detectors we'll be training, and which data goes into each.
    if not pose_model_names:
        pose_model_list = config['pose']
        pose_model_names = pose_model_list.keys()

    # Set the logging level.
    tf.logging.set_verbosity(tf.logging.DEBUG)

    performance = {n: None for n in pose_model_names}
    for model in pose_model_names:
        checkpoint_path = os.path.join(project, 'pose', model + '_model')
        if not os.path.isdir(checkpoint_path):
            os.mkdir(project, 'pose', 'model' + '_model')
            checkpoint_path = os.path.join(project, 'pose', model + '_log')
            if not os.path.isdir(checkpoint_path):
                sys.exit("Couldn't find any training output for " + model + ", you should run train first.")
            else:
                print("Couldn't find a saved model for " + model + ", using latest training checkpoint instead.")

        tf_dir = os.path.join(project, 'pose', model + '_tfrecords_pose')
        tfrecords = glob.glob(os.path.join(tf_dir, 'test_dataset-*'))

        summary_dir = os.path.join(project, 'pose', model + '_evaluation')
        if not os.path.isdir(summary_dir):
            os.mkdir(summary_dir)

        cfg = parse_config_file(os.path.join(project, 'pose', 'config_test.yaml'))

        evaluation(
            tfrecords=tfrecords,
            summary_dir=summary_dir,
            checkpoint_path=checkpoint_path,
            num_images=0,
            show_heatmaps=False,
            show_layer_heatmaps=False,
            cfg=cfg
        )

        performance[model] = coco_eval(project, pose_model_names=pose_model_names)

    return performance


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pose model on a tfrecord file')

    parser.add_argument('--project_path', dest='project_path',
                        help='full path to MARS project', type=str, required=True)

    parser.add_argument('--model', dest='model',
                        help='Name of model to evaluate (defaults to all.)',
                        required=False, type=str, default=None)

    parser.add_argument('--num_images', dest='num_images',
                        help='Maximum number of images in test set to analyze. Set to 0 to run on all images.',
                        required=False, type=int, default=0)

    # parser.add_argument('--show_heatmaps', dest='show_heatmaps',
    #                     help='Make figure of final heatmaps for all body parts.',
    #                     action='store_true')
    #
    # parser.add_argument('--show_layer_heatmaps', dest='show_layer_heatmaps',
    #                     help='Make figure of heatmaps after each hourglass layer.',
    #                     action='store_true')
    # parser.set_defaults(show_heatmaps=False, show_layer_heatmaps=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    run_test(
        project=args.project,
        pose_model_names=args.model,
        num_images=args.num_images,
        # show_heatmaps=args.show_heatmaps,
        # show_layer_heatmaps=args.show_layer_heatmaps
    )
