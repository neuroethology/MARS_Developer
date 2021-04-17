"""
File for detecting parts on images without ground truth.
"""
import pdb
import argparse
from io import StringIO
import numpy as np
import sys, os
import json
import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim
from tensorflow.python.util import deprecation
import time
import matplotlib
from matplotlib import pyplot as plt
from scipy import interpolate
from MARSeval.coco import COCO
from MARSeval.cocoeval import COCOeval
from config import parse_config_file
from evaluation import eval_inputs as inputs
import model_pose
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

deprecation._PRINT_DEPRECATION_WARNINGS = False

# change our plotting if we're running in a notebook
try:
  cfg = get_ipython().config
  if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
  pass

def eval_coco(infile=[], gt_keypoints=[], pred_keypoints=[], view='top', parts=[], fixedSigma=''):
  if infile:
    with open(infile) as jsonfile:
      cocodata = json.load(jsonfile)
    gt_keypoints = cocodata['gt_keypoints']
    pred_keypoints = cocodata['pred_keypoints']
    view = cocodata['view']
    parts = cocodata['partNames']

  # Parse things for COCO evaluation.
  gt_coco = COCO()
  gt_coco.dataset = gt_keypoints
  gt_coco.createIndex()
  pred_coco = gt_coco.loadRes(pred_keypoints)

  # Actually perform the evaluation.
  if fixedSigma:
    assert fixedSigma in ['narrow','moderate','wide','ultrawide']
    cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='fixed', useParts=[fixedSigma])
  elif view.lower() == 'top':
    cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='MARS_top', useParts=parts)
  elif view.lower() == 'front':
    cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='MARS_front', useParts=parts)
  else:
    raise ValueError('Camera view must be either top or front')
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()


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
  heatmaps = np.expand_dims(heatmaps, 0)  # Add another row to the heatmaps array; (Necessary because it has 3 channels?)
  resized_heatmaps = session.run(plotcfg['resize_to_input_size'], {plotcfg['image_to_resize']: heatmaps})  # Resize the heatmaps
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
    heatmaps = outputs[-8+i][batch]
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


def process_tfrecord(tfrecords, checkpoint_path, summary_dir, cfg, view='Top', max_iterations=0, show_heatmaps=False, show_layer_heatmaps=False, prep_cocoEval=True):

  # parse the config file.
  cfg = parse_config_file(cfg)
  partNames = cfg['PARTS']['NAMES'] #note, these names should match the names of parts we have defined sigmas for in MARS_pycocotools

  # Set the logging level.
  tf.logging.set_verbosity(tf.logging.DEBUG)

  # Initialize the graph.
  graph = tf.Graph()

  with graph.as_default():
    
    batched_images, batched_bboxes, batched_parts, batched_part_visibilities, batched_image_ids, \
     batched_image_height_widths, batched_crop_bboxes = inputs.input_nodes(
      tfrecords=tfrecords,
      num_parts = cfg.PARTS.NUM_PARTS,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      capacity = cfg.QUEUE_CAPACITY,
      shuffle_batch=True,
      cfg=cfg
    )

    # Set batch_norm parameters.
    batch_norm_params = {
        'decay': cfg.BATCHNORM_MOVING_AVERAGE_DECAY,
        'epsilon': 0.001,
        'variables_collections' : [tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
        'is_training' : False
    }

    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_regularizer=slim.l2_regularizer(0.00004)) as scope:
      
      predicted_heatmaps = model_pose.build(
        input = batched_images, 
        num_parts = cfg.PARTS.NUM_PARTS,
        num_stacks = cfg.NUM_STACKS
      )

    # Set parameters for the EMA.
    ema = tf.train.ExponentialMovingAverage(
      decay=cfg.MOVING_AVERAGE_DECAY
    )   
    shadow_vars = {
      ema.average_name(var) : var
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
      #device_filters = device_filters,
      allow_soft_placement = True,
      gpu_options = tf.GPUOptions(
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
            'convert_to_uint8':     tf.image.convert_image_dtype(tf.add(tf.div(image_to_convert, 2.0), 0.5), tf.uint8),
            'image_to_resize': image_to_resize,
            'resize_to_input_size': tf.image.resize_bilinear(image_to_resize, size=[cfg.INPUT_SIZE, cfg.INPUT_SIZE]),

            # for plotting intermediate heatmaps
            'num_layers_cols': cfg.PARTS.NUM_PARTS,
            'num_layers_rows': 8, # TODO: This should be the # of hourglass units in use.

            # for plotting final heatmaps
            'num_heatmap_cols': 3,
            'num_heatmap_rows': int(np.ceil(cfg.PARTS.NUM_PARTS /3.))
          }

        step = 0
        done = False
        print_str = ', '.join(['Step: %d', 'Time/image network (ms): %.1f'])

        while not coord.should_stop() and not done:
          t = time.time()
          outputs = session.run(fetches)
          dt = time.time() - t

          for b in range(cfg.BATCH_SIZE):
            bbox                = outputs[0][b]
            parts               = outputs[1][b]
            part_visibilities   = outputs[2][b]
            image_id            = outputs[3][b]
            image_height_widths = outputs[4][b]
            crop_bboxes         = outputs[5][b]
            heatmaps            = outputs[-1][b]

            fig_heatmaps = []
            fig_layers = []
            if show_heatmaps:
              fig_heatmaps = show_final_heatmaps(session, outputs, b, plotcfg, cfg)
            if show_layer_heatmaps:
              fig_layers = show_all_heatmaps(session, outputs, b, plotcfg, cfg)
            if show_heatmaps or show_layer_heatmaps:
              r = input("Press Enter to continue, s to save the current figure, or any other key to exit")
              if r == 's' or r == 'S':
                savedir = os.path.join(summary_dir,'saved_images')
                if not os.path.exists(savedir):
                  os.makedirs(savedir)
                if fig_heatmaps:
                  fig_heatmaps.savefig(os.path.join(savedir, str(image_id[0]) + '_' + str(step) + 'heatmaps.png'))
                  fig_heatmaps.savefig(os.path.join(savedir, str(image_id[0]) + '_' + str(step) + 'heatmaps.pdf'))
                if fig_layers:
                  fig_layers.savefig(os.path.join(savedir, str(image_id[0]) + '_' + str(step) + 'layers.png'))
                  fig_layers.savefig(os.path.join(savedir, str(image_id[0]) + '_' + str(step) + 'layers.pdf'))
              elif r != "":
                done = True
                break

             # Transform the keypoints back to the original image space.
            image_height, image_width = image_height_widths
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bboxes
            crop_w, crop_h = np.array([crop_x2 - crop_x1, crop_y2 - crop_y1]) * np.array([image_width, image_height], dtype=np.float32)

            if cfg.LOOSE_BBOX_CROP:
              keypoints = get_local_maxima(heatmaps, crop_x1, crop_y1, crop_w, crop_h, image_width, image_height)
            else:

              bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
              bbox_w = (bbox_x2 - bbox_x1) * image_width
              bbox_h = (bbox_y2 - bbox_y1) * image_height

              if bbox_h > bbox_w:
                input_size = bbox_h
              else:
                input_size = bbox_w

              keypoints = get_local_maxima(heatmaps, bbox_x1, bbox_y1, input_size, input_size, image_width, image_height)

            selected_scores = []
            pred_parts = []
            for k in keypoints:
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

            avg_score = np.mean(selected_scores)
            # Store the results
            pred_annotations.append({
              'image_id' : gt_image_id,
              'keypoints' : pred_parts,
              'score' : avg_score.item(),
              'category_id' : 1
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

            gt_annotations.append({
              "id" : gt_annotation_id,
              "image_id" : gt_image_id,
              "category_id" : 1,
              "area" : (w * h).item(),
              "bbox" : [x1.item(), y1.item(), w.item(), h.item()],
              "iscrowd" : 0,
              "keypoints" : gt_parts,
              "num_keypoints" : int(np.sum(part_visibilities>0 ))
            })

            dataset_image_ids.add(gt_image_id)

            gt_annotation_id += 1
            gt_image_id += 1

          print(print_str % (step, (dt / cfg.BATCH_SIZE) * 1000))
          step += 1

          if max_iterations > 0 and step == max_iterations:
              break

      except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)

      # When done, ask the threads to stop. Requesting stop twice is fine.
      coord.request_stop()
      # Wait for them to stop before preceding.
      coord.join(threads)

      gt_dataset = {
        'annotations' : gt_annotations,
        'images' : [{'id' : img_id} for img_id in dataset_image_ids],
        'categories' : [{ 'id' : 1 }]
      }

      captured_stdout = StringIO()

      if prep_cocoEval:
        cocodata = {'gt_keypoints': gt_dataset, 'pred_keypoints': pred_annotations, 'view':view, 'partNames': partNames}
        with open(os.path.join(summary_dir,'MARS_results_CoCo.json'),'w') as jsonfile:
          json.dump(cocodata,jsonfile)

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        eval_coco(gt_keypoints=gt_dataset, pred_keypoints=pred_annotations, view=view, parts=partNames)
        sys.stdout = old_stdout

      # Store the output as a TF summary, so we can view it in Tensorboard.
      summary_op = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(summary_dir)
      summary = tf.Summary()
      summary.ParseFromString(session.run(summary_op))

      # Since we captured the stdout while we were outputting commandline summaries, we can parse that for our TF
      # summaries right now.
      for line in captured_stdout.getvalue().split('\n'):
        if line != "":
          description, score = line.rsplit("=", 1)
          description = description.strip()
          score = float(score)

          summary.value.add(tag=description, simple_value=score)

          print("%s: %0.3f" % (description, score))
      
      summary_writer.add_summary(summary, global_step)
      summary_writer.flush()
      summary_writer.close()


def parse_args():

    parser = argparse.ArgumentParser(description='Evaluate pose model on a tfrecord file')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--checkpoint_path', dest='checkpoint_path',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=True)
                          
    parser.add_argument('--summary_dir', dest='summary_dir',
                        help='Path to the directory where the results will be saved',
                        required=True, type=str)
                        
    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--view', dest='view',
                        help='Camera position, (Top) or Front',
                        required=False, type=str, default='Top')
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run. Set to 0 to run on all records.',
                        required=False, type=int, default=0)

    parser.add_argument('--show_heatmaps', dest='show_heatmaps',
                        help='Make figure of final heatmaps for all body parts.',
                        action='store_true')

    parser.add_argument('--show_layer_heatmaps', dest='show_layer_heatmaps',
                        help='Make figure of heatmaps after each hourglass layer.',
                        action='store_true')

    parser.add_argument('--no_coco', dest='prep_cocoEval',
                        help='Skip saving CoCo evaluation files.',
                        action='store_false')

    parser.set_defaults(show_heatmaps=False,show_layer_heatmaps=False,prep_cocoEval=True)

    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    args = parse_args()

    process_tfrecord(
      tfrecords=args.tfrecords,
      checkpoint_path=args.checkpoint_path,
      summary_dir=args.summary_dir,
      cfg=args.config_file,
      view=args.view,
      max_iterations=args.max_iterations,
      show_heatmaps = args.show_heatmaps,
      show_layer_heatmaps = args.show_layer_heatmaps,
      prep_cocoEval = args.prep_cocoEval
    )
