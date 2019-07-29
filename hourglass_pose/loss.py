import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

def add_heatmaps_loss(gt_heatmaps, pred_heatmaps, add_summaries, cfg=''):
  """
  Args:
    gt_heatmaps :
        The ground-truth heatmaps.
        (should be list of PART_NUMxINPUT_SIZExINPUT_SIZE images/arrays)
    pred_heatmaps :
        an array of heatmaps with the same shape as gt_heatmaps
  """
  total_loss = 0.0
  summaries = []

  l = 0.0
  for i, pred in enumerate(pred_heatmaps):  # For each hourglass unit...
    l = tf.nn.l2_loss(gt_heatmaps - pred)

    tf.losses.add_loss(l)
    total_loss += l

    if add_summaries:
      summaries.append(tf.summary.scalar('heatmap_loss_%d' % i, l))

  return total_loss, summaries



