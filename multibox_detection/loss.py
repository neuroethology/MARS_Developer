import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import tensorflow.contrib.slim as slim

SMALL_EPSILON = 1e-10

def compute_assignments(locations, confidences, gt_bboxes, num_gt_bboxes, batch_size, alpha):
  """
  locations: [batch_size * num_predictions, 4]
  confidences: [batch_size * num_predictions]
  gt_bboxes: [batch_size, max num gt_bboxes, 4]
  num_gt_bboxes : [batch_size]  The number of gt bboxes in each image of the batch
  """
  
  num_predictions = locations.shape[0] / batch_size
  assignment_partitions = np.zeros(batch_size * num_predictions, dtype=np.int32)
  #stacked_gt_bboxes = []
  stacked_gt_bboxes = np.zeros([0, 4], dtype=np.float32)
  
  log_confidences = np.log(confidences)
  v = 1. - confidences
  v[v > 1.] = 1.
  v[v <= 0] = SMALL_EPSILON
  log_one_minus_confidences = np.log(v)
  
  # Go through each image in the batch
  for b in range(batch_size):
    
    offset = b * num_predictions
    
    # we need to construct the cost matrix
    C = np.zeros((num_predictions, num_gt_bboxes[b]))
    for j in range(num_gt_bboxes[b]):
      C[:, j] = (alpha / 2.) * (np.linalg.norm(locations[offset:offset+num_predictions] - gt_bboxes[b][j], axis=1))**2 - log_confidences[offset:offset+num_predictions] + log_one_minus_confidences[offset:offset+num_predictions]
    
    #print C
    
    # Compute the assignments
    row_ind, col_ind = linear_sum_assignment(C)
    
    #print row_ind, col_ind
    
    for r, c in zip(row_ind, col_ind):
      assignment_partitions[offset + r] = 1
      #stacked_gt_bboxes.append(gt_bboxes[b][c])
      gt_box = gt_bboxes[b][c].reshape([1,4])
      stacked_gt_bboxes = np.concatenate((stacked_gt_bboxes, gt_box))
      
  #stacked_gt_bboxes = np.array(stacked_gt_bboxes).astype(np.float32)
  stacked_gt_bboxes = stacked_gt_bboxes.astype(np.float32)
    
  return [assignment_partitions, stacked_gt_bboxes]

def add_loss(locations, confidences, batched_bboxes, batched_num_bboxes, bbox_priors, location_loss_alpha):
  
  batch_size = tf.shape(locations)[0]#locations.get_shape().as_list()[0]

  with tf.name_scope("loss"):
    # ground truth bounding boxes:
    # [batch_size, # of ground truth bounding boxes, 4]
    # we also need to know the number of ground truth bounding boxes for each image in the batch
    # (it can be different for each image...)
    # We could assume 1 for now.
    
    # Pass the locations, confidences, and ground truth labels to the matching function
    locations = tf.reshape(locations, [-1, 4])
    confidences = tf.reshape(confidences, [-1])
    
    # add the priors to the predicted residuals
    locations += tf.tile(bbox_priors, [batch_size, 1])
    
    # add a small epsilon to the confidences
    confidences += SMALL_EPSILON
    
    # print "Shapes"
    # print locations.get_shape().as_list()
    # print confidences.get_shape().as_list()
    # print batched_bboxes.get_shape().as_list()
    # print batched_num_bboxes.get_shape().as_list()
    params = [locations, confidences, batched_bboxes, batched_num_bboxes, batch_size, location_loss_alpha]
    matching, stacked_gt_bboxes = tf.py_func(compute_assignments, params, [tf.int32, tf.float32], name="bipartite_matching") 
    
    # matching: [num_predictions * batch_size] 0s and 1s for partitioning
    # stacked_gt_bboxes : [total number of gt bboxes for this batch, 4]
    
    # dynamic partition the bounding boxes and confidences into "positives" and "negatives"
    unmatched_locations, matched_locations = tf.dynamic_partition(locations, matching, 2)
    unmatched_confidences, matched_confidences = tf.dynamic_partition(confidences, matching, 2)
    
    # Because we just did a dynamic partition, it could be the case that either the unmatched or matched matrices is empty. 
    # It could also be the case that there were no ground truth bboxes in this batch.
    # Lets tack on some default values so that the loss calculations are well behaved. 
    matched_locations = tf.concat(axis=0, values=[matched_locations, tf.zeros([1, 4])])
    stacked_gt_bboxes = tf.concat(axis=0, values=[stacked_gt_bboxes, tf.zeros([1, 4])])
    matched_confidences = tf.concat(axis=0, values=[matched_confidences, tf.ones([1])])
    unmatched_confidences = tf.concat(axis=0, values=[unmatched_confidences, tf.zeros([1])])
    
    
    location_loss = location_loss_alpha * tf.nn.l2_loss(matched_locations - stacked_gt_bboxes)
    confidence_loss = -1. * tf.reduce_sum(tf.log(matched_confidences)) - tf.reduce_sum(tf.log((1. - unmatched_confidences) + SMALL_EPSILON))
    
    # It could be the case that there are no ground truth bounding boxes
    # num_gt_bboxes = tf.reduce_sum(batched_num_bboxes)
    
    # loc_loss = lambda: location_loss_alpha * tf.nn.l2_loss(matched_locations - stacked_gt_bboxes)
    # zero_loc_loss = lambda: tf.zeros(shape=[])
    # location_loss = tf.cond(num_gt_bboxes > 0, loc_loss, zero_loc_loss)

    # conf_loss = lambda: -1. * tf.reduce_sum(tf.log(matched_confidences)) - tf.reduce_sum(tf.log((1. - unmatched_confidences) + SMALL_EPSILON))
    # all_negative_conf_loss = lambda : -1. * tf.reduce_sum(tf.log((1. - unmatched_confidences) + SMALL_EPSILON))
    # confidence_loss = tf.cond(num_gt_bboxes > 0, conf_loss, all_negative_conf_loss)
    
    slim.losses.add_loss(location_loss)
    slim.losses.add_loss(confidence_loss)

  return location_loss, confidence_loss