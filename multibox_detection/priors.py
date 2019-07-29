"""
Code to cluster aspect ratios, for generating dataset specific priors.
"""

from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import imread, imresize
from sklearn.cluster import KMeans
import pdb

def generate_aspect_ratios(dataset, num_aspect_ratios=11, visualize=True, warp_bboxes=True):
  """
  Args:
    dataset (list): A list of image data, as returned by one of the dataset functions
    num_aspect_ratios (int) : The number of aspect ratios to return
    warp_aspect_ratios : If True, then the bounding box coordinates will be warped such that the image is square prior to computing
      the aspect ratio.
  """
  small_epsilon = 1e-10

  feature_vectors = []
  image_ids = []
  original_bboxes = []

  if visualize:
    plt.ion()

  for image_data in dataset:
    
    bbox_xmin =  np.atleast_2d(image_data['object']['bbox']['xmin']).T
    bbox_xmax =  np.atleast_2d(image_data['object']['bbox']['xmax']).T 
    bbox_ymin =  np.atleast_2d(image_data['object']['bbox']['ymin']).T
    bbox_ymax =  np.atleast_2d(image_data['object']['bbox']['ymax']).T 
    
    bboxes = np.hstack([bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax])
    original_bboxes.extend(bboxes.tolist())
    
    if warp_bboxes:
      image_width = float(image_data['width'])
      image_height = float(image_data['height'])

      if image_width > image_height:
        s = image_width / image_height
        bbox_ymin *= s
        bbox_ymax *= s
      else:
        s = image_height / image_width
        bbox_xmin *= s
        bbox_xmax *= s
      
    aspect_ratios = (bbox_xmax - bbox_xmin) / (bbox_ymax - bbox_ymin)

    feature_vectors.extend(aspect_ratios.tolist())
    image_ids.extend([image_data['id']] * bboxes.shape[0])
  
  
  image_ids = np.array(image_ids)  
  X = np.array(feature_vectors)
  original_bboxes = np.array(original_bboxes)

  # little bit of sanity checking
  i = np.isclose(X[:,0], 5, 5) # make sure the aspect ratio isn't crazy, somwhere in the range of [0, 10] should be fine
  X = X[i] 
  image_ids = image_ids[i]
  original_bboxes = original_bboxes[i]

  i = ~np.isinf(X).any(axis=1) # there could have been a divide by 0 
  X = X[i]
  image_ids = image_ids[i]
  original_bboxes = original_bboxes[i]
  
  # We can visualize the aspect ratios
  if visualize:
    plt.figure('aspect ratios')
    plt.scatter(X, np.zeros_like(X))
    plt.title('Aspect Ratios')
    plt.show()
  
  # Do the clustering
  cluster = KMeans(n_clusters=num_aspect_ratios, n_jobs=8)
  cluster.fit(X)

  labels = cluster.labels_
  cluster_centers = cluster.cluster_centers_

  labels_unique = np.unique(labels)
  n_clusters_ = len(labels_unique)
  
  # Sort the clusters by membership count
  cnt = Counter(list(labels.ravel()))
  cf = [[f, cnt[i]] for i, f in enumerate(cluster_centers)]
  cf.sort(key=lambda x: x[1])
  cf.reverse()
  
  print "Clusters:"
  for aspect_ratio, c in cf:
    print "Aspect ratio %0.4f, membership count %d" % (aspect_ratio, c)
  
  aspect_ratios = np.array([x[0] for x in cf])

  # Lets try to render these cluster centers
  if visualize:
    plt.figure('Cluster Centers')
    img_w = 299
    img_h = 299
    scale = 0.1
    img = np.zeros([img_h, img_w, 3])
    for i, aspect_ratio in enumerate(aspect_ratios):
      
      print "Cluster %d" % (i,)
      
      w = scale * np.sqrt(aspect_ratio)
      h = scale / np.sqrt(aspect_ratio)
      
      print "%0.3f width x %0.3f height" % (w, h)
      
      center_i = 0.5
      center_j = 0.5

      x1 = center_j - (w / 2.)
      x2 = center_j + (w / 2.)
      y1 = center_i - (h / 2.)
      y2 = center_i + (h / 2.)
      
      plt.imshow(img)
      
      xmin, ymin, xmax, ymax = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
      bbox_w = xmax - xmin
      bbox_h = ymax - ymin 
      print "BBox: (%d, %d) to (%d, %d) [%d width x %d height] [%0.3f aspect ratio]" % (int(xmin), int(ymin), int(xmax), int(ymax), int(xmax - xmin), int(ymax - ymin), aspect_ratio)
      plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
      
      plt.show()
      
      t = raw_input("push enter")
      if t != '':
        break
      
      plt.clf()
  
    # Visualize the images at the cluster centers
    dataset_dict = {image['id'] : image for image in dataset}
    num_images_to_show = 6
    num_rows = 3
    num_cols = 2
    assert num_images_to_show <= num_rows * num_cols
    fig = plt.figure('Cluster Centers on Images')
    for i, cluster_center in enumerate(aspect_ratios):
      dists = np.linalg.norm(X - cluster_center, axis=1)
      min_indices = np.argsort(dists)
      
      plt.clf()
      fig.suptitle("Aspect Ratio: %0.3f" % (cluster_center,))
      
      for j in range(num_images_to_show):
        min_idx = min_indices[j]
        
        image_id = image_ids[min_idx]
        image_path = dataset_dict[image_id]['filename']
        
        image = imread(image_path)
        
        if warp_bboxes:
          image = imresize(image, [299, 299, 3])
        
        height, width = image.shape[:2]
        
        fig.add_subplot(num_rows, num_cols, j+1)
        plt.imshow(image)
        
        gt_bbox = original_bboxes[min_idx]
        xmin, ymin, xmax, ymax = gt_bbox * np.array([width, height, width, height])
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')
        plt.axis('off')
        
      
      plt.show()
    
      t = raw_input("push enter")
      if t != '':
        break

  return aspect_ratios

def generate_priors(aspect_ratios, min_scale=0.1, max_scale=0.95, restrict_to_image_bounds=True):
  """
  Args:
    aspect_ratios (list): a list of aspect ratios
    min_scale (float): The scale of the boxes, in the range (0, 1], for the 8x8 grid size
    max_scale (float): The scale of the boxes, in the range (0, 1], for the 1x1 grid size of the dection head
    restrict_to_image_bounds (bool): If True, each box will be scaled (in an aspect preservering way) such that
      the entire box is contained within the image bounds. If False, each box will be clipped (not preserving the
      aspect ratio) such that the entire box is contained in the image bounds.
  """
  # These grids correspond to the detection heads.
  grids = [8, 6, 4, 3, 2, 1]
  num_scales = len(grids)
  scales = []
  for i in range(1, num_scales+1):
    scales.append(min_scale + (max_scale - min_scale) * (i - 1) / (num_scales - 1))

  prior_bboxes = []
  for k, (grid, scale) in enumerate(zip(grids, scales)):
    
    # special case for the 1x1 cell ( we only need one aspect ratio)
    if grid == 1:
      
      center_i = 0.5
      center_j = 0.5

      #aspect ratio
      a = 1.
      
      w = scale * np.sqrt(a)
      h = scale / np.sqrt(a)
      
      x1 = center_j - (w / 2.)
      x2 = center_j + (w / 2.)
      y1 = center_i - (h / 2.)
      y2 = center_i + (h / 2.)
      
      # we may need to rescale if this box goes out of the image bounds
      # we want to respect the aspect ratio and the center location
      if restrict_to_image_bounds:
        right_trim = abs(min(0, x1))
        left_trim = abs(min(0, 1-x2))
        top_trim = abs(min(0, y1))
        bottom_trim = abs(min(0, 1-y2))
        
        width_trim = max(right_trim, left_trim)
        height_trim = max(top_trim, bottom_trim)
        
        trim = max(width_trim, height_trim)
        
        if h > w:
          width_trim = trim * a
          height_trim = trim
        else:
          width_trim = trim 
          height_trim = trim / a
          
        x1_t = x1 + width_trim
        x2_t = x2 - width_trim
        y1_t = y1 + height_trim
        y2_t = y2 - height_trim
        
        x1 = min(x1_t, x2_t)
        x2 = max(x1_t, x2_t)
        y1 = min(y1_t, y2_t)
        y2 = max(y1_t, y2_t)
      
      bbox = [
        max(x1, 0.),
        max(y1, 0.),
        min(x2, 1.),
        min(y2, 1.)
      ]
      
      prior_bboxes.append(bbox)
      
    else:
      for i in range(grid):
        for j in range(grid):

          center_i = (i + 0.5) / grid
          center_j = (j + 0.5) / grid
      
          for a in aspect_ratios:
            
            w = scale * np.sqrt(a)
            h = scale / np.sqrt(a)
            
            x1 = center_j - (w / 2.)
            x2 = center_j + (w / 2.)
            y1 = center_i - (h / 2.)
            y2 = center_i + (h / 2.)
            
            if restrict_to_image_bounds:
              right_trim = abs(min(0, x1))
              left_trim = abs(min(0, 1-x2))
              top_trim = abs(min(0, y1))
              bottom_trim = abs(min(0, 1-y2))
              
              width_trim = max(right_trim, left_trim)
              height_trim = max(top_trim, bottom_trim)
              
              trim = max(width_trim, height_trim)
              
              if h > w:
                width_trim = trim * a
                height_trim = trim
              else:
                width_trim = trim 
                height_trim = trim / a
                
              x1_t = x1 + width_trim
              x2_t = x2 - width_trim
              y1_t = y1 + height_trim
              y2_t = y2 - height_trim
              
              x1 = min(x1_t, x2_t)
              x2 = max(x1_t, x2_t)
              y1 = min(y1_t, y2_t)
              y2 = max(y1_t, y2_t)
            
            bbox = [
              max(x1, 0.),
              max(y1, 0.),
              min(x2, 1.),
              min(y2, 1.)
            ]
            
            prior_bboxes.append(bbox)
  
  return prior_bboxes     

def PrintBox(loc, height, width, color='red'):
    """A utility function to help visualizing boxes."""
    xmin, ymin, xmax, ymax = loc[0] * width, loc[1] * height, loc[2] * width, loc[3] * height 
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], '-', color=color)
    

def show_priors_at_cell(priors, cell_offset, num_priors_per_cell, image_height, image_width):
  
  colors = ['red', 'green', 'blue', 'white', 'yellow', 'aquamarine', 
  'burlywood', 'darkgreen', 'deepskyblue', 'hotpink', 
  'lemonchiffon', 'lemonchiffon', 'mintcream', 'salmon', 'royalblue']
  
  image = np.zeros([image_height, image_width, 3])
  plt.imshow(image)
  
  for i in range(num_priors_per_cell):
    PrintBox(priors[num_priors_per_cell*cell_offset + i], image_height, image_width, colors[i % len(colors)])
  
  
  
def visualize_priors(priors, num_priors_per_cell=11, image_height=299, image_width=299):
  

  # Offsets for the cells
  offset_data = [
    ('8x8 cell', 8*4 + 4),
    ('6x6 cell', 8*8 + 6 * 3 + 2),
    ('4x4 cell', 8*8 + 6*6 + 4*2 + 2),
    ('3x3 cell', 8*8 + 6*6 + 4*4 + 3*1 + 1),
    ('2x2 cell', 8*8 + 6*6 + 4*4 + 3*3 + 2*1 + 0)
  ]
  
  plt.figure('Priors')
  for cell_name, offset in offset_data:
    plt.clf()
    show_priors_at_cell(priors, offset, num_priors_per_cell, image_height, image_width)
    plt.title(cell_name)
    plt.show()
    
    t = raw_input("push enter")
    if t != '':
      break