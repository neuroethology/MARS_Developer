import cv2
import numpy as np
from scipy.misc import imresize
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import random


def reshape_bboxes(xmin, ymin, xmax, ymax, pad_percentage=0.25):
    """Reshape normalized bounding box coordinates so that they are ready for cropping.
  """

    xmin = np.atleast_1d(np.squeeze(xmin))
    ymin = np.atleast_1d(np.squeeze(ymin))
    xmax = np.atleast_1d(np.squeeze(xmax))
    ymax = np.atleast_1d(np.squeeze(ymax))

    shifted_xmin = []
    shifted_ymin = []
    shifted_xmax = []
    shifted_ymax = []

    for i in range(xmin.shape[0]):

        x1 = xmin[i]
        y1 = ymin[i]
        x2 = xmax[i]
        y2 = ymax[i]

        w = x2 - x1
        h = y2 - y1

        [new_x1, new_x2] = pad_bbox(x1, x2, 0, 1, pad_percentage)
        [new_y1, new_y2] = pad_bbox(y1, y2, 0, 1, pad_percentage)


        shifted_xmin.append(new_x1)
        shifted_ymin.append(new_y1)
        shifted_xmax.append(new_x2)
        shifted_ymax.append(new_y2)

    shifted_xmin = np.array(shifted_xmin).astype(np.float32)
    shifted_ymin = np.array(shifted_ymin).astype(np.float32)
    shifted_xmax = np.array(shifted_xmax).astype(np.float32)
    shifted_ymax = np.array(shifted_ymax).astype(np.float32)

    return [shifted_xmin, shifted_ymin, shifted_xmax, shifted_ymax]


def extract_crop(image, bbox, extract_centered_bbox=False, pad_percentage=0.25):
    """ Extract a bounding box crop from the image.
  Args:
    image : float32 image
    bbox : bbox in image coordinates
    extract_centered_bbox : If True, then a larger area centered around the bbox will be cropped. If False, then just the bbox will be cropped and
      placed in the upper left hand corner.
  Returns:
    np.array : The cropped region of the image
    np.array : The new upper left hand coordinate (x, y). This can be used to offset part locations.
  """

    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = bbox

    if extract_centered_bbox:

        w = x2 - x1
        h = y2 - y1

        center_x = int(np.round(x1 + w / 2.))
        center_y = int(np.round(y1 + h / 2.))

        if w > h:

            pad = np.round(pad_percentage * w / 2.)

            new_x1 = x1 - pad
            new_x2 = x2 + pad
            new_w = np.round(new_x2 - new_x1)
            new_h = new_w
            new_y1 = center_y - new_h / 2.
            new_y2 = center_y + new_h / 2.

        else:

            pad = np.round(pad_percentage * h / 2.)

            new_y1 = y1 - pad
            new_y2 = y2 + pad
            new_h = np.round(new_y2 - new_y1)
            new_w = new_h
            new_x1 = center_x - new_w / 2.
            new_x2 = center_x + new_w / 2.

        new_x1 = int(np.round(new_x1))
        new_x2 = int(np.round(new_x2))
        new_y1 = int(np.round(new_y1))
        new_y2 = int(np.round(new_y2))

        new_w = int(np.round(new_x2 - new_x1))
        new_h = int(np.round(new_y2 - new_y1))

        cropped_bbox = np.zeros([new_h, new_w, 3])

        cropped_idx_x1 = 0 if new_x1 >= 0 else np.abs(new_x1)
        cropped_idx_x2 = new_w if new_x2 <= image_width else new_w - (new_x2 - image_width)
        cropped_idx_y1 = 0 if new_y1 >= 0 else np.abs(new_y1)
        cropped_idx_y2 = new_h if new_y2 <= image_height else new_h - (new_y2 - image_height)

        image_idx_x1 = max(0, new_x1)
        image_idx_x2 = min(image_width, new_x2)
        image_idx_y1 = max(0, new_y1)
        image_idx_y2 = min(image_height, new_y2)

        cropped_bbox[cropped_idx_y1:cropped_idx_y2, cropped_idx_x1:cropped_idx_x2] = image[image_idx_y1:image_idx_y2,
                                                                                     image_idx_x1:image_idx_x2]

        cropped_bbox = cropped_bbox.astype(np.float32)
        upper_left_x_y = np.array([new_x1, new_y1]).astype(np.float32)

    else:

        bbox_x1 = x1
        bbox_x2 = x2
        bbox_y1 = y1
        bbox_y2 = y2

        bbox_image = image[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
        bbox_h, bbox_w = bbox_image.shape[:2]
        max_dim = max(bbox_w, bbox_h)
        cropped_bbox = np.zeros([max_dim, max_dim, 3])
        cropped_bbox[:bbox_h, :bbox_w, :] = bbox_image[:, :, :]

        cropped_bbox = cropped_bbox.astype(np.float32)
        upper_left_x_y = np.array([x1, y1]).astype(np.float32)

    return [cropped_bbox, upper_left_x_y]


def extract_resized_crop_bboxes(image, bboxes, input_size=256):
    """Crop out, tight, resized bounding boxes from an image.

  There could be multiple objects in a given image.

  Args:
    image : np.array [H, W, 3]
    bboxes : np.array [[x1, y1, x2, y2]] Normalized coordinates
    image_size :
    heatmap_size :
    maintain_aspect_ratio :

  Returns:
    preped_images :

  """

    # if image.dtype != np.uint8:
    #  uint8_image = image.astype(np.uint8)
    # else:
    # uint8_image = image

    num_instances = bboxes.shape[0]

    preped_images = np.zeros((0, input_size, input_size, 3), dtype=np.uint8)
    image_height, image_width = image.shape[:2]

    scaled_bboxes = np.round(bboxes * np.array([image_width, image_height, image_width, image_height])).astype(int)

    for i, bbox in enumerate(scaled_bboxes):

        # bbox_x1 = int(np.floor(bbox[0] * image_width))
        # bbox_x2 = int(np.ceil(bbox[2] * image_width))
        # bbox_y1 = int(np.floor(bbox[1] * image_height))
        # bbox_y2 = int(np.ceil(bbox[3] * image_height))

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

        bbox_image = image[bbox_y1:bbox_y2, bbox_x1:bbox_x2]

        bbox_h, bbox_w = bbox_image.shape[:2]
        if bbox_h > bbox_w:
            new_height = input_size
            height_factor = float(1.0)
            width_factor = new_height / float(bbox_h)
            new_width = int(np.round(bbox_w * width_factor))
            im_scale = width_factor
        else:
            new_width = input_size
            width_factor = float(1.0)
            height_factor = new_width / float(bbox_w)
            new_height = int(np.round(bbox_h * height_factor))
            im_scale = height_factor

        im = imresize(
            bbox_image,
            (new_height, new_width)
        )
        im = np.pad(im, ((0, input_size - new_height), (0, input_size - new_width), (0, 0)), 'constant')
        # preped_images[i, 0:im.shape[0], 0:im.shape[1], :] = im

        im = np.expand_dims(im, 0)
        preped_images = np.concatenate([preped_images, im])

    # Make sure the correct types are being returned
    # preped_images = preped_images.astype(np.uint8)
    # preped_images = preped_images.astype(np.float32)
    return [preped_images]


def pad_bbox(bbox_1, bbox_2, min_b, max_b, pad_percentage=0.0):
    '''Pad the edges of a bounding box by a given amount.'''
    # Get the length of the side
    bbox_l = (bbox_2*1.0 - bbox_1*1.0)

    # Compute the tentative padding length
    pad_tentative = pad_percentage*bbox_l/2.

    # Compute the maximum feasible padding amounts
    right_max = max_b - bbox_2
    left_max = bbox_1 - min_b

    # Restrict the pad to what's feasible.
    pad = np.minimum(right_max, left_max)
    pad = np.minimum(pad, pad_tentative)

    # Add the padding
    new_bbox_1 = bbox_1 - pad
    new_bbox_2 = bbox_2 + pad

    # Rectify the bounding boxes to be feasible (Shouldn't be necessary)
    new_bbox_1 = (np.maximum(new_bbox_1, min_b))
    new_bbox_2 = (np.minimum(new_bbox_2, max_b))

    return [new_bbox_1, new_bbox_2]


def build_heatmaps_etc(image, bboxes,
                       all_parts, all_part_visibilities,
                       part_sigmas, areas,
                       scale_sigmas_by_area=False,
                       input_size=256, heatmap_size=64,
                       loose_crop=False, pad_percentage=0.25,
                       left_right_pairs=None,
                       bg_add_target_left_right_pairs=True, bg_add_non_target_parts=True,
                       bg_non_target_include_occluded=True, bg_add_non_target_left_right_pairs=True,
                       do_random_padding=False, random_padding_freq=0.0,
                       random_padding_min=0.1, random_padding_max=0.0,
                       do_random_blurring=False, random_blur_freq=0.0, max_blur=0.0,
                       do_random_noise = False, random_noise_freq = 0.0, random_noise_scale = 0.0,
                       do_jpeg_artifacts = False, random_jpeg_freq = 0.0,
                       random_jpeg_qual_min = 25, random_jpeg_qual_max = 75):
    """
  Args:
    image (uint8)
    bboxes (flat32) : [num instances x 4] normalized coords
    parts (float32) : [num instances x num parts * 2] normalized coords
    part_visibilities (int) : [num instances x num parts]
    part_sigmas (float32) : [num parts]
    areas (float32) : [num parts]

  Returns:
    np.array (uint8) [num instances, input_size, input_size, 3] : The bounding box crops
    np.array (float32) [num instances, heatmap_size, heatmap_size, num parts] : The heatmaps for each instance
    np.array (float32) [num instances, num parts * 2] : The normalized keypoint locations in reference to the crops
  """
    # Get the number of parts and number of instances --doesn't have to be from the visibilities necessarily.
    num_instances, num_parts = all_part_visibilities.shape

    # Get the image dimensions.
    image_height, image_width = image.shape[:2]

    # Combine
    image_width_height = np.array([image_width, image_height])

    float_heatmap_size = float(heatmap_size)
    heat_map_to_target_ratio = float_heatmap_size / input_size

    # Scale the normalized bounding boxes and parts to be in image space
    scaled_bboxes = np.round(bboxes * np.array([image_width, image_height, image_width, image_height])).astype(int)
    scaled_all_parts = (all_parts.reshape([-1, 2]) * image_width_height).reshape([num_instances, num_parts * 2])

    # Initialize the return values
    cropped_bbox_images = np.zeros((num_instances, input_size, input_size, 3), dtype=np.uint8)
    all_heatmaps = np.zeros((num_instances, heatmap_size, heatmap_size, num_parts), dtype=np.float32)
    heatmap_part_locs = np.zeros((num_instances, num_parts * 2), dtype=np.float32)
    background_heatmaps = np.zeros((num_instances, heatmap_size, heatmap_size, num_parts), dtype=np.float32)

    # For each instance, crop out the bounding box, construct the heatmaps, and shift the keypoints
    for i in range(num_instances):
        bbox = scaled_bboxes[i]
        parts = scaled_all_parts[i]
        part_visibilities = all_part_visibilities[i]
        area = areas[i]

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

        if loose_crop:
            if (do_random_padding)&(random.random() < random_padding_freq):
                pad_used = random.uniform(random_padding_min, random_padding_max)
            else:
                pad_used = pad_percentage
            [bbox_x1, bbox_x2] = pad_bbox(bbox_x1, bbox_x2, 0, image_width, pad_used)
            [bbox_y1, bbox_y2] = pad_bbox(bbox_y1, bbox_y2, 0, image_height, pad_used)
            bbox_x1 = int(bbox_x1)
            bbox_x2 = int(bbox_x2)
            bbox_y1 = int(bbox_y1)
            bbox_y2 = int(bbox_y2)


        bbox_image = image[bbox_y1:bbox_y2, bbox_x1:bbox_x2]

        bbox_h, bbox_w = bbox_image.shape[:2]

        # Compute the factor to
        if bbox_h > bbox_w:
            new_height = input_size
            width_factor = new_height / float(bbox_h)
            im_scale = width_factor
        else:
            new_width = input_size
            height_factor = new_width / float(bbox_w)
            im_scale = height_factor

        # Resize the image from its extracted bbox dimensions, to the input dimensions
        if im_scale > 1.:
            im = cv2.resize(bbox_image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        else:
           im = cv2.resize(bbox_image, (input_size, input_size), interpolation=cv2.INTER_AREA)



        ## Do the rest of our image alterations here too.
        if do_random_blurring:
            if np.random.random() < random_blur_freq:
                gaussian_sigma = (random.randint(1, max_blur))
                kernel = cv2.getGaussianKernel(5,gaussian_sigma)
                im = cv2.filter2D(im, -1, kernel)

        if do_random_noise:
            r = np.random.random()
            # print r
            if r < random_noise_freq:
                # print im
                noise_mat = np.random.normal(loc=0.0,scale=random_noise_scale, size=(input_size,input_size))
                # print noise_mat
                noise_mat = np.repeat(noise_mat[:,:,np.newaxis], 3, axis=2)
                im = np.float32(im)
                # new_scale = np.max(im) - np.min(im)
                im = im + (noise_mat*255.)
                # im = (noise_mat*255.)
                im = np.clip(im,0,255)
                im = im.astype(np.uint8)
                # print im

        if do_jpeg_artifacts:
                if np.random.random() < random_jpeg_freq:
                    quality_to_use = random.randrange(random_jpeg_qual_min, random_jpeg_qual_max)
                    _ , imj = cv2.imencode('.jpg', im, (cv2.IMWRITE_JPEG_QUALITY, quality_to_use))
                    im = cv2.imdecode(imj, cv2.CV_LOAD_IMAGE_COLOR)


        cropped_bbox_images[i] = im

        # Offset the parts based on the bounding box
        upper_left_x_y = [bbox_x1, bbox_y1]
        offset_parts = (parts.reshape([-1, 2]) - upper_left_x_y).reshape([-1])

        # Stretch the parts to fit the newly-stretched bounding_box
        parts_x = offset_parts[0::2]
        parts_y = offset_parts[1::2]

        if bbox_h > bbox_w:
            parts_x *= float(bbox_h)/float(bbox_w)
        else:
            parts_y *= float(bbox_w)/float(bbox_h)

        parts_x = np.expand_dims(parts_x,1)
        parts_y = np.expand_dims(parts_y,1)

        new_off= np.stack([parts_x, parts_y], 1)
        offset_parts = new_off.reshape([-1])

        #print(offset_parts)

        # raw_input("")

        # Scale the keypoints for the heatmap size
        scaled_offset_parts = offset_parts * im_scale * heat_map_to_target_ratio

        # Force the keypoints to lie on a pixel
        int_scaled_offset_parts = np.round(scaled_offset_parts).astype(int)

        # If we are scaling the part sigmas, then compute the scale factor
        sigma_scale = im_scale * heat_map_to_target_ratio * np.sqrt(area) * 2.

        for j in range(num_parts):
            ind = j * 2
            x, y = int_scaled_offset_parts[ind:ind + 2]
            v = part_visibilities[j]

            if v > 0:
                # GVH: ignore the image scale issue, and use the sigmas directly
                # sigma_x = im_scale * heat_map_to_target_ratio * np.sqrt(area) * 2. * part_sigmas[j]
                if scale_sigmas_by_area:
                    sigma_x = sigma_scale * part_sigmas[j]
                else:
                    sigma_x = part_sigmas[j]

                sigma_y = sigma_x
                heat_map = two_d_gaussian(x, y, sigma_x, sigma_y, heatmap_size)

                all_heatmaps[i, :, :, j] = heat_map

            else:
                # the heat map blob is prefilled with zeros, so we are good to go.
                pass

        heatmap_part_locs[i] = (int_scaled_offset_parts / float_heatmap_size)[:]

        # Compute the "background" heatmap
        # This will consist of the left/right pair of the parts along with visible parts from other instances that fall within the crop

        target_left_right_swap_background_heatmap = np.zeros([heatmap_size, heatmap_size, num_parts], dtype=np.float32)
        if bg_add_target_left_right_pairs:
            heatmap = all_heatmaps[i]
            for left_idx, right_idx in left_right_pairs:
                left_heatmap = np.copy(heatmap[:, :, left_idx])
                target_left_right_swap_background_heatmap[:, :, left_idx] += heatmap[:, :, right_idx]
                target_left_right_swap_background_heatmap[:, :, right_idx] += left_heatmap

        # Compute the heatmaps from other instances that fall within the target's bounding box
        others_background_heatmap = np.zeros([heatmap_size, heatmap_size, num_parts], dtype=np.float32)
        others_left_right_swap_background_heatmap = np.zeros([heatmap_size, heatmap_size, num_parts], dtype=np.float32)
        if bg_add_non_target_parts:
            if num_instances > 1:
                other_indices = range(num_instances)
                other_indices.remove(i)
                other_parts = scaled_all_parts[other_indices]
                other_part_visibilities = all_part_visibilities[other_indices]
                other_areas = areas[other_indices]
                others_background_heatmap = compute_background_heatmaps(bbox, other_parts, other_part_visibilities,
                                                                        im_scale * heat_map_to_target_ratio,
                                                                        part_sigmas, other_areas, scale_sigmas_by_area,
                                                                        heatmap_size=64,
                                                                        include_occluded=bg_non_target_include_occluded)

            # Compute the left/right swaps for the background instances
            if bg_add_non_target_left_right_pairs:
                for left_idx, right_idx in left_right_pairs:
                    left_heatmap = np.copy(others_background_heatmap[:, :, left_idx])
                    others_left_right_swap_background_heatmap[:, :, left_idx] += others_background_heatmap[:, :,
                                                                                 right_idx]
                    others_left_right_swap_background_heatmap[:, :, right_idx] += left_heatmap

        # Compute the final background heatmap
        background_heatmaps[
            i] = target_left_right_swap_background_heatmap + others_background_heatmap + others_left_right_swap_background_heatmap

    # cropped_bbox_images = cropped_bbox_images.astype(np.float32)
    # all_heatmaps = np.array(all_heatmaps).astype(np.float32)
    # heatmap_part_locs = np.array(heatmap_part_locs).astype(np.float32)
    return [cropped_bbox_images, all_heatmaps, heatmap_part_locs, background_heatmaps]


def get_background_parts(bbox, instance_index, all_parts, all_part_visibilities):
    """Given an instance's bounding box, compute which parts from the other instances overlap this instance.
  Args:
    bbox : [x1, y1, x2, y2], same coordinate system as `all_parts`
    instance_index : the index of the parts that correspond to the instance with `bbox`
    all_parts : [num instances, num parts *2], same coordinates system as `bbox`
    all_part_visibilities : [num instances, num parts]
  """
    num_instances, num_parts = all_part_visibilities.shape

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

    overlapping_parts = [[] for i in range(num_parts)]
    for i in range(num_instances):
        if i == instance_index:
            continue
        for p in range(num_parts):
            v = all_part_visibilities[i][p]
            if v > 0:
                idx = p * 2
                x, y = all_parts[i][idx:idx + 2]
                if x > bbox_x1 and x < bbox_x2:
                    if y > bbox_y1 and y < bbox_y2:
                        overlapping_parts[p] += [x, y]

    return overlapping_parts


def compute_background_heatmaps(bbox, all_parts, all_part_visibilities, scaling_factor, part_sigmas, areas,
                                scale_sigmas_by_area=False, heatmap_size=64, include_occluded=False):
    """
  Args:
    bbox : in image space
    all_parts : in image space. This should probably not contain the parts that correspond with `bbox`
    all_part_visibilites :
    scaling_factor : the value to scale the keypoints by to transform them from image space to heatmap space.
  """
    # Get the number of instances and number of parts.
    num_instances, num_parts = all_part_visibilities.shape

    # Get the coordinate
    upper_left_x_y = bbox[:2]
    offset_bottom_right_x, offset_bottom_right_y = bbox[2:] - upper_left_x_y

    # shift all the parts
    offset_parts = (all_parts.reshape([-1, 2]) - upper_left_x_y).reshape([num_instances, num_parts * 2])

    # Scale the keypoints for the heatmap size
    scaled_offset_parts = offset_parts * scaling_factor

    # Force the keypoints to lie on a pixel
    int_scaled_offset_parts = np.round(scaled_offset_parts).astype(int)

    if include_occluded:
        visibility_at_least = 1
    else:
        visibility_at_least = 2

    heatmaps = np.zeros([heatmap_size, heatmap_size, num_parts], dtype=np.float32)

    for j in range(num_parts):

        # Create a new index for when we reference the 1d version of our 2d array
        ind = j * 2

        # Get the X and Y coordinate of our parts.
        xs, ys = offset_parts[:, ind:ind + 2].T
        v = all_part_visibilities[:, j]

        # Set up conditions to get the valid indices
        x_condition = (xs >= 0) & (xs <= offset_bottom_right_x)
        y_condition = (ys >= 0) & (ys <= offset_bottom_right_y)
        visibility_condition = (v >= visibility_at_least)

        # Get the valid indices.
        index_test = x_condition & y_condition  & visibility_condition
        indices = np.where(index_test)

        visible_parts = int_scaled_offset_parts[:, ind:ind + 2][indices, :].ravel()
        valid_areas = areas[indices].ravel()

        for i in range(sum(index_test)):
            ind_2 = i * 2
            x, y = visible_parts[ind_2:ind_2 + 2]
            area = valid_areas[i]
            if scale_sigmas_by_area:
                sigma_x = scaling_factor * np.sqrt(area) * 2.
            else:
                sigma_x = part_sigmas[j]
            sigma_y = sigma_x
            heat_map = two_d_gaussian(x, y, sigma_x, sigma_y, heatmap_size)

            heatmaps[:, :, j] += heat_map

        else:
            # the heat map blob is prefilled with zeros, so we are good to go.
            pass

    # We may want to clamp at 1?
    return heatmaps


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.
  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


# used
def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
    lc = 0.5
    hc = 1.1

    lg = 0.75
    hg = 1.
    b_delta = 32. / 255.
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=b_delta)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=b_delta)
        else:
            if color_ordering == 0:
                image = apply_random_gamma(image, lg, hg)
                image = tf.image.random_brightness(image, max_delta=b_delta)
                # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                # image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=lc, upper=hc)
                image = apply_random_gamma(image, lg, hg)
            elif color_ordering == 1:
                # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=b_delta)

                image = tf.image.random_contrast(image, lower=lc, upper=hc)
                image = apply_random_gamma(image, lg, hg)
                # image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:

                image = tf.image.random_contrast(image, lower=lc, upper=hc)

                # image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=b_delta)
                image = apply_random_gamma(image, lg, hg)

                # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                # image = tf.image.random_hue(image, max_delta=0.2)
                # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

                image = tf.image.random_contrast(image, lower=lc, upper=hc)
                image = tf.image.random_brightness(image, max_delta=b_delta)
                image = apply_random_gamma(image, lg, hg)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


# used
def distorted_shifted_bounding_box(xmin, ymin, xmax, ymax, num_bboxes, image_height, image_width,
                                   max_num_pixels_to_shift=5):
    """ Distort the bounding box coordinates by a given maximum amount.
  """

    image_width = tf.cast(image_width, tf.float32)
    image_height = tf.cast(image_height, tf.float32)
    one_pixel_width = 1. / image_width
    one_pixel_height = 1. / image_height
    max_width_shift = one_pixel_width * max_num_pixels_to_shift
    max_height_shift = one_pixel_height * max_num_pixels_to_shift

    xmin -= tf.random_uniform([1, num_bboxes], minval=0, maxval=max_width_shift, dtype=tf.float32)
    xmax += tf.random_uniform([1, num_bboxes], minval=0, maxval=max_width_shift, dtype=tf.float32)
    ymin -= tf.random_uniform([1, num_bboxes], minval=0, maxval=max_height_shift, dtype=tf.float32)
    ymax += tf.random_uniform([1, num_bboxes], minval=0, maxval=max_height_shift, dtype=tf.float32)

    # ensure that the coordinates are still valid
    ymin = tf.clip_by_value(ymin, 0.0, 1.)
    xmin = tf.clip_by_value(xmin, 0.0, 1.)
    ymax = tf.clip_by_value(ymax, 0.0, 1.)
    xmax = tf.clip_by_value(xmax, 0.0, 1.)

    return [xmin, ymin, xmax, ymax]

def apply_random_gamma(image, low, high):
    rand_gam = np.random.uniform(low, high)
    new_img = tf.image.adjust_gamma(image, rand_gam)
    return new_img


# used
def two_d_gaussian(center_x, center_y, sigma_x, sigma_y, size):
    x, y = np.arange(size), np.arange(size)
    x = 1.0*x
    y = 1.0*y
    gx = np.exp(-(x - center_x) ** 2 / (2 * sigma_x ** 2))
    gy = np.exp(-(y - center_y) ** 2 / (2 * sigma_y ** 2))
    g = np.outer(gy, gx)
    # g /= np.sum(g)  # normalize, if you want that

    return g.astype(np.float32)


# used
def flip_parts_left_right(parts_x, parts_y, parts_v, left_right_pairs, num_parts):
    """Flip the parts horizontally. The parts are in normalized coordinates
  """

    flipped_parts = np.vstack([np.squeeze(parts_x), np.squeeze(parts_y), np.squeeze(parts_v)]).transpose([1, 0])
    flipped_parts[:, 0] = 1. - flipped_parts[:, 0]

    num_instances = flipped_parts.shape[0] / num_parts

    for i in range(num_instances):
        for left_idx, right_idx in left_right_pairs:
            l = i * num_parts + left_idx
            r = i * num_parts + right_idx
            x, y, v = flipped_parts[l]
            flipped_parts[l] = flipped_parts[r][:]
            flipped_parts[r] = [x, y, v]

    flipped_parts = flipped_parts.astype(np.float32)

    flipped_x = np.expand_dims(flipped_parts[:, 0].ravel(), 0)
    flipped_y = np.expand_dims(flipped_parts[:, 1].ravel(), 0)
    flipped_v = np.expand_dims(flipped_parts[:, 2].ravel().astype(np.int32), 0)

    return [flipped_x, flipped_y, flipped_v]


def flip_heatmaps_left_right(heatmaps, left_right_pairs):
    heatmaps = np.fliplr(heatmaps)
    for left_idx, right_idx in left_right_pairs:
        l = np.copy(heatmaps[:, :, left_idx])
        heatmaps[:, :, left_idx] = heatmaps[:, :, right_idx]
        heatmaps[:, :, right_idx] = l[:, :]
    return heatmaps.astype(np.float32)
