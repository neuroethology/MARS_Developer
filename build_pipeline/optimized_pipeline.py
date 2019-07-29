import time
import os, sys
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt
import cv2

from seqIo import seqIo_reader,parse_ann


class ImportGraphDetection():
    #importing and running isolated detection graphs
    def __init__(self, quant_model):
        #read the graph protbuf file and parse it to retrive the unserilaized graph_def
        with tf.gfile.GFile(quant_model,'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        #load the graph stored in graph_def into graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")

        # create session we use to exectute the model
        sess_config = tf.ConfigProto(
            log_device_placement =False,
            allow_soft_placement = True,
            gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction=.9))

        self.sess = tf.Session(graph=self.graph, config=sess_config)

        #access to input and output nodes
        self.input_op = self.graph.get_operation_by_name('images')
        self.input_tensor = self.input_op.outputs[0]
        self.output_op_loc = self.graph.get_operation_by_name('predicted_locations')
        self.output_tensor_loc = self.output_op_loc.outputs[0]
        self.output_op_conf = self.graph.get_operation_by_name('Multibox/Sigmoid')
        self.output_tensor_conf  =  self.output_op_conf.outputs[0]

    def run(self, input_image):
        return self.sess.run([self.output_tensor_loc,self.output_tensor_conf], {self.input_tensor: input_image})

class ImportGraphPose():
    #importing and running isolated detection graphs
    def __init__(self, quant_model):
        #read the graph protbuf file and parse it to retrive the unserilaized graph_def
        with tf.gfile.GFile(quant_model,'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        #load the graph stored in graph_def into graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name="")

        # create session we use to exectute the model
        sess_config = tf.ConfigProto(
            log_device_placement =False,
            allow_soft_placement = True,
            gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction=.9))

        self.sess = tf.Session(graph=self.graph, config=sess_config)

        #access to input and output nodes
        self.input_op = self.graph.get_operation_by_name('images')
        self.input_tensor = self.input_op.outputs[0]
        self.output_op_heatmaps = self.graph.get_operation_by_name('HourGlass/Conv_30/BiasAdd')
        self.output_tensor_heatmaps = self.output_op_heatmaps.outputs[0]


    def run(self, cropped_images):
        return self.sess.run([self.output_tensor_heatmaps], {self.input_tensor: cropped_images})

def pre_process_image(image):
    prep_image = imresize(image, [299, 299])
    prep_image = prep_image.astype(np.float32)
    prep_image = (prep_image - 128.) / 128.
    prep_image = prep_image.ravel()
    # prepare batch
    return np.expand_dims(prep_image, 0)

def post_process_detection(locations,confidences):
    pred_locs = np.clip(locations[0],0.,1.)
    # we want to filter our proposals that are not in the square
    #pred_locs [x1,y1,x2,y2] in normalized coordinates
    filtered_bboxes = []
    filtered_confs = []
    for bbox, conf in zip(pred_locs,confidences[0]):
        if bbox[0] < 0.: continue
        if bbox[1] < 0.: continue
        if bbox[2] > 1.: continue
        if bbox[3] > 1.: continue
        filtered_bboxes.append(bbox)
        filtered_confs.append(conf)
    filtered_bboxes = np.array(filtered_bboxes)
    filtered_confs = np.array(filtered_confs)
    if filtered_bboxes.shape[0]!= 0 :
        sorted_idxs = np.argsort(filtered_confs.ravel())[::-1]
        filtered_bboxes = filtered_bboxes[sorted_idxs]
        filtered_confs = filtered_confs[sorted_idxs]
        bbox_to_keep = filtered_bboxes[0].ravel()
        conf_to_keep = float(np.asscalar(filtered_confs[0]))
        #are we enough confident?
        if conf_to_keep > .005:
            xmin, ymin,xmax,ymax = bbox_to_keep
            if xmin - .06 > 0.: xmin -= .06
            if xmax + .06 < 1.: xmax += .06
            if ymin - .06 > 0.: ymin -= .06
            if ymax + .06 < 1.: ymax += .06
            return [xmin,ymin,xmax,ymax],conf_to_keep
        else:
            return [],0.

def extract_resize_crop_bboxes(bboxes, IM_W, IM_H, image):
    preped_images = np.zeros((0, 256, 256, 3), dtype=np.uint8)
    scaled_bboxes = np.round(bboxes * np.array([IM_W, IM_H, IM_W, IM_H])).astype(int)
    for i, bbox in enumerate(scaled_bboxes):
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        bbox_image = image[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
        bbox_h, bbox_w = bbox_image.shape[:2]
        if bbox_h > bbox_w:
            new_h = 256
            h_factor = float(1.)
            w_factor = new_h / float(bbox_h)
            new_w = int(np.round(bbox_w * w_factor))
        else:
            new_w = 256
            w_factor = float(1.)
            h_factor = new_w / float(bbox_w)
            new_h = int(np.round(bbox_h * h_factor))

        im = imresize(bbox_image, (new_h, new_w))
        im = np.pad(im, ((0, 256 - new_h), (0, 256 - new_w), (0, 0)), 'constant')
        im = np.expand_dims(im, 0)
        preped_images = np.concatenate([preped_images, im])

    preped_images = preped_images.astype(np.uint8)
    preped_images = preped_images.astype(np.float32)
    preped_images = np.subtract(preped_images, 128.)
    preped_images = np.divide(preped_images, 128.)
    return preped_images

def post_proc_heatmaps(predicted_heatmaps, bboxes, IM_W, IM_H, NUM_PARTS):
        keypoints_res = []
        for b in range(len(predicted_heatmaps[0])):
            heatmaps = predicted_heatmaps[0][b]
            heatmaps = np.clip(heatmaps,0.,1.)
            resized_heatmaps = cv2.resize(heatmaps, (POSE_IM_SIZE,POSE_IM_SIZE) , interpolation=cv2.INTER_LINEAR)
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bboxes[b]
            bbox_w = (bbox_x2 - bbox_x1) * IM_W
            bbox_h = (bbox_y2 - bbox_y1) * IM_H
            if bbox_h > bbox_w:
                new_h = POSE_IM_SIZE
                h_factor = float(1.)
                w_factor = new_h / float(bbox_h)
                new_w = int(np.round(bbox_w * w_factor))
            else:
                new_w = POSE_IM_SIZE
                w_factor = float(1.)
                h_factor = new_w / float(bbox_w)
                new_h = int(np.round(bbox_h * h_factor))
            resized_heatmaps = resized_heatmaps[:new_h, :new_w, :]
            rescaled_heatmaps = cv2.resize(resized_heatmaps, (int(np.round(bbox_w)), int(np.round(bbox_h))), interpolation=cv2.INTER_LINEAR)
            keypoints = np.zeros((NUM_PARTS,3))
            for j in range(NUM_PARTS):
                hm = rescaled_heatmaps[:,:,j]
                score = float(np.max(hm))
                x,y = np.array(np.unravel_index(np.argmax(hm), hm.shape)[::-1])
                imx = x + bbox_x1 * IM_W
                imy = y + bbox_y1 * IM_H
                keypoints[j,:] = [imx,imy,score]
            keypoints_res.append(keypoints)
        return keypoints_res

def closest_sorted(l, x):
    """Returns the index of the closest value in an ascending sorted list l to x."""
    # Get the index of the value on the right side of the zero crossing
    i = np.searchsorted(l, x)
    if i > 0 and abs(l[i] - x) > abs(l[i - 1] - x):
        # The item on the left of the zero crossing exists and was smaller
        return i - 1
    else:
        return i

def create_alignment_mapping(a, b):
    """Given two lists of timestamps, return a new list with the index
    of the closest item in b for each value in a."""
    return [closest_sorted(b, t) for t in a]



DET_IM_SIZE = 299
POSE_IM_SIZE = 256

# QUANT_B_PATH = './quant_detection_black_1.pb'
# QUANT_W_PATH = './quant_detection_white_1.pb'
# QUANT_POSE  =  './quant_pose_11.pb'

QUANT_TOP_B_PATH = './optimized_model_top_detection_black_1.pb'
QUANT_TOP_W_PATH = './optimized_model_top_detection_white_1.pb'
QUANT_TOP_POSE  =  './model_pose_top.pb'

QUANT_FRONT_B_PATH = './optimized_model_front_detection_black_1.pb'
QUANT_FRONT_W_PATH = './optimized_model_front_detection_white_1.pb'
QUANT_FRONT_POSE  =  './model_pose_front.pb'

#import graphs
top_det_black = ImportGraphDetection(QUANT_TOP_B_PATH)
top_det_white = ImportGraphDetection(QUANT_TOP_W_PATH)
top_pose = ImportGraphPose(QUANT_TOP_POSE)

front_det_black = ImportGraphDetection(QUANT_FRONT_B_PATH)
front_det_white = ImportGraphDetection(QUANT_FRONT_W_PATH)
front_pose = ImportGraphPose(QUANT_FRONT_POSE)

#load an image and process it
VIDEO_PATH = '../video/Mouse_2016-05-11_15-10-13/'
VIDEO_NAME = VIDEO_PATH.split('/')[-2]
VIDEO_TOP_NAME = VIDEO_NAME + '_Top_J85.seq'
VIDEO_FRONT_NAME = VIDEO_NAME + '_Front_J85.seq'
if not os.path.exists(VIDEO_PATH + VIDEO_FRONT_NAME):
    VIDEO_FRONT_NAME = VIDEO_NAME + '_FroHi_J85.seq'

#read video
sr_top = seqIo_reader(VIDEO_PATH + VIDEO_TOP_NAME)
NUM_FRAMES_TOP = sr_top.header['allocated_frames']
IM_TOP_H = sr_top.header['image_height']
IM_TOP_W = sr_top.header['image_width']
sr_top.build_seek_table()
if len(sr_top.seek_table) != NUM_FRAMES_TOP:
    sr_top.timestamp_length = 16
    sr_top.build_seek_table()

sr_front = seqIo_reader(VIDEO_PATH + VIDEO_FRONT_NAME)
NUM_FRAMES_FRONT = sr_front.header['allocated_frames']
IM_FRONT_H = sr_front.header['image_height']
IM_FRONT_W = sr_front.header['image_width']
sr_front.build_seek_table()
if len(sr_front.seek_table) != NUM_FRAMES_FRONT:
    sr_front.timestamp_length = 16
    sr_front.build_seek_table()

NUM_FRAMES= min(NUM_FRAMES_TOP, NUM_FRAMES_FRONT)

# get frame
top_prev_loc_b = np.zeros([4,1]);top_prev_ok_conf_b=0
top_prev_loc_w = np.zeros([4,1]);top_prev_ok_conf_w=0

front_prev_loc_b = np.zeros([4,1]);front_prev_ok_conf_b=0
front_prev_loc_w = np.zeros([4,1]);front_prev_ok_conf_w=0

TOP_NUM_PARTS = 7
FRONT_NUM_PARTS = 11
top_pose_frames =[]
front_pose_frames = []

for f in range(NUM_FRAMES):
    tt =time.time()

    # pre process image
    top_image = sr_top.read_frame_by_index(f)[0]
    if len(top_image.shape) != 3:
        new_im = np.empty((IM_TOP_H,IM_TOP_W, 3), dtype=np.uint8)
        new_im[:,:,:] = top_image[:,:, np.newaxis]
        top_image = new_im.astype(np.float32)

    front_image = sr_front.read_frame_by_index(f)[0]
    if len(front_image.shape) != 3:
        new_im = np.empty((IM_FRONT_H,IM_FRONT_W, 3), dtype=np.uint8)
        new_im[:,:,:] = front_image[:,:, np.newaxis]
        front_image = new_im.astype(np.float32)

    ##################################### detection #####################################
    # pre process the image
    top_input_image = pre_process_image(top_image)
    front_input_image = pre_process_image(front_image)


    #run detection!
    # t = time.time()
    top_locations_b, top_confidences_b = top_det_black.run(top_input_image)
    front_locations_b, front_confidences_b = front_det_black.run(front_input_image)

    # dt = time.time() - t

    # t = time.time()
    top_bbox_to_pass_b, top_conf_to_pass_b = post_process_detection(top_locations_b,top_confidences_b)
    if top_conf_to_pass_b > .005:
        top_prev_ok_loc_b = top_bbox_to_pass_b; top_prev_ok_conf_b = top_conf_to_pass_b
    else:
        top_bbox_to_pass_b = top_prev_ok_loc_b; top_conf_to_pass_b = top_prev_ok_conf_b
    front_bbox_to_pass_b, front_conf_to_pass_b = post_process_detection(front_locations_b,front_confidences_b)
    if front_conf_to_pass_b > .005:
        front_prev_ok_loc_b = front_bbox_to_pass_b; front_prev_ok_conf_b = front_conf_to_pass_b
    else:
        front_bbox_to_pass_b = front_prev_ok_loc_b; front_conf_to_pass_b = front_prev_ok_conf_b
    # dtt = time.time() - t
    # print(" %d - Exectution time: %.2f (ms), post proc time: %.2f (ms)" % (f, dt * 1000, dtt * 1000))


    # t = time.time()
    top_locations_w, top_confidences_w = top_det_white.run(top_input_image)
    front_locations_w, front_confidences_w = front_det_white.run(front_input_image)

    # dt = time.time() - t

    # t = time.time()
    top_bbox_to_pass_w, top_conf_to_pass_w = post_process_detection(top_locations_w,top_confidences_w)
    if top_conf_to_pass_w > .005:
        top_prev_ok_loc_w = top_bbox_to_pass_w; top_prev_ok_conf_w = top_conf_to_pass_w
    else:
        top_bbox_to_pass_w = top_prev_ok_loc_w; top_conf_to_pass_w = top_prev_ok_conf_w
    front_bbox_to_pass_w, front_conf_to_pass_w = post_process_detection(front_locations_w,front_confidences_w)
    if front_conf_to_pass_w > .005:
        front_prev_ok_loc_w = front_bbox_to_pass_w; front_prev_ok_conf_w = front_conf_to_pass_w
    else:
        front_bbox_to_pass_w = front_prev_ok_loc_w; front_conf_to_pass_w = front_prev_ok_conf_w
    # dtt = time.time() - t
    # print(" %d - Exectution time: %.2f (ms), post proc time: %.2f (ms)" % (f, dt * 1000, dtt * 1000))


    #check that detection went ok
    # fig = plt.figure(figsize=(DET_IM_SIZE,DET_IM_SIZE), frameon=False)
    # ax = fig.add_axes([0.0,0.0,1.0,1.0])
    # bbox_to_pass_b = np.array(bbox_to_pass_b)
    # bbox_to_pass_w = np.array(bbox_to_pass_w)
    # plt.imshow(imresize(sr.read_frame_by_index(f)[0],[DET_IM_SIZE,DET_IM_SIZE]),cmap='gray')
    # xmin_b,xmax_b = bbox_to_pass_b[[0,2]] * DET_IM_SIZE
    # ymin_b,ymax_b = bbox_to_pass_b[[1,3]] * DET_IM_SIZE
    # plt.plot([xmin_b,xmax_b,xmax_b,xmin_b,xmin_b],[ymin_b,ymin_b,ymax_b,ymax_b,ymin_b],'r-')
    # xmin_w,xmax_w = bbox_to_pass_w[[0,2]] * DET_IM_SIZE
    # ymin_w,ymax_w = bbox_to_pass_w[[1,3]] * DET_IM_SIZE
    # plt.plot([xmin_w,xmax_w,xmax_w,xmin_w,xmin_w],[ymin_w,ymin_w,ymax_w,ymax_w,ymin_w],'b-')
    # plt.show(block=False)

    ########################## pose ##########################################
    #preprocess bbox for input to hourglass
    #crop out , tight and resize bboxes from an image

    top_bboxes = np.array([top_bbox_to_pass_b,top_bbox_to_pass_w])
    top_preped_images  = extract_resize_crop_bboxes(top_bboxes, IM_TOP_W, IM_TOP_H, top_image)
    front_bboxes = np.array([front_bbox_to_pass_b,front_bbox_to_pass_w])
    front_preped_images  = extract_resize_crop_bboxes(front_bboxes, IM_FRONT_W, IM_FRONT_H, front_image)

    #run hourglass
    # t = time.time()
    top_predicted_heatmaps = top_pose.run(top_preped_images)
    front_predicted_heatmaps = front_pose.run(front_preped_images)

    # dt = time.time() - t

    #post process heatmaps to get keypoints
    # t = time.time()
    top_keypoints_res =  post_proc_heatmaps(top_predicted_heatmaps, top_bboxes, IM_TOP_W, IM_TOP_H, TOP_NUM_PARTS)
    front_keypoints_res =  post_proc_heatmaps(front_predicted_heatmaps, front_bboxes, IM_FRONT_W, IM_FRONT_H, FRONT_NUM_PARTS)

    # dtt = time.time() - t
    # print(" %d - Exectution time: %.2f (ms), post proc: %.2f (ms)" % (f, dt * 1000, dtt * 1000))

    # check pose
    # plt.ion()
    # plt.figure()
    # plt.imshow(image,cmap='gray')
    # for i in range(NUM_PARTS):
    #     plt.plot(keypoints_res[0][i, 0],keypoints_res[0][i, 1],'ro')
    #     plt.plot(keypoints_res[1][i, 0], keypoints_res[1][i, 1], 'bo')
    # pdb.set_trace()
    # plt.close()

    dttt = time.time() - tt
    print(" %d - total time/image: %.2f (ms)" % (f, dttt * 1000))

    top_pose_frames.append(top_keypoints_res)
    front_pose_frames.append(front_keypoints_res)


sr_top.close()
sr_front.close()
np.savez(VIDEO_PATH + 'pose.npz', top_pose = top_pose_frames, front_pose = front_pose_frames)


################################################################# feature extraction





################################################################## actions classification





























############################## actions annotations #########################################

# f_ann = [each for each in os.listdir(VIDEO_PATH) if each.endswith('.txt')]
# actions_frame = []
# actions_idx = []
# if f_ann:
#     print 'Extracting actions'
#     ann_dict = parse_ann(VIDEO_PATH + '/' + f_ann[0])
#     actions_frame = ann_dict['action_frame']
#     actions_idx = ann_dict['type_frame']
#     print 'Actions extraced'
# else:
#     print ('Annotation file not found')

########################################################################





















# # verify that we can access to the list of ops in the graph
# for op in graph.get_operations():
#     print op.name

#utility to write the graph and visualize it to tensorboard
# model_writer = tf.summary.FileWriter('.')
# model_writer.add_graph(sess.graph)






