from __future__ import print_function,division
import numpy as np
import scipy.io as sp
import json
import yaml
import os
import sys
import cv2
from collections.abc import Iterable
import scipy.spatial.distance as dist
import scipy.signal as sig
import math as mh
from skimage.transform import resize as rs
import warnings
from itertools import zip_longest
import csv
import json
import cmath as cmh
import time
import scipy.io as sio
import numpy.core.records as npc
warnings.filterwarnings('ignore')
sys.path.append('./')
from Util.genericVideo import *
from behavior_classification.MARS_feature_machinery import *


flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, Iterable) else (a,)))


def extract_features_top(sequence, cfg, progress_bar_sig=''):
    # there's a better version of this code in prep, for now we're leaving out user selection of features.

    video_fullpath = sequence['metadata']['video']
    video_name = os.path.basename(video_fullpath)

    num_points = len(sequence['scores'][0][0])
    # align frames
    ext = video_fullpath[-3:]

    reader = vidReader(video_fullpath)
    num_frames = reader.NUM_FRAMES
    im_h = reader.IM_H
    im_w = reader.IM_W
    fps = reader.fps

    pix_per_cm = cfg['pixels_per_cm']
    eps = np.spacing(1)
    parts = ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base']

    features = [
        'nose_x', 'nose_y',
        'right_ear_x', 'right_ear_y',
        'left_ear_x', 'left_ear_y',
        'neck_x', 'neck_y',
        'right_side_x', 'right_side_y',
        'left_side_x', 'left_side_y',
        'tail_base_x', 'tail_base_y',

        'centroid_x', 'centroid_y',
        'centroid_head_x', 'centroid_head_y',
        'centroid_hips_x', 'centroid_hips_y',
        'centroid_body_x', 'centroid_body_y',
        'phi',
        'ori_head',
        'ori_body',
        'angle_head_body_l',
        'angle_head_body_r',
        'major_axis_len',
        'minor_axis_len',
        'axis_ratio',
        'area_ellipse',
        'dist_edge_x',
        'dist_edge_y',
        'dist_edge',

        'speed',
        'speed_centroid',
        'acceleration',
        'acceleration_centroid',
        'speed_fwd',

        # pixel based features
        'resh_twd_itrhb',
        'pixel_change_ubbox_mice',
        'pixel_change',        # patches
        'nose_pc', 'right_ear_pc', 'left_ear_pc', 'neck_pc', 'right_side_pc', 'left_side_pc', 'tail_base_pc',

        'rel_angle_social',  # Angle of heading relative to other animal (PNAS Feat. 12, Eyrun's Feat. 11) 12? not used
        # 'rel_angle',  # direction difference btw mice (Eyrun's Feat. 11) not used
        'rel_dist_gap',  # distance as defined in PNAS (Feat. 13)
        'rel_dist_scaled',  # distance in units of semiaxis (Feat. 14)
        'rel_dist_centroid',
        'rel_dist_nose',
        'rel_dist_head',
        'rel_dist_body',
        'rel_dist_head_body',
        'rel_dist_centroid_change',  # not used
        'overlap_bboxes',
        'area_ellipse_ratio',
        'angle_between',
        'facing_angle',
        'radial_vel',
        'tangential_vel',

        # inter distances
        'dist_m1nose_m2nose',
        'dist_m1nose_m2right_ear',
        'dist_m1nose_m2left_ear',
        'dist_m1nose_m2neck',
        'dist_m1nose_m2right_side',
        'dist_m1nose_m2left_side',
        'dist_m1nose_m2tail_base',
        'dist_m1right_ear_m2nose',
        'dist_m1right_ear_m2right_ear',
        'dist_m1right_ear_m2left_ear',
        'dist_m1right_ear_m2neck',
        'dist_m1right_ear_m2right_side',
        'dist_m1right_ear_m2left_side',
        'dist_m1right_ear_m2tail_base',
        'dist_m1left_ear_m2nose',
        'dist_m1left_ear_m2right_ear',
        'dist_m1left_ear_m2left_ear',
        'dist_m1left_ear_m2neck',
        'dist_m1left_ear_m2right_side',
        'dist_m1left_ear_m2left_side',
        'dist_m1left_ear_m2tail_base',
        'dist_m1neck_m2nose',
        'dist_m1neck_m2right_ear',
        'dist_m1neck_m2left_ear',
        'dist_m1neck_m2neck',
        'dist_m1neck_m2right_side',
        'dist_m1neck_m2left_side',
        'dist_m1neck_m2tail_base',
        'dist_m1right_side_m2nose',
        'dist_m1right_side_m2right_ear',
        'dist_m1right_side_m2left_ear',
        'dist_m1right_side_m2neck',
        'dist_m1right_side_m2right_side',
        'dist_m1right_side_m2left_side',
        'dist_m1right_side_m2tail_base',
        'dist_m1left_side_m2nose',
        'dist_m1left_side_m2right_ear',
        'dist_m1left_side_m2left_ear',
        'dist_m1left_side_m2neck',
        'dist_m1left_side_m2right_side',
        'dist_m1left_side_m2left_side',
        'dist_m1left_side_m2tail_base',
        'dist_m1tail_base_m2nose',
        'dist_m1tail_base_m2right_ear',
        'dist_m1tail_base_m2left_ear',
        'dist_m1tail_base_m2neck',
        'dist_m1tail_base_m2right_side',
        'dist_m1tail_base_m2left_side',
        'dist_m1tail_base_m2tail_base',

        # intra distance
        'dist_nose_right_ear',
        'dist_nose_left_ear',
        'dist_nose_neck',
        'dist_nose_right_side',
        'dist_nose_left_side',
        'dist_nose_tail_base',
        'dist_right_ear_left_ear',
        'dist_right_ear_neck',
        'dist_right_ear_right_side',
        'dist_right_ear_left_side',
        'dist_right_ear_tail_base',
        'dist_left_ear_neck',
        'dist_left_ear_right_side',
        'dist_left_ear_left_side',
        'dist_left_ear_tail_base',
        'dist_neck_right_side',
        'dist_neck_left_side',
        'dist_neck_tail_base',
        'dist_right_side_left_side',
        'dist_right_side_tail_base',
        'dist_left_side_tail_base',

        # velocity centoird w 2 5 10
        'speed_centroid_w2',
        'speed_centroid_w5',
        'speed_centroid_w10',
        # velocity allpoints w2 w5 w10
        'speed_nose_w2',
        'speed_nose_w5',
        'speed_nose_w10',
        'speed_right_ear_w2',
        'speed_right_ear_w5',
        'speed_right_ear_w10',
        'speed_left_ear_w2',
        'speed_left_ear_w5',
        'speed_left_ear_w10',
        'speed_neck_w2',
        'speed_neck_w5',
        'speed_neck_w10',
        'speed_right_side_w2',
        'speed_right_side_w5',
        'speed_right_side_w10',
        'speed_left_side_w2',
        'speed_left_side_w5',
        'speed_left_side_w10',
        'speed_tail_base_w2',
        'speed_tail_base_w5',
        'speed_tail_base_w10'
    ]

    num_features = len(features)
    keypoints = [f for f in sequence['keypoints']]

    smooth_kernel = np.array([1, 2, 1]) / 4.

    try:
        bar = progressbar.ProgressBar(widgets=
                                      [progressbar.FormatLabel('Feats frame %(value)d'), '/', progressbar.FormatLabel('%(max)d  '), progressbar.Percentage(), ' -- ', ' [', progressbar.Timer(), '] ',
                                       progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=num_frames-1)
        bar.start()
        track = {
            'vid_name': video_name,
            'features': features,
            'data': np.zeros((2, num_frames, num_features)),
            'data_smooth': np.zeros((2, num_frames, num_features)),
            'bbox': np.zeros((2, 4, num_frames)),
            'keypoints': keypoints,
            'fps': fps
        }

        allx, ally = [], []
        ############################################### collecting data from bbox and keypoints
        for f in range(num_frames):
            track['bbox'][:, :, f] = np.asarray(sequence['bbox'][f])
            # track['bbox_front'][:,:,f] = np.asarray(frames_pose_front['bbox'][f])
            keypoints = sequence['keypoints'][f]
            if 'conf_nose' in features:
                track['data'][0][f][14:21] = sequence['scores'][f][0]
                track['data'][1][f][14:21] = sequence['scores'][f][1]
            # mouse 1 data
            xm1 = np.asarray(keypoints[0][0])
            ym1 = np.asarray(keypoints[0][1])
            m1k = np.asarray(keypoints[0]).T
            track['data'][0, f, :14] = np.asarray(keypoints[0]).T.flatten()

            # mouse 2 data
            xm2 = np.asarray(keypoints[1][0])
            ym2 = np.asarray(keypoints[1][1])
            m2k = np.asarray(keypoints[1]).T
            track['data'][1, f, :14] = np.asarray(keypoints[1]).T.flatten()

            allx.append(xm1)
            allx.append(xm2)
            ally.append(ym1)
            ally.append(ym2)

            #################################################################  position features

            # fit ellipse
            cx1, cy1, ra1, rb1, alpha1, _, _, _, _ = fit_ellipse(xm1, ym1)
            ind = features.index('centroid_x')
            track['data'][0, f, ind] = cx1  # 'centroid_x'
            ind = features.index('centroid_y')
            track['data'][0, f, ind] = cy1  # 'centroid_y'
            ind = features.index('phi')
            track['data'][0, f, ind] = alpha1  # 'phi'
            ind = features.index('major_axis_len')
            track['data'][0, f, ind] = ra1 if ra1 > 0. else eps  # 'major_axis_len'
            ind = features.index('minor_axis_len')
            track['data'][0, f, ind] = rb1 if rb1 > 0. else eps  # 'minor_axis_len'
            ind = features.index('axis_ratio')
            track['data'][0, f, ind] = ra1 / rb1 if rb1 > 0. else eps  # 'axis_ratio'
            ind = features.index('area_ellipse')
            track['data'][0, f, ind] = mh.pi * ra1 * rb1 if ra1 * rb1 > 0. else eps  # 'area_ellipse'

            cx2, cy2, ra2, rb2, alpha2, _, _, _, _ = fit_ellipse(xm2, ym2)
            ind = features.index('centroid_x')
            track['data'][1, f, ind] = cx2  # 'centroid_x'
            ind = features.index('centroid_y')
            track['data'][1, f, ind] = cy2  # 'centroid_y'
            ind = features.index('phi')
            track['data'][1, f, ind] = alpha2  # 'phi'
            ind = features.index('major_axis_len')
            track['data'][1, f, ind] = ra2 if ra2 > 0. else eps  # 'major_axis_len'
            ind = features.index('minor_axis_len')
            track['data'][1, f, ind] = rb2 if rb2 > 0. else eps  # 'minor_axis_len'
            ind = features.index('axis_ratio')
            track['data'][1, f, ind] = ra2 / rb2 if rb2 > 0. else eps  # 'axis_ratio'
            ind = features.index('area_ellipse')
            track['data'][1, f, ind] = mh.pi * ra2 * rb2 if ra2 * rb2 > 0. else eps  # 'area_ellipse'

            # overlap bbox
            ind = features.index('overlap_bboxes')
            track['data'][0, f, ind] = bb_intersection_over_union(track['bbox'][0, :, f], track['bbox'][1, :, f], im_w, im_h)
            track['data'][1, f, ind] = bb_intersection_over_union(track['bbox'][1, :, f], track['bbox'][0, :, f], im_w, im_h)

            # compute ori head (neck to nose)
            # orim1_head  =
            ind = features.index('ori_head')
            track['data'][0, f, ind] = get_angle(xm1[3], ym1[3], xm1[0], ym1[0])
            # orim2_head  =
            track['data'][1, f, ind] = get_angle(xm2[3], ym2[3], xm2[0], ym2[0])

            # compute ori body (tail to neck)
            # orim1_body  =
            ind = features.index('ori_body')
            track['data'][0, f, ind] = get_angle(xm1[6], ym1[6], xm1[3], ym1[3])
            # orim2_body  =
            track['data'][1, f, ind] = get_angle(xm2[6], ym2[6], xm2[3], ym2[3])

            # compute angle betwwen head and body
            ind = features.index('angle_head_body_l')
            track['data'][0, f, ind] = interior_angle([xm1[2], ym1[2]], [xm1[3], ym1[3]], [xm1[5], ym1[5]])
            track['data'][1, f, ind] = interior_angle([xm2[2], ym2[2]], [xm2[3], ym2[3]], [xm2[5], ym2[5]])
            ind = features.index('angle_head_body_r')
            track['data'][0, f, ind] = interior_angle([xm1[1], ym1[1]], [xm1[3], ym1[3]], [xm1[4], ym1[4]])
            track['data'][1, f, ind] = interior_angle([xm2[1], ym2[1]], [xm2[3], ym2[3]], [xm2[4], ym2[4]])
            # track['data'][0, f, ind] = get_angle(np.mean(xm1[4:7]), np.mean(ym1[4:7]), np.mean(xm1[:3]), np.mean(ym1[:3]))
            # track['data'][1, f, ind] = get_angle(np.mean(xm2[4:7]), np.mean(ym2[4:7]), np.mean(xm2[:3]), np.mean(ym2[:3]))

            # centroid head
            ind = features.index('centroid_head_x')
            track['data'][0, f, ind] = np.mean(xm1[:3])
            track['data'][1, f, ind] = np.mean(xm2[:3])
            ind = features.index('centroid_head_y')
            track['data'][0, f, ind] = np.mean(ym1[:3])
            track['data'][1, f, ind] = np.mean(ym2[:3])

            # centroid hips
            ind = features.index('centroid_hips_x')
            track['data'][0, f, ind] = np.mean(xm1[4:7])
            track['data'][1, f, ind] = np.mean(xm2[4:7])
            ind = features.index('centroid_hips_y')
            track['data'][0, f, ind] = np.mean(ym1[4:7])
            track['data'][1, f, ind] = np.mean(ym2[4:7])

            # centroid body
            ind = features.index('centroid_body_x')
            track['data'][0, f, ind] = np.mean(xm1[3:7])
            track['data'][1, f, ind] = np.mean(xm2[3:7])
            ind = features.index('centroid_body_y')
            track['data'][0, f, ind] = np.mean(ym1[3:7])
            track['data'][1, f, ind] = np.mean(ym2[3:7])

        for f in range(num_features):
            if f == features.index('phi') or f == features.index('ori_head') or f == features.index('ori_body') \
                    or f == features.index('angle_head_body_l') or f == features.index('angle_head_body_r'):
                # track['data_smooth'][:, :, f] = np.mod(track['data_smooth'][:, :, f] + 2*np.pi,2*np.pi)
                track['data_smooth'][:, :, f] = track['data'][:, :, f]
            else:
                track['data_smooth'][0, :, f] = track['data'][0, :, f]
                track['data_smooth'][0, 1:-1, f] = sig.convolve(track['data_smooth'][0, :, f], smooth_kernel, 'valid')
                track['data_smooth'][1, :, f] = track['data'][1, :, f]
                track['data_smooth'][1, 1:-1, f] = sig.convolve(track['data_smooth'][1, :, f], smooth_kernel, 'valid')
        ############################################### features

        # velocity
        # Metric of speed that accounts for both the head movement and the body movement
        # Also this isn't the speed calculated in the PNAS paper (that's speed_fwd)
        if 'speed' in features:
            ind = features.index('speed')
            chx1 = track['data'][0, :, features.index('centroid_head_x')]
            chy1 = track['data'][0, :, features.index('centroid_head_y')]
            chix1 = track['data'][0, :, features.index('centroid_hips_x')]
            chiy1 = track['data'][0, :, features.index('centroid_hips_y')]
            chx2 = track['data'][1, :, features.index('centroid_head_x')]
            chy2 = track['data'][1, :, features.index('centroid_head_y')]
            chix2 = track['data'][1, :, features.index('centroid_hips_x')]
            chiy2 = track['data'][1, :, features.index('centroid_hips_y')]
            x_diff0 = np.diff(np.vstack((chx1, chix1)))
            y_diff0 = np.diff(np.vstack((chy1, chiy1)))
            x_diff1 = np.diff(np.vstack((chx2, chix2)))
            y_diff1 = np.diff(np.vstack((chy2, chiy2)))
            vel1 = np.linalg.norm([x_diff0, y_diff0], axis=(0, 1))
            vel2 = np.linalg.norm([x_diff1, y_diff1], axis=(0, 1))
            track['data'][0, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm
            track['data'][1, :, ind] = np.hstack((vel2[0], vel2)) * fps / pix_per_cm

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            chix1[1:-1] = sig.convolve(chix1, smooth_kernel, 'valid')
            chiy1[1:-1] = sig.convolve(chiy1, smooth_kernel, 'valid')
            chix2[1:-1] = sig.convolve(chix2, smooth_kernel, 'valid')
            chiy2[1:-1] = sig.convolve(chiy2, smooth_kernel, 'valid')
            x_diff0 = np.diff(np.vstack((chx1, chix1)))
            y_diff0 = np.diff(np.vstack((chy1, chiy1)))
            x_diff1 = np.diff(np.vstack((chx2, chix2)))
            y_diff1 = np.diff(np.vstack((chy2, chiy2)))
            vel1 = np.linalg.norm([x_diff0, y_diff0], axis=(0, 1))
            vel2 = np.linalg.norm([x_diff1, y_diff1], axis=(0, 1))
            track['data_smooth'][0, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm
            track['data_smooth'][1, :, ind] = np.hstack((vel2[0], vel2)) * fps / pix_per_cm

        # forward speed
        if 'speed_fwd' in features:
            ind = features.index('speed_fwd')
            cent_x_ind = features.index('centroid_x')
            cent_y_ind = features.index('centroid_y')
            dir_mot = np.zeros((2, num_frames))
            cx0 = track['data'][0, :, cent_x_ind]
            cy0 = track['data'][0, :, cent_y_ind]
            cx1 = track['data'][1, :, cent_x_ind]
            cy1 = track['data'][1, :, cent_y_ind]
            nskip = 1
            for f in range(num_frames - nskip):
                dir_mot[0, f] = get_angle(cx0[f], cy0[f], cx0[f + nskip], cy0[f + nskip])
                dir_mot[1, f] = get_angle(cx1[f], cy1[f], cx1[f + nskip], cy1[f + nskip])
            for f in range(nskip):
                dir_mot[0, num_frames - nskip + f] = 0
                dir_mot[1, num_frames - nskip + f] = 0
            t_step = 4
            filt = np.ones(t_step)
            x_diff0 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][0, :, cent_x_ind]))), filt, mode='same')
            y_diff0 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][0, :, cent_y_ind]))), filt, mode='same')
            x_diff1 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][1, :, cent_x_ind]))), filt, mode='same')
            y_diff1 = sig.correlate(np.concatenate((np.zeros(1), np.diff(track['data'][1, :, cent_y_ind]))), filt, mode='same')
            ori_ind = features.index('ori_body')
            cos_fac_0 = np.cos(track['data'][0, :, ori_ind] -dir_mot[0, :])
            cos_fac_1 = np.cos(track['data'][1, :, ori_ind] -dir_mot[1, :])
            fs1 = np.multiply(np.linalg.norm([x_diff0, y_diff0], axis=0), cos_fac_0)
            fs2 = np.multiply(np.linalg.norm([x_diff1, y_diff1], axis=0), cos_fac_1)
            track['data'][0, :, ind] = fs1 * fps / pix_per_cm
            track['data'][1, :, ind] = fs2 * fps / pix_per_cm
            fs1[1:-1] = sig.convolve(fs1, smooth_kernel, 'valid')
            fs2[1:-1] = sig.convolve(fs2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = fs1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind] = fs2 * fps / pix_per_cm

        # acceleration
        if 'acceleration' in features:
            ind = features.index('acceleration')
            chx1 = track['data'][0, :, features.index('centroid_head_x')]
            chy1 = track['data'][0, :, features.index('centroid_head_y')]
            chix1 = track['data'][0, :, features.index('centroid_hips_x')]
            chiy1 = track['data'][0, :, features.index('centroid_hips_y')]
            chx2 = track['data'][1, :, features.index('centroid_head_x')]
            chy2 = track['data'][1, :, features.index('centroid_head_y')]
            chix2 = track['data'][1, :, features.index('centroid_hips_x')]
            chiy2 = track['data'][1, :, features.index('centroid_hips_y')]
            x_acc = np.diff(np.vstack((chx1, chix1)), n=2)
            y_acc = np.diff(np.vstack((chy1, chiy1)), n=2)
            acc1 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data'][0, :, ind] = np.hstack((np.hstack((acc1[0], acc1)), acc1[-1])) * fps / (pix_per_cm ** 2)
            x_acc = np.diff(np.vstack((chx2, chix2)), n=2)
            y_acc = np.diff(np.vstack((chy2, chiy2)), n=2)
            acc2 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data'][1, :, ind] = np.hstack((np.hstack((acc2[0], acc2)), acc2[-1])) * fps / (pix_per_cm ** 2)

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            chix1[1:-1] = sig.convolve(chix1, smooth_kernel, 'valid')
            chiy1[1:-1] = sig.convolve(chiy1, smooth_kernel, 'valid')
            chix2[1:-1] = sig.convolve(chix2, smooth_kernel, 'valid')
            chiy2[1:-1] = sig.convolve(chiy2, smooth_kernel, 'valid')
            x_acc = np.diff(np.vstack((chx1, chix1)), n=2)
            y_acc = np.diff(np.vstack((chy1, chiy1)), n=2)
            acc1 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data_smooth'][0, :, ind] = np.hstack((np.hstack((acc1[0], acc1)), acc1[-1])) * fps / (pix_per_cm ** 2)
            x_acc = np.diff(np.vstack((chx2, chix2)), n=2)
            y_acc = np.diff(np.vstack((chy2, chiy2)), n=2)
            acc2 = np.linalg.norm([x_acc, y_acc], axis=(0, 1))
            track['data_smooth'][1, :, ind] = np.hstack((np.hstack((acc2[0], acc2)), acc2[-1])) * fps / (pix_per_cm ** 2)

        # distance between mice's centroids
        if 'rel_dist_centroid' in features:
            ind = features.index('rel_dist_centroid')
            cent_x_ind = features.index('centroid_x')
            cent_y_ind = features.index('centroid_y')
            x_dif = track['data'][0, :, cent_x_ind] - track['data'][1, :, cent_x_ind]
            y_dif = track['data'][0, :, cent_y_ind] - track['data'][1, :, cent_y_ind]
            cent_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = cent_dif / pix_per_cm
            track['data'][1, :, ind] = cent_dif / pix_per_cm
            cent_dif[1:-1] = sig.convolve(cent_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = cent_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = cent_dif / pix_per_cm

        # distance change mice centroids
        if 'rel_dist_centroid_change' in features:
            ind = features.index('rel_dist_centroid_change')
            dist_bet = np.diff(track['data'][0, :, features.index('rel_dist_centroid')] * pix_per_cm)
            rdcc = np.hstack((dist_bet[0], dist_bet))
            track['data'][0, :, ind] = rdcc / pix_per_cm
            track['data'][1, :, ind] = rdcc / pix_per_cm
            rdcc[1:-1] = sig.convolve(rdcc, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = rdcc / pix_per_cm
            track['data_smooth'][1, :, ind] = rdcc / pix_per_cm

        # distance between mice's nose
        if 'rel_dist_nose' in features:
            ind = features.index('rel_dist_nose')
            nose_x_ind = features.index('nose_x')
            nose_y_ind = features.index('nose_y')
            x_dif = track['data'][1, :, nose_x_ind] - track['data'][0, :, nose_x_ind]
            y_dif = track['data'][1, :, nose_y_ind] - track['data'][0, :, nose_y_ind]
            nose_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = nose_dif / pix_per_cm
            track['data'][1, :, ind] = nose_dif / pix_per_cm
            nose_dif[1:-1] = sig.convolve(nose_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = nose_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = nose_dif / pix_per_cm

        # rel dist mice's head
        if 'rel_dist_head' in features:
            ind = features.index('rel_dist_head')  # centroid_body_x
            h_x_ind = features.index('centroid_head_x')
            h_y_ind = features.index('centroid_head_y')
            x_dif = track['data'][1, :, h_x_ind] - track['data'][0, :, h_x_ind]
            y_dif = track['data'][1, :, h_y_ind] - track['data'][0, :, h_y_ind]
            bod_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = bod_dif / pix_per_cm
            track['data'][1, :, ind] = bod_dif / pix_per_cm
            bod_dif[1:-1] = sig.convolve(bod_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = bod_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = bod_dif / pix_per_cm

        # distance between mice's body centroids
        if 'rel_dist_body' in features:
            ind = features.index('rel_dist_body')  # centroid_body_x
            bod_x_ind = features.index('centroid_body_x')
            bod_y_ind = features.index('centroid_body_y')
            x_dif = track['data'][1, :, bod_x_ind] - track['data'][0, :, bod_x_ind]
            y_dif = track['data'][1, :, bod_y_ind] - track['data'][0, :, bod_y_ind]
            bod_dif = np.linalg.norm(np.vstack((x_dif, y_dif)), axis=0)
            track['data'][0, :, ind] = bod_dif / pix_per_cm
            track['data'][1, :, ind] = bod_dif / pix_per_cm
            bod_dif[1:-1] = sig.convolve(bod_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = bod_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = bod_dif / pix_per_cm

        # distance between mice's head body
        if 'rel_dist_head_body' in features:
            ind = features.index('rel_dist_head_body')  # centroid_body_x
            bod_x_ind = features.index('centroid_body_x')
            bod_y_ind = features.index('centroid_body_y')
            h_x_ind = features.index('centroid_head_x')
            h_y_ind = features.index('centroid_head_y')
            x1_dif = track['data'][0, :, h_x_ind] - track['data'][1, :, bod_x_ind]
            y1_dif = track['data'][0, :, h_y_ind] - track['data'][1, :, bod_y_ind]
            x2_dif = track['data'][1, :, h_x_ind] - track['data'][0, :, bod_x_ind]
            y2_dif = track['data'][1, :, h_y_ind] - track['data'][0, :, bod_y_ind]
            hb1_dif = np.linalg.norm(np.vstack((x1_dif, y1_dif)), axis=0)
            hb2_dif = np.linalg.norm(np.vstack((x2_dif, y2_dif)), axis=0)
            track['data'][0, :, ind] = hb1_dif / pix_per_cm
            track['data'][1, :, ind] = hb2_dif / pix_per_cm
            hb1_dif[1:-1] = sig.convolve(hb1_dif, smooth_kernel, 'valid')
            hb2_dif[1:-1] = sig.convolve(hb2_dif, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = hb1_dif / pix_per_cm
            track['data_smooth'][1, :, ind] = hb2_dif / pix_per_cm

        # Angle of heading relative to other animal (PNAS Feat. 12, Eyrun's Feat. 11)
        if 'rel_angle_social' in features:
            ind = features.index('rel_angle_social')
            bod_x_ind = features.index('centroid_x')
            bod_y_ind = features.index('centroid_y')
            ang_ind = features.index('ori_body')
            x_dif = track['data'][1, :, bod_x_ind] - track['data'][0, :, bod_x_ind]  # r_x
            y_dif = track['data'][1, :, bod_y_ind] - track['data'][0, :, bod_y_ind]  # r_y
            theta0 = (np.arctan2(y_dif, x_dif) + 2 * np.pi) % 2 * np.pi
            theta1 = (np.arctan2(-y_dif, -x_dif) + 2 * np.pi) % 2 * np.pi
            ang0 = np.mod(theta0 - track['data'][0, :, ang_ind], 2 * np.pi)
            ang1 = np.mod(theta1 - track['data'][1, :, ang_ind], 2 * np.pi)
            track['data'][0, :, ind] = np.minimum(ang0, 2 * np.pi - ang0)
            track['data'][1, :, ind] = np.minimum(ang1, 2 * np.pi - ang1)
            track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
            track['data_smooth'][0, 1:-1, ind] = np.mod(sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid') + 2 * np.pi, 2 * np.pi)
            track['data_smooth'][1, :, ind] = track['data'][1, :, ind]
            track['data_smooth'][1, 1:-1, ind] = np.mod(sig.convolve(track['data_smooth'][1, :, ind], smooth_kernel, 'valid') + 2 * np.pi, 2 * np.pi)

        # semiaxis length
        M_ind = features.index('major_axis_len')
        m_ind = features.index('minor_axis_len')
        phi_ind = features.index('phi')
        c_M_0 = np.multiply(track['data'][0, :, M_ind] * pix_per_cm, np.sin(track['data'][0, :, phi_ind]))
        c_m_0 = np.multiply(track['data'][0, :, m_ind] * pix_per_cm, np.cos(track['data'][0, :, phi_ind]))
        c_M_1 = np.multiply(track['data'][1, :, M_ind] * pix_per_cm, np.sin(track['data'][1, :, phi_ind]))
        c_m_1 = np.multiply(track['data'][1, :, m_ind] * pix_per_cm, np.cos(track['data'][1, :, phi_ind]))

        # distance as defined in PNAS (Feat. 13)
        if 'rel_dist_gap' in features:
            ind = features.index('rel_dist_gap')
            rel_dist_ind = features.index('rel_dist_body')
            comb_norm = np.linalg.norm(np.vstack((c_M_0, c_m_0)), axis=0) + np.linalg.norm(np.vstack((c_M_1, c_m_1)), axis=0)
            d1 = (track['data'][0, :, rel_dist_ind] * pix_per_cm - comb_norm)
            d2 = (track['data'][1, :, rel_dist_ind] * pix_per_cm - comb_norm)
            track['data'][0, :, ind] = d1 / pix_per_cm
            track['data'][1, :, ind] = d2 / pix_per_cm
            d1[1:-1] = sig.convolve(d1, smooth_kernel, 'valid')
            d2[1:-1] = sig.convolve(d2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = d1 / pix_per_cm
            track['data_smooth'][1, :, ind] = d2 / pix_per_cm

        # distance in units of semiaxis (Feat. 14)
        if 'rel_dist_scaled' in features:
            ind = features.index('rel_dist_scaled')
            rel_dist_gap_ind = features.index('rel_dist_gap')
            d1 = np.divide(track['data'][0, :, rel_dist_gap_ind] * pix_per_cm, np.linalg.norm(np.vstack((c_M_0, c_m_0)), axis=0))
            d2 = np.divide(track['data'][1, :, rel_dist_gap_ind] * pix_per_cm, np.linalg.norm(np.vstack((c_M_1, c_m_1)), axis=0))
            track['data'][0, :, ind] = d1 / pix_per_cm
            track['data'][1, :, ind] = d2 / pix_per_cm
            d1[1:-1] = sig.convolve(d1, smooth_kernel, 'valid')
            d2[1:-1] = sig.convolve(d2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = d1 / pix_per_cm
            track['data_smooth'][1, :, ind] = d2 / pix_per_cm

        # Ratio of mouse 1's ellipse area to mouse 0's ellipse area (Feat. 15)
        if 'area_ellipse_ratio' in features:
            ind = features.index('area_ellipse_ratio')
            area_ind = features.index('area_ellipse')
            track['data'][0, :, ind] = np.divide(track['data'][0, :, area_ind], track['data'][1, :, area_ind], dtype=float)
            track['data'][1, :, ind] = np.divide(track['data'][1, :, area_ind], track['data'][0, :, area_ind], dtype=float)
            track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
            track['data_smooth'][0, 1:-1, ind] = sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid')
            track['data_smooth'][1, :, ind] = track['data'][1, :, ind]
            track['data_smooth'][1, 1:-1, ind] = sig.convolve(track['data_smooth'][1, :, ind], smooth_kernel, 'valid')

        # angle between & facing angle
        if 'angle_between' in features:
            ind = features.index('angle_between')
            ori_ind = features.index(('phi'))
            vec_rot = np.vstack((np.cos(track['data'][0, :, ori_ind]), -np.sin(track['data'][0, :, ori_ind])))
            vec_rot2 = np.vstack((np.cos(track['data'][1, :, ori_ind]), -np.sin(track['data'][1, :, ori_ind])))
            angle_btw = np.arccos((vec_rot * vec_rot2).sum(axis=0))
            track['data'][0, :, ind] = angle_btw
            track['data'][1, :, ind] = angle_btw
            angle_btw[1:-1] = sig.convolve(angle_btw, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = angle_btw
            track['data_smooth'][1, :, ind] = angle_btw

        if 'facing_angle' in features:
            ori_ind = features.index(('phi'))
            vec_rot = np.vstack((np.cos(track['data'][0, :, ori_ind]), -np.sin(track['data'][0, :, ori_ind])))
            vec_rot2 = np.vstack((np.cos(track['data'][1, :, ori_ind]), -np.sin(track['data'][1, :, ori_ind])))
            c1 = np.vstack((track['data'][0, :, features.index('centroid_x')], track['data'][0, :, features.index('centroid_y')]))
            c2 = np.vstack((track['data'][1, :, features.index('centroid_x')], track['data'][1, :, features.index('centroid_y')]))
            vec_btw = c2 - c1
            norm_btw = np.linalg.norm(np.vstack((vec_btw[0, :], vec_btw[1, :])), axis=0)
            vec_btw = vec_btw / np.repeat([norm_btw], 2, axis=0)
            fa1 = np.arccos((vec_rot * vec_btw).sum(axis=0))
            fa2 = np.arccos((vec_rot2 * -vec_btw).sum(axis=0))
            ind = features.index('facing_angle')
            track['data'][0, :, ind] = fa1
            track['data'][1, :, ind] = fa2
            fa1[1:-1] = sig.convolve(fa1, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = fa1
            fa2[1:-1] = sig.convolve(fa2, smooth_kernel, 'valid')
            track['data_smooth'][1, :, ind] = fa2

        # distance points mouse1 to mouse 2, m1 and m1 , m2 and m2
        if 'dist_m1nose_m2nose' in features or 'dist_nose_right_ear' in features:
            if 'dist_m1nose_m2nose' in features:
                m1m2d = np.zeros((num_frames, num_points ** 2))
                idx_triu = np.triu_indices(num_points, 1)
                n_idx = len(idx_triu[0])
            if 'dist_nose_right_ear' in features:
                m1m1d = np.zeros((num_frames, n_idx))
                m2m2d = np.zeros((num_frames, n_idx))

            for f in range(num_frames):
                m1 = track['data'][0, f, :(num_points * 2)]
                m2 = track['data'][1, f, :(num_points * 2)]
                m1_x = m1[::2]
                m1_y = m1[1::2]
                m2_x = m2[::2]
                m2_y = m2[1::2]
                m1_p = np.vstack((m1_x, m1_y)).T
                m2_p = np.vstack((m2_x, m2_y)).T

                if 'dist_m1nose_m2nose' in features:
                    m1m2d[f, :] = dist.cdist(m1_p, m2_p).flatten()
                if 'dist_nose_right_ear' in features:
                    m1m1d[f, :] = dist.cdist(m1_p, m1_p)[idx_triu]
                    m2m2d[f, :] = dist.cdist(m2_p, m2_p)[idx_triu]

            if 'dist_m1nose_m2nose' in features:
                ind_m1m2d = features.index('dist_m1nose_m2nose')
                track['data'][0, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                track['data'][1, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                for d in range(m1m2d.shape[1]): m1m2d[1:-1, d] = sig.convolve(m1m2d[:, d], smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
                track['data_smooth'][1, :, ind_m1m2d:ind_m1m2d + num_points ** 2] = m1m2d / pix_per_cm
            if 'dist_nose_right_ear' in features:
                ind_md = features.index('dist_nose_right_ear')
                track['data'][0, :, ind_md:ind_md + n_idx] = m1m1d / pix_per_cm
                track['data'][1, :, ind_md:ind_md + n_idx] = m2m2d / pix_per_cm
                for d in range(m1m1d.shape[1]): m1m1d[1:-1, d] = sig.convolve(m1m1d[:, d], smooth_kernel, 'valid')
                for d in range(m2m2d.shape[1]): m2m2d[1:-1, d] = sig.convolve(m2m2d[:, d], smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind_md:ind_md + n_idx] = m1m1d / pix_per_cm
                track['data_smooth'][1, :, ind_md:ind_md + n_idx] = m2m2d / pix_per_cm

        # velocity centroid windows -2 -5 -10
        if 'speed_centroid' in features or 'speed_centroid_x' in features or 'speed_centroid_y' in features:
            chx1 = track['data'][0, :, features.index('centroid_x')]
            chy1 = track['data'][0, :, features.index('centroid_y')]
            chx2 = track['data'][1, :, features.index('centroid_x')]
            chy2 = track['data'][1, :, features.index('centroid_y')]
            x_diff0 = np.diff(chx1)
            y_diff0 = np.diff(chy1)
            x_diff1 = np.diff(chx2)
            y_diff1 = np.diff(chy2)
            if 'speed_centroid' in features:
                ind = features.index('speed_centroid')
                vel0 = np.linalg.norm([x_diff0, y_diff0], axis=0)
                track['data'][0, :, ind] = np.hstack((vel0[0], vel0)) * fps / pix_per_cm
                vel1 = np.linalg.norm([x_diff1, y_diff1], axis=0)
                track['data'][1, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm

            if 'speed_centroid_x' in features:
                indx = features.index('speed_centroid_x')
                track['data'][0, :, indx] = np.hstack((x_diff0[0], x_diff0))
                track['data'][1, :, indx] = np.hstack((x_diff1[0], x_diff1))

            if 'speed_centroid_y' in features:
                indy = features.index('speed_centroid_y')
                track['data'][0, :, indy] = np.hstack((y_diff0[0], y_diff0))
                track['data'][1, :, indy] = np.hstack((y_diff1[0], y_diff1))

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            x_diff0 = np.diff(chx1)
            y_diff0 = np.diff(chy1)
            x_diff1 = np.diff(chx2)
            y_diff1 = np.diff(chy2)

            if 'speed_centroid_x' in features:
                track['data_smooth'][0, :, indx] = np.hstack((x_diff0[0], x_diff0))
                track['data_smooth'][1, :, indx] = np.hstack((x_diff1[0], x_diff1))
            if 'speed_centroid_y' in features:
                track['data_smooth'][0, :, indy] = np.hstack((y_diff0[0], y_diff0))
                track['data_smooth'][1, :, indy] = np.hstack((y_diff1[0], y_diff1))
            if 'speed_centroid' in features:
                vel0 = np.linalg.norm([x_diff0, y_diff0], axis=0)
                track['data_smooth'][0, :, ind] = np.hstack((vel0[0], vel0)) * fps / pix_per_cm
                vel1 = np.linalg.norm([x_diff1, y_diff1], axis=0)
                track['data_smooth'][1, :, ind] = np.hstack((vel1[0], vel1)) * fps / pix_per_cm

        # acceleration centroid
        if 'acceleration_centroid' in features or 'acceleration_centroid_x' in features or 'acceleration_centroid_y' in features:
            chx1 = track['data'][0, :, features.index('centroid_x')]
            chy1 = track['data'][0, :, features.index('centroid_y')]
            chx2 = track['data'][1, :, features.index('centroid_x')]
            chy2 = track['data'][1, :, features.index('centroid_y')]
            x_acc0 = np.diff(chx1, n=2)
            y_acc0 = np.diff(chy1, n=2)
            x_acc1 = np.diff(chx2, n=2)
            y_acc1 = np.diff(chy2, n=2)

            if 'acceleration_centroid' in features:
                ind = features.index('acceleration_centroid')
                acc0 = np.linalg.norm([x_acc0, y_acc0], axis=0)
                acc1 = np.linalg.norm([x_acc1, y_acc1], axis=0)
                track['data'][0, :, ind] = np.hstack((acc0[0], acc0[0], acc0)) * fps / (pix_per_cm ** 2)
                track['data'][1, :, ind] = np.hstack((acc1[0], acc1[0], acc1)) * fps / (pix_per_cm ** 2)

            if 'acceleration_centroid_x' in features:
                indx = features.index('acceleration_centroid_x')
                track['data'][0, :, indx] = np.hstack((x_acc0[0], x_acc0[0], x_acc0))
                track['data'][1, :, indx] = np.hstack((x_acc1[0], x_acc1[0], x_acc1))

            if 'acceleration_centroid_y' in features:
                indy = features.index('acceleration_centroid_y')
                track['data'][0, :, indy] = np.hstack((y_acc0[0], y_acc0[0], y_acc0))
                track['data'][1, :, indy] = np.hstack((y_acc1[0], y_acc1[0], y_acc1))

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            x_acc0 = np.diff(chx1, n=2)
            y_acc0 = np.diff(chy1, n=2)
            x_acc1 = np.diff(chx2, n=2)
            y_acc1 = np.diff(chy2, n=2)

            if 'acceleration_centroid' in features:
                acc0 = np.linalg.norm([x_acc0, y_acc0], axis=0)
                acc1 = np.linalg.norm([x_acc1, y_acc1], axis=0)
                track['data_smooth'][0, :, ind] = np.hstack((acc0[0], acc0[0], acc0)) * fps / (pix_per_cm ** 2)
                track['data_smooth'][1, :, ind] = np.hstack((acc1[0], acc1[0], acc1)) * fps / (pix_per_cm ** 2)

            if 'acceleration_centroid_x' in features:
                track['data_smooth'][0, :, indx] = np.hstack((x_acc0[0], x_acc0[0], x_acc0))
                track['data_smooth'][1, :, indx] = np.hstack((x_acc1[0], x_acc1[0], x_acc1))

            if 'acceleration_centroid_y' in features:
                track['data_smooth'][0, :, indy] = np.hstack((y_acc0[0], y_acc0[0], y_acc0))
                track['data_smooth'][1, :, indy] = np.hstack((y_acc1[0], y_acc1[0], y_acc1))

        if 'speed_centroid_w2' in features:
            ind_w2 = features.index('speed_centroid_w2')
            ind_w5 = features.index('speed_centroid_w5')
            ind_w10 = features.index('speed_centroid_w10')
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(2, num_frames - 2):
                x_diff0 = chx1[f] - chx1[f - 2]
                y_diff0 = chy1[f] - chy1[f - 2]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 2]
                y_diff1 = chy2[f] - chy2[f - 2]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w2] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w2] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(5, num_frames - 5):
                x_diff0 = chx1[f] - chx1[f - 5]
                y_diff0 = chy1[f] - chy1[f - 5]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 5]
                y_diff1 = chy2[f] - chy2[f - 5]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w5] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w5] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(10, num_frames - 10):
                x_diff0 = chx1[f] - chx1[f - 10]
                y_diff0 = chy1[f] - chy1[f - 10]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 10]
                y_diff1 = chy2[f] - chy2[f - 10]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data'][0, :, ind_w10] = vel1 * fps / pix_per_cm
            track['data'][1, :, ind_w10] = vel2 * fps / pix_per_cm

            chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
            chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
            chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
            chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(2, num_frames - 2):
                x_diff0 = chx1[f] - chx1[f - 2]
                y_diff0 = chy1[f] - chy1[f - 2]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 2]
                y_diff1 = chy2[f] - chy2[f - 2]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w2] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w2] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(5, num_frames - 5):
                x_diff0 = chx1[f] - chx1[f - 5]
                y_diff0 = chy1[f] - chy1[f - 5]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 5]
                y_diff1 = chy2[f] - chy2[f - 5]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w5] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w5] = vel2 * fps / pix_per_cm
            vel1 = np.zeros(num_frames)
            vel2 = np.zeros(num_frames)
            for f in range(10, num_frames - 10):
                x_diff0 = chx1[f] - chx1[f - 10]
                y_diff0 = chy1[f] - chy1[f - 10]
                vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                x_diff1 = chx2[f] - chx2[f - 10]
                y_diff1 = chy2[f] - chy2[f - 10]
                vel2[f] = np.linalg.norm([x_diff1, y_diff1])
            track['data_smooth'][0, :, ind_w10] = vel1 * fps / pix_per_cm
            track['data_smooth'][1, :, ind_w10] = vel2 * fps / pix_per_cm

        # velocity points windows -2 -5 -10
        if 'speed_nose_w2' in features:
            for p in range(0, num_points * 2, 2):
                chx1 = track['data'][0, :, p]
                chy1 = track['data'][0, :, p + 1]
                chx2 = track['data'][1, :, p]
                chy2 = track['data'][1, :, p + 1]
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                ind_w2 = features.index('speed_' + parts[int(p / 2)] + '_w2')
                ind_w5 = features.index('speed_' + parts[int(p / 2)] + '_w5')
                ind_w10 = features.index('speed_' + parts[int(p / 2)] + '_w10')
                for f in range(2, num_frames - 2):
                    x_diff0 = chx1[f] - chx1[f - 2]
                    y_diff0 = chy1[f] - chy1[f - 2]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 2]
                    y_diff1 = chy2[f] - chy2[f - 2]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w2] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w2] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(5, num_frames - 5):
                    x_diff0 = chx1[f] - chx1[f - 5]
                    y_diff0 = chy1[f] - chy1[f - 5]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 5]
                    y_diff1 = chy2[f] - chy2[f - 5]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w5] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w5] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(10, num_frames - 10):
                    x_diff0 = chx1[f] - chx1[f - 10]
                    y_diff0 = chy1[f] - chy1[f - 10]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 10]
                    y_diff1 = chy2[f] - chy2[f - 10]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data'][0, :, ind_w10] = vel1 * fps / pix_per_cm
                track['data'][1, :, ind_w10] = vel2 * fps / pix_per_cm

                chx1[1:-1] = sig.convolve(chx1, smooth_kernel, 'valid')
                chy1[1:-1] = sig.convolve(chy1, smooth_kernel, 'valid')
                chx2[1:-1] = sig.convolve(chx2, smooth_kernel, 'valid')
                chy2[1:-1] = sig.convolve(chy2, smooth_kernel, 'valid')
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(2, num_frames - 2):
                    x_diff0 = chx1[f] - chx1[f - 2]
                    y_diff0 = chy1[f] - chy1[f - 2]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 2]
                    y_diff1 = chy2[f] - chy2[f - 2]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w2] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w2] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(5, num_frames - 5):
                    x_diff0 = chx1[f] - chx1[f - 5]
                    y_diff0 = chy1[f] - chy1[f - 5]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 5]
                    y_diff1 = chy2[f] - chy2[f - 5]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w5] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w5] = vel2 * fps / pix_per_cm
                vel1 = np.zeros(num_frames)
                vel2 = np.zeros(num_frames)
                for f in range(10, num_frames - 10):
                    x_diff0 = chx1[f] - chx1[f - 10]
                    y_diff0 = chy1[f] - chy1[f - 10]
                    vel1[f] = np.linalg.norm([x_diff0, y_diff0])
                    x_diff1 = chx2[f] - chx2[f - 10]
                    y_diff1 = chy2[f] - chy2[f - 10]
                    vel2[f] = np.linalg.norm([x_diff1, y_diff1])
                track['data_smooth'][0, :, ind_w10] = vel1 * fps / pix_per_cm
                track['data_smooth'][1, :, ind_w10] = vel2 * fps / pix_per_cm

        # distance to edge
        if 'dist_edge_x' in features:
            ind = features.index('dist_edge_x')
            allx = np.concatenate(allx).ravel()
            ally = np.concatenate(ally).ravel()
            allx = allx[allx > 0]
            ally = ally[ally > 0]
            minx = np.percentile(allx, 1)
            maxx = np.percentile(allx, 99)
            miny = np.percentile(ally, 1)
            maxy = np.percentile(ally, 99)

            cx1 = track['data'][0, :, features.index('centroid_x')]
            cx2 = track['data'][1, :, features.index('centroid_x')]
            cy1 = track['data'][0, :, features.index('centroid_y')]
            cy2 = track['data'][1, :, features.index('centroid_y')]
            distsx1 = np.amin(np.stack((np.maximum(0, cx1 - minx), np.maximum(0, maxx - cx1)), axis=-1), axis=1)
            distsx2 = np.amin(np.stack((np.maximum(0, cx2 - minx), np.maximum(0, maxx - cx2)), axis=-1), axis=1)
            distsy1 = np.amin(np.stack((np.maximum(0, cy1 - miny), np.maximum(0, maxy - cy1)), axis=-1), axis=1)
            distsy2 = np.amin(np.stack((np.maximum(0, cy2 - miny), np.maximum(0, maxy - cy2)), axis=-1), axis=1)
            dists1 = np.amin(np.stack((distsx1, distsy1), axis=-1), axis=1)
            dists2 = np.amin(np.stack((distsx2, distsy2), axis=-1), axis=1)
            track['data'][0, :, ind] = distsx1
            track['data'][0, :, ind + 1] = distsy1
            track['data'][0, :, ind + 2] = dists1
            track['data'][1, :, ind] = distsx2
            track['data'][1, :, ind + 1] = distsy2
            track['data'][1, :, ind + 2] = dists2
            # distsx1[1:-1]= sig.convolve(distsx1, smooth_kernel, 'valid')
            # distsx2[1:-1]= sig.convolve(distsx2, smooth_kernel, 'valid')
            # distsy1[1:-1]= sig.convolve(distsy1, smooth_kernel, 'valid')
            # distsy2[1:-1]= sig.convolve(distsy2, smooth_kernel, 'valid')
            # dists1[1:-1]= sig.convolve(dists1, smooth_kernel, 'valid')
            # dists2[1:-1]= sig.convolve(dists2, smooth_kernel, 'valid')
            track['data_smooth'][0, :, ind] = distsx1
            track['data_smooth'][0, :, ind + 1] = distsy1
            track['data_smooth'][0, :, ind + 2] = dists1
            track['data_smooth'][1, :, ind] = distsx2
            track['data_smooth'][1, :, ind + 1] = distsy2
            track['data_smooth'][1, :, ind + 2] = dists2

        # radial vel and tangetial vel
        if 'radial_vel' in features or ' tangential_vel' in features:
            cx1 = track['data'][0, :, features.index('centroid_x')]
            cx2 = track['data'][1, :, features.index('centroid_x')]
            cy1 = track['data'][0, :, features.index('centroid_y')]
            cy2 = track['data'][1, :, features.index('centroid_y')]
            ddx1 = cx2 - cx1
            ddx2 = cx1 - cx2
            ddy1 = cy2 - cy1
            ddy2 = cy1 - cy2
            len1 = np.sqrt(ddx1 ** 2 + ddy1 ** 2)
            len1[len1 == 0] = eps
            len2 = np.sqrt(ddx2 ** 2 + ddy2 ** 2)
            len2[len2 == 0] = eps
            ddx1 = ddx1 / len1
            ddx2 = ddx2 / len2
            ddy1 = ddy1 / len1
            ddy2 = ddy2 / len2
            dx1 = np.diff(cx1)
            dx2 = np.diff(cx2)
            dy1 = np.diff(cy1)
            dy2 = np.diff(cy2)

            if 'radial_vel' in features:
                ind = features.index('radial_vel')
                radial_vel1 = dx1 * ddx1[1:] + dy1 * ddy1[1:]
                radial_vel2 = dx2 * ddx2[1:] + dy2 * ddy2[1:]
                track['data'][0, :, ind] = np.hstack((radial_vel1[0], radial_vel1))
                track['data'][1, :, ind] = np.hstack((radial_vel2[0], radial_vel2))
                radial_vel1[1:-1] = sig.convolve(radial_vel1, smooth_kernel, 'valid')
                radial_vel2[1:-1] = sig.convolve(radial_vel2, smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind] = np.hstack((radial_vel1[0], radial_vel1))
                track['data_smooth'][1, :, ind] = np.hstack((radial_vel2[0], radial_vel2))

            # tangential vel
            if 'tangential_vel' in features:
                ind = features.index('tangential_vel')
                orth_ddx1 = -ddy1
                orth_ddx2 = -ddy2
                orth_ddy1 = -ddx1
                orth_ddy2 = -ddx2
                tang_vel1 = np.abs(dx1 * orth_ddx1[1:] + dy1 * orth_ddy1[1:])
                tang_vel2 = np.abs(dx2 * orth_ddx2[1:] + dy2 * orth_ddy2[1:])
                track['data'][0, :, ind] = np.hstack((tang_vel1[0], tang_vel1))
                track['data'][1, :, ind] = np.hstack((tang_vel2[0], tang_vel2))
                tang_vel1[1:-1] = sig.convolve(tang_vel1, smooth_kernel, 'valid')
                tang_vel2[1:-1] = sig.convolve(tang_vel2, smooth_kernel, 'valid')
                track['data_smooth'][0, :, ind] = np.hstack((tang_vel1[0], tang_vel1))
                track['data_smooth'][1, :, ind] = np.hstack((tang_vel2[0], tang_vel2))


                ################################ pixel based features

        # pixel based features
        if 'pixel_change' in features or 'pixel_change_ubbox_mice' in features or 'pc_nose' in features :

            if 'pixel_change' in features:
                ind = features.index('pixel_change')
            if 'pixel_change_ubbox_mice' in features:
                ind1 = features.index('pixel_change_ubbox_mice')
            if 'nose_pc' in features:
                indn = features.index('nose_pc')

            for f in range(1, num_frames):
                bar.update(f)

                if progress_bar_sig:
                    if f <= 1:
                        progress_bar_sig.emit(f,num_frames-1)
                    progress_bar_sig.emit(f,0)

                if f == 1:
                    frame1 = reader.getFrame(f-1)
                    frame1 = frame1.astype(np.float32)
                else:
                    frame1 = frame2

                frame2 = reader.getFrame(f)
                frame2 = frame2.astype(np.float32)

                if 'pixel_change' in features:
                    track['data'][0, f, ind] = (np.sum((frame2 - frame1) ** 2)) / float((np.sum((frame1) ** 2)))
                    track['data'][1, f, ind] = track['data'][0, f, ind]

                if 'pixel_change_ubbox_mice' in features:
                    ind_bbo = features.index('overlap_bboxes')
                    f1_bb = track['data'][0, f - 1, ind_bbo]
                    f2_bb = track['data'][0, f, ind_bbo]

                    if f1_bb > 0. or f2_bb > 0.:
                        bbox_f1_m1 = track['bbox'][0, :, f - 1]
                        xmin11, xmax11 = bbox_f1_m1[[0, 2]] * im_w
                        ymin11, ymax11 = bbox_f1_m1[[1, 3]] * im_h
                        bbox_f1_m2 = track['bbox'][1, :, f - 1]
                        xmin12, xmax12 = bbox_f1_m2[[0, 2]] * im_w
                        ymin12, ymax12 = bbox_f1_m2[[1, 3]] * im_h

                        bbox_f2_m1 = track['bbox'][0, :, f]
                        xmin21, xmax21 = bbox_f2_m1[[0, 2]] * im_w
                        ymin21, ymax21 = bbox_f2_m1[[1, 3]] * im_h
                        bbox_f2_m2 = track['bbox'][1, :, f]
                        xmin22, xmax22 = bbox_f2_m2[[0, 2]] * im_w
                        ymin22, ymax22 = bbox_f2_m2[[1, 3]] * im_h

                        # xA1 = max(xmin11, xmin12);  yA1 = max(ymin11, ymin12)
                        # xB1 = min(xmax11, xmax12);  yB1 = min(ymax11, ymax12)
                        #
                        # xA2 = max(xmin21, xmin22);  yA2 = max(ymin21, ymin22)
                        # xB2 = min(xmax21, xmax22);  yB2 = min(ymax21, ymax22)

                        # if (xB1 > xA1 and yB1 > yA1) or  (xB2 > xA2 and yB2 > yA2): # there is intersection
                        tmp1 = frame1[int(min(ymin11, ymin12)):int(max(ymax11, ymax12)), int(min(xmin11, xmin12)):int(max(xmax11, xmax12))]
                        tmp2 = frame2[int(min(ymin21, ymin22)):int(max(ymax21, ymax22)), int(min(xmin21, xmin22)):int(max(xmax21, xmax22))]

                        if np.prod(tmp1.shape) > np.prod(tmp2.shape):
                            tmp2 = cv2.resize(tmp2, (tmp1.shape[1], tmp1.shape[0]))
                        else:
                            tmp1 = cv2.resize(tmp1, (tmp2.shape[1], tmp2.shape[0]))

                        track['data'][0, f, ind1] = (np.sum((tmp2 - tmp1) ** 2)) / float((np.sum((tmp1) ** 2)))
                        track['data'][1, f, ind1] = (np.sum((tmp2 - tmp1) ** 2)) / float((np.sum((tmp1) ** 2)))

                if 'nose_pc' in features:
                    for s in range(2):
                        coords = track['data'][s, f - 1, :14]
                        coords2 = track['data'][s, f, :14]

                        xm1 = coords[::2]
                        ym1 = coords[1::2]

                        xm2 = coords2[::2]
                        ym2 = coords2[1::2]
                        ps = 10
                        for p in range(num_points):
                            miny1 = int(ym1[p] - ps)
                            minx1 = int(xm1[p] - ps)
                            maxy1 = int(ym1[p] + ps)
                            maxx1 = int(xm1[p] + ps)
                            while miny1 < 0: miny1 += 1
                            # while maxy1 < 0: maxy1 += 1
                            while minx1 < 0: minx1 += 1
                            # while maxx1 < 0: maxx1 += 1
                            while maxy1 > im_h: maxy1 -= 1
                            # while miny1 > im_h -1 : miny1 -= 1
                            while maxx1 > im_w: maxx1 -= 1
                            # while minx1 > im_w -1: minx1 -= 1

                            miny2 = int(ym2[p] - ps)
                            minx2 = int(xm2[p] - ps)
                            maxy2 = int(ym2[p] + ps)
                            maxx2 = int(xm2[p] + ps)
                            while miny2 < 0: miny2 += 1
                            # while maxy2 < 0: maxy2 += 1
                            while minx2 < 0: minx2 += 1
                            # while maxx2 < 0: maxx2 += 1
                            while maxy2 > im_h: maxy2 -= 1
                            # while miny2 > im_h-1: miny2 -= 1
                            while maxx2 > im_w: maxx2 -= 1
                            # while minx2 > im_w-1: minx2 -= 1

                            f1x = range(minx1, maxx1) if minx1 < maxx1 else range(maxx1, minx1)
                            f1y = range(miny1, maxy1) if miny1 < maxy1 else range(maxy1, miny1)
                            f2x = range(minx2, maxx2) if minx2 < maxx2 else range(maxx2, minx2)
                            f2y = range(miny2, maxy2) if miny2 < maxy2 else range(maxy2, miny2)
                            x1, y1 = np.ix_(f1x, f1y)
                            x2, y2 = np.ix_(f2x, f2y)
                            patch1 = frame1[y1, x1]
                            patch2 = frame2[y2, x2]
                            max_h = max(patch1.shape[0], patch2.shape[0])
                            max_w = max(patch1.shape[1], patch2.shape[1])
                            patch2 = cv2.resize(patch2, (max_w, max_h))
                            patch1 = cv2.resize(patch1, (max_w, max_h))
                            track['data'][s, f, indn + p] = (np.sum((patch2 - patch1) ** 2)) / float((np.sum((patch1) ** 2)))
                            # else:
                            #     track['data'][s, f, ind + p]=0


            if 'pixel_change' in features:
                track['data_smooth'][0, :, ind] = track['data'][0, :, ind]
                track['data_smooth'][0, 1:-1, ind] = sig.convolve(track['data_smooth'][0, :, ind], smooth_kernel, 'valid')
                track['data_smooth'][1, :, ind] = track['data_smooth'][0, :, ind]

            if 'pixel_change_ubbox_mice' in features:
                track['data_smooth'][0, :, ind1] = track['data'][0, :, ind1]
                track['data_smooth'][0, 1:-1, ind1] = sig.convolve(track['data_smooth'][0, :, ind1], smooth_kernel, 'valid')
                track['data_smooth'][1, :, ind1] = track['data'][1, :, ind1]
                track['data_smooth'][1, 1:-1, ind1] = sig.convolve(track['data_smooth'][1, :, ind1], smooth_kernel, 'valid')

            if 'nose_pc' in features:
                for s in range(2):
                    for p in range(num_points):
                        track['data_smooth'][s, :, indn + p] = track['data'][s, :, indn + p]
                        track['data_smooth'][s, 1:-1, indn + p] = sig.convolve(track['data_smooth'][s, :, indn + p], smooth_kernel, 'valid')

        del track['data']
        bar.finish()

        reader.close()
        return track
    except Exception as e:
        import linecache
        print("Error when extracting features (extract_features_top):")
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
        print(e)
        reader.close()
        return []


def extract_features(project, progress_bar_sig=''):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    for key in ['train', 'test', 'val']:
        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_data.json')) as f:
            data = json.load(f)

        feats = {'feature_names': [], 'sequences': {cfg['project_name']: {}}}
        keylist = list(data['sequences'][cfg['project_name']].keys())
        for i, k in enumerate(keylist):
            print('%s (%i/%i): %s' % (key, i+1, len(keylist), k))
            feat_dict = extract_features_top(data['sequences'][cfg['project_name']][k], cfg, progress_bar_sig=progress_bar_sig)
            if feat_dict == []:
                print('skipping for no feats, something went wrong')
            else:
                feats['feature_names'] = feat_dict['features']
                feats['vocabulary'] = data['vocabulary']
                feats['sequences'][cfg['project_name']][k] = {'features': feat_dict['data_smooth'].tolist(),
                                                              'annotations': data['sequences'][cfg['project_name']][k]['annotations']}

        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_features.json'), 'w') as f:
            json.dump(feats, f)