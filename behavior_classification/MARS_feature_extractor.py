from __future__ import print_function,division
import sys, os
import copy
import warnings
import json, yaml
import numpy as np
warnings.filterwarnings('ignore')
from behavior_classification.MARS_feature_machinery import *
import pdb

flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


def load_pose(pose_fullpath):
    try:
        with open(pose_fullpath, 'r') as fp:
            pose = json.load(fp)
        return pose
    except Exception as e:
        raise e


def generate_valid_feature_list(cfg):

    num_mice = len(cfg['animal_names']) * cfg['num_obj']
    mice = ['m'+str(i) for i in range(num_mice)]
    # TODO: multi-camera support
    # TODO: replace hard-coded feature names with keypoints from the project config file.
    cameras = {'top':   ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base'],
               'front': ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base',
                         'left_front_paw', 'right_front_paw', 'left_rear_paw', 'right_rear_paw']}
    inferred_parts = ['centroid', 'centroid_head', 'centroid_body']

    feats = {}
    for cam in cameras:
        pairmice = copy.deepcopy(mice)
        feats[cam] = {}
        for mouse in mice:
            feats[cam][mouse] = {}
            feats[cam][mouse]['absolute_orientation'] = ['phi', 'ori_head', 'ori_body']
            feats[cam][mouse]['joint_angle'] = ['angle_head_body_l', 'angle_head_body_r']
            feats[cam][mouse]['fit_ellipse'] = ['major_axis_len', 'minor_axis_len', 'axis_ratio', 'area_ellipse']
            feats[cam][mouse]['distance_to_walls'] = ['dist_edge_x', 'dist_edge_y', 'dist_edge']
            feats[cam][mouse]['speed'] = ['speed', 'speed_centroid', 'speed_fwd']
            feats[cam][mouse]['acceleration'] = ['acceleration_head', 'acceleration_body', 'acceleration_centroid']

        pairmice.remove(mouse)
        for mouse2 in pairmice:
            feats[cam][mouse+mouse2] = {}
            feats[cam][mouse+mouse2]['social_angle'] = ['angle_between', 'facing_angle', 'angle_social']
            feats[cam][mouse+mouse2]['social_speed'] = ['radial_vel', 'tangential_vel']
            feats[cam][mouse+mouse2]['relative_size'] = ['area_ellipse_ratio']
            feats[cam][mouse+mouse2]['social_distance'] = ['dist_centroid', 'dist_nose', 'dist_head', 'dist_body',
                                                   'dist_head_body', 'dist_gap', 'dist_scaled', 'overlap_bboxes']

    for cam, parts in zip(cameras.keys(), [cameras[i] for i in cameras.keys()]):
        for mouse in mice:
            feats[cam][mouse]['raw_coordinates'] = [(p + c) for p in parts for c in ['_x', '_y']]
            [feats[cam][mouse]['raw_coordinates'].append(p + c) for p in inferred_parts for c in ['_x', '_y']]
            feats[cam][mouse]['intramouse_distance'] = [('dist_' + p + '_' + q) for p in parts for q in parts if q != p]
        for mouse2 in pairmice:
            feats[cam][mouse+mouse2]['intermouse_distance'] = [('dist_m1' + p + '_m2' + q) for p in parts for q in parts]

    return feats


def list_features(project):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    feats = generate_valid_feature_list(cfg)
    print('The following feature categories are available:')
    for cam in feats.keys():
        for mouse in feats[cam].keys():
            print("in feats[" + cam + "][" + mouse + "]:")
            for k in feats[cam][mouse].keys():
                print("    '" + k + "'")
            print(' ')
    print('\nFeatures included in each category are:')
    for cam in feats.keys():
        for mouse in feats[cam].keys():
            for feat in feats[cam][mouse].keys():
                print(cam + '|' + mouse + '|' + feat + ':')
                print("   {'" + "', '".join(feats[cam][mouse][feat]) + "'}")
        print(' ')


def generate_lambdas():
    # define the lambdas for all the features, grouped by their required inputs.
    # all function definitions are in MARS_feature_machinery.
    # units for all lambdas are in pixels and frames. These must be converted to mouselengths and seconds
    # by the extract_features function.

    eps = np.spacing(1)
    parts_list = ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base', 'left_front_paw',
                   'right_front_paw', 'left_rear_paw', 'right_rear_paw']

    # lambdas are grouped by what kind of input they take. not very intuitive naming, this is supposed
    # to be behind the scenes. if you want to play with which features you compute, modify the groups
    # in generate_feature_list, above.
    lam = {'ell_ang': {}, 'ell': {}, 'ell_area': {}, 'xy_ang': {}, 'xy': {}, 'xybd': {}, 'dt': {}, '2mdt': {},
           'd2t': {}, 'xyxy_ang': {}, 'xyxy': {}, 'bb': {}}

    # features based on a fit ellipse ###################################################
    lam['ell_ang']['phi'] = lambda ell: ell['phi']
    lam['ell']['major_axis_len'] = lambda ell: ell['ra'] if ell['ra'] > 0. else eps
    lam['ell']['minor_axis_len'] = lambda ell: ell['rb'] if ell['rb'] > 0. else eps
    lam['ell_ang']['axis_ratio'] = lambda ell: ell['ra'] / ell['rb'] if ell['rb'] > 0. else eps
    lam['ell_area']['area_ellipse'] = lambda ell: mh.pi * ell['ra'] * ell['rb'] if ell['ra'] * ell['rb'] > 0. else eps

    # features based on the location of one mouse #######################################
    lam['xy_ang']['ori_head'] = lambda x, y: get_angle(x[3], y[3], x[0], y[0])
    lam['xy_ang']['ori_body'] = lambda x, y: get_angle(x[6], y[6], x[3], y[3])
    lam['xy_ang']['angle_head_body_l'] = lambda x, y: interior_angle([x[2], y[2]], [x[3], y[3]], [x[5], y[5]])
    lam['xy_ang']['angle_head_body_r'] = lambda x, y: interior_angle([x[1], y[1]], [x[3], y[3]], [x[4], y[4]])
    lam['xy']['centroid_x'] = lambda x, y: np.mean(x)
    lam['xy']['centroid_y'] = lambda x, y: np.mean(y)
    lam['xy']['centroid_head_x'] = lambda x, y: np.mean(x[:3])
    lam['xy']['centroid_head_y'] = lambda x, y: np.mean(y[:3])
    lam['xy']['centroid_body_x'] = lambda x, y: np.mean(x[4:])
    lam['xy']['centroid_body_y'] = lambda x, y: np.mean(y[4:])
    for i, p1 in enumerate(parts_list):
        for j, p2 in enumerate(parts_list):
            if p1 != p2:
                lam['xy']['dist_' + p1 + '_' + p2] = lambda x, y, ind1=i, ind2=j: \
                                                            np.linalg.norm([x[ind1] - x[ind2], y[ind1] - y[ind2]])
    for i, part in enumerate(parts_list):
        lam['xy'][part + '_x'] = lambda x, y, ind=i: x[ind]
        lam['xy'][part + '_y'] = lambda x, y, ind=i: y[ind]

    # features based on position w.r.t. arena ###########################################
    lam['xybd']['dist_edge_x'] = lambda x, y, xlims, ylims:\
        np.amin(np.stack((np.maximum(0, lam['xy']['centroid_x'](x, y) - xlims[0]),
                          np.maximum(0, xlims[1] - lam['xy']['centroid_x'](x, y))), axis=-1), axis=0)
    lam['xybd']['dist_edge_y'] = lambda x, y, xlims, ylims: \
        np.amin(np.stack((np.maximum(0, lam['xy']['centroid_y'](x, y) - ylims[0]),
                          np.maximum(0, ylims[1] - lam['xy']['centroid_y'](x, y))), axis=-1), axis=0)
    lam['xybd']['dist_edge'] = lambda x, y, xlims, ylims:\
        np.amin(np.stack((lam['xybd']['dist_edge_x'](x, y, xlims, ylims),
                          lam['xybd']['dist_edge_y'](x, y, xlims, ylims)), axis=-1), axis=0)

    # velocity features #################################################################
    # question: should we instead estimate velocities with a kalman filter, to reduce noise?
    lam['dt']['speed'] = lambda xt1, yt1, xt2, yt2: speed_head_hips(lam, xt1, yt1, xt2, yt2)
    lam['dt']['speed_centroid'] = lambda xt1, yt1, xt2, yt2: speed_centroid(lam, xt1, yt1, xt2, yt2)
    lam['dt']['speed_fwd'] = lambda xt1, yt1, xt2, yt2: speed_fwd(lam, xt1, yt1, xt2, yt2)
    # going to omit the windowed [('speed_centroid_' + w) for w in ['w2', 'w5', 'w10']],
    # as these are too sensitive to changes in imaging framerate

    # social velocity features ##########################################################
    lam['2mdt']['radial_vel'] = lambda xt2, yt2, xt1, yt1, x2, y2: radial_vel(lam, xt2, yt2, xt1, yt1, x2, y2)
    lam['2mdt']['tangential_vel'] = lambda xt2, yt2, xt1, yt1, x2, y2: tangential_vel(lam, xt2, yt2, xt1, yt1, x2, y2)

    # acceleration features #############################################################
    lam['d2t']['acceleration_head'] = lambda x2, y2, x1, y1, x0, y0: acceleration_head(lam, x2, y2, x1, y1, x0, y0)
    lam['d2t']['acceleration_body'] = lambda x2, y2, x1, y1, x0, y0: acceleration_body(lam, x2, y2, x1, y1, x0, y0)
    lam['d2t']['acceleration_centroid'] = lambda x2, y2, x1, y1, x0, y0: acceleration_ctr(lam, x2, y2, x1, y1, x0, y0)

    # features based on the locations of both mice ######################################
    lam['xyxy_ang']['facing_angle'] = lambda x1, y1, x2, y2: facing_angle(lam, x1, y1, x2, y2)
    lam['xyxy_ang']['angle_between'] = lambda x1, y1, x2, y2: angle_between(lam, x1, y1, x2, y2)
    lam['xyxy_ang']['angle_social'] = lambda x1, y1, x2, y2: soc_angle(lam, x1, y1, x2, y2)

    lam['xyxy']['dist_nose'] = lambda x1, y1, x2, y2: dist_nose(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_body'] = lambda x1, y1, x2, y2: dist_body(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_head'] = lambda x1, y1, x2, y2: dist_head(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_centroid'] = lambda x1, y1, x2, y2: dist_centroid(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_head_body'] = lambda x1, y1, x2, y2: dist_head_body(lam, x1, y1, x2, y2)
    lam['xyxy']['dist_gap'] = lambda x1, y1, x2, y2: dist_gap(lam, x1, y1, x2, y2)
    lam['xyxy_ang']['area_ellipse_ratio'] = lambda x1, y1, x2, y2: \
        lam['ell_area']['area_ellipse'](fit_ellipse(x1, y1))/lam['ell_area']['area_ellipse'](fit_ellipse(x2, y2))

    for i, p1 in enumerate(parts_list):
        for j, p2 in enumerate(parts_list):
            lam['xyxy']['dist_m1' + p1 + '_m2' + p2] = \
                lambda x1, y1, x2, y2, ind1=i, ind2=j: np.linalg.norm([x1[ind1] - x2[ind2], y1[ind1] - y2[ind2]])

    # features based on the bounding boxes ##############################################
    lam['bb']['overlap_bboxes'] = lambda box1, box2: bb_intersection_over_union(box1, box2)

    return lam


def flatten_feats(feats, use_grps=[], use_cams=[], use_mice=[]):
    features = []
    for cam in use_cams:
        for mouse in use_mice:
            for feat_class in use_grps:
                if feat_class in feats[cam][mouse].keys():
                    features = features + ["_".join((cam, mouse, s)) for s in feats[cam][mouse][feat_class]]

    return features


def center_on_mouse(xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb, xlims, ylims):
    # Translate and rotate points so that the neck of mouse A is at the origin, and ears are flat.
    ori_x = xa[3]
    ori_y = ya[3]
    phi = np.arctan2(ya[2] - ya[1], xa[2] - xa[1])
    lam_x = lambda x, y:  (x - ori_x) * np.cos(-phi) + (y - ori_y) * np.sin(-phi)
    lam_y = lambda x, y: -(x - ori_x) * np.sin(-phi) + (y - ori_y) * np.cos(-phi)

    xa_r = lam_x(xa, ya)
    ya_r = lam_y(xa, ya)
    xb_r = lam_x(xb, yb)
    yb_r = lam_y(xb, yb)
    xa0_r = lam_x(xa0, ya0)
    ya0_r = lam_y(xa0, ya0)
    xa00_r = lam_x(xa00, ya00)
    ya00_r = lam_y(xa00, ya00)
    boxa_r = [lam_x(boxa[0], boxa[1]), lam_y(boxa[0], boxa[1]),
              lam_x(boxa[2], boxa[3]), lam_y(boxa[2], boxa[3])]
    boxb_r = [lam_x(boxb[0], boxb[1]), lam_y(boxb[0], boxb[1]),
              lam_x(boxb[2], boxb[3]), lam_y(boxb[2], boxb[3])]
    xlims_r = xlims - ori_x
    ylims_r = ylims - ori_y

    return xa_r, ya_r, xb_r, yb_r, xa0_r, ya0_r, xa00_r, ya00_r, boxa_r, boxb_r, xlims_r, ylims_r


def smooth_keypoint_trajectories(keypoints):
    # TODO: do Kalman smoothing or something to get rid of bad keypoints.
    keypoints_sm = keypoints

    return keypoints_sm


def get_mars_keypoints(keypoints, num_mice, partorder):
    xraw = []
    yraw = []
    for m in range(num_mice):
        xraw.append(np.asarray(keypoints[m][0]))
        yraw.append(np.asarray(keypoints[m][1]))
    xm = []
    ym = []
    for m in range(num_mice):
        xm.append(np.array([]))
        ym.append(np.array([]))
        for part in partorder:
            xm[m] = np.append(xm[m], np.mean(xraw[m][part]))
            ym[m] = np.append(ym[m], np.mean(yraw[m][part]))
    return xm, ym


def run_feature_extraction(sequence, cfg, use_grps=[], use_cam='top', mouse_list=[], smooth_keypoints=False, center_mouse=False):

    keypoints = [f for f in sequence['keypoints']]

    # currently not implemented smoothing:
    if smooth_keypoints:
        keypoints = smooth_keypoint_trajectories(keypoints)

    dscale = cfg['pixels_per_cm']
    fps = cfg['framerate']
    num_frames = len(keypoints)
    num_mice = len(cfg['animal_names'])*cfg['num_obj']
    if not mouse_list:
        mouse_list = ['m' + str(i) for i in range(num_mice)]

    parts = cfg['keypoints']
    nose       = [parts.index(i) for i in cfg['mars_name_matching']['nose']]
    left_ear   = [parts.index(i) for i in cfg['mars_name_matching']['left_ear']]
    right_ear  = [parts.index(i) for i in cfg['mars_name_matching']['right_ear']]
    neck       = [parts.index(i) for i in cfg['mars_name_matching']['neck']]
    left_side  = [parts.index(i) for i in cfg['mars_name_matching']['left_side']]
    right_side = [parts.index(i) for i in cfg['mars_name_matching']['right_side']]
    tail       = [parts.index(i) for i in cfg['mars_name_matching']['tail']]
    # num_parts = len(parts)
    num_parts = 7  # for now we're just supporting the MARS-style keypoints
    partorder = [nose, right_ear, left_ear, neck, right_side, left_side, tail]

    feats = generate_valid_feature_list(cfg)
    lam = generate_lambdas()
    if not use_grps:
        use_grps = []
        for mouse in mouse_list:
            use_grps = use_grps + list(feats[use_cam][mouse].keys())
    else:
        for grp in use_grps:
            if grp not in feats[use_cam][mouse_list[0]].keys() and grp not in feats[use_cam][mouse_list[1]+mouse_list[0]].keys():
                raise Exception(grp+' is not a valid feature group name.')
    features = flatten_feats(feats, use_grps=use_grps, use_cams=[use_cam], use_mice=mouse_list)
    print(features)
    features_ordered = []
    num_features = len(features)

    try:
        # bar = progressbar.ProgressBar(widgets=[progressbar.FormatLabel('Feats frame %(value)d'), '/',
        #                                        progressbar.FormatLabel('%(max)d  '),
        #                                        progressbar.Percentage(), ' -- ', ' [', progressbar.Timer(), '] ',
        #                                        progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=num_frames)
        track = {'features': features,
                 'data': np.zeros((num_mice, num_frames, num_features)),
                 'bbox': np.zeros((num_mice, 4, num_frames)),
                 'keypoints': keypoints,
                 'fps': fps}

        # get some scaling parameters ###################################################
        mouse_length = np.zeros(num_frames)
        allx = []
        ally = []
        for f in range(num_frames):
            keypoints = sequence['keypoints'][f]
            xm, ym = get_mars_keypoints(keypoints, num_mice, partorder)

            [allx.append(x) for x in np.ravel(xm)]
            [ally.append(y) for y in np.ravel(ym)]
            mouse_length[f] = np.linalg.norm((xm[0][3] - xm[0][6], ym[0][3] - ym[0][6]))

        # estimate the extent of our arena from tracking data
        allx = np.asarray(allx)/dscale
        ally = np.asarray(ally)/dscale
        xlims_0 = [np.percentile(allx, 1), np.percentile(allx, 99)]
        ylims_0 = [np.percentile(ally, 1), np.percentile(ally, 99)]
        xm0 = [np.array([]) for i in range(num_mice)]
        ym0 = [np.array([]) for i in range(num_mice)]
        xm00 = [np.array([]) for i in range(num_mice)]
        ym00 = [np.array([]) for i in range(num_mice)]

        # extract features ##############################################################
        pr = np.linspace(0, num_frames - 1, num_frames)  # for tracking progress
        # bar.start()
        for f in range(num_frames):
            # bar.update(pr[f])

            if f > 1:
                for m in range(num_mice):
                    xm00[m] = xm0[m]
                    ym00[m] = ym0[m]
            if f != 0:
                for m in range(num_mice):
                    xm0[m] = xm[m]
                    ym0[m] = ym[m]

            keypoints = sequence['keypoints'][f]
            xm, ym = get_mars_keypoints(keypoints, num_mice, partorder)

            bboxes = []
            for m in range(num_mice):
                bboxes.append(np.asarray(sequence['bbox'])[m, :, f])
            if f == 0:
                for m in range(num_mice):
                    xm0[m] = xm[m]
                    ym0[m] = ym[m]
            if f <= 1:
                for m in range(num_mice):
                    xm00[m] = xm0[m]
                    ym00[m] = ym0[m]

            mouse_vals = []
            if num_mice > 1:
                for mouse1 in range(num_mice):
                    for mouse2 in range(num_mice):
                        if mouse2 == mouse1:
                            continue
                        mouse_vals.append(('m'+str(mouse1), 'm'+str(mouse2), xm[mouse1], ym[mouse1], xm[mouse2], ym[mouse2], xm0[mouse1], ym0[mouse1], xm00[mouse1], ym00[mouse1], bboxes[mouse1], bboxes[mouse2]))
            else:
                mouse_vals.append(('m0', '', xm[0], ym[0], xm[0], ym[0], xm0[0], ym0[0], xm00[0], ym00[0], bboxes[0], bboxes[0]))
            for m, (maStr, mbStr, xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb) in enumerate(mouse_vals):
                if center_mouse:
                    (xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb, xlims, ylims) = \
                        center_on_mouse(xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb, xlims_0, ylims_0)
                else:
                    xlims = xlims_0
                    ylims = ylims_0

                # single-mouse features. Lambda returns pixels, convert to cm.
                for feat in lam['xy'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xy'][feat](xa, ya) / dscale
                        if m==0 and f==0:  # only do this for the first mouse
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

                # single-mouse angle or ratio features. No unit conversion needed.
                for feat in lam['xy_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xy_ang'][feat](xa, ya)
                        if m == 0 and f == 0:
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

                # ellipse-based features. Lambda returns pixels, convert to cm.
                ell = fit_ellipse(xa, ya)
                for feat in lam['ell'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell'][feat](ell) / dscale
                        if m == 0 and f == 0:
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

                # ellipse-based angle or ratio features. No unit conversion needed.
                for feat in lam['ell_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell_ang'][feat](ell)
                        if m == 0 and f == 0:
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

                # ellipse-based area features. Lambda returns pixels^2, convert to cm^2.
                for feat in lam['ell_area'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell_area'][feat](ell) / (dscale ** 2)
                        if m == 0 and f == 0:
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

                # velocity features. Lambda returns pix/frame, convert to cm/second.
                for feat in lam['dt'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['dt'][feat](xa, ya, xa0, ya0) * fps / dscale
                        if m == 0 and f == 0:
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

                # acceleration features. Lambda returns pix/frame^2, convert to cm/second^2.
                for feat in lam['d2t'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = \
                            lam['d2t'][feat](xa, ya, xa0, ya0, xa00, ya00) * fps * fps / dscale
                        if m == 0 and f == 0:
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

                if num_mice > 1:
                    # two-mouse features. Lambda returns pixels, convert to cm.
                    for feat in lam['xyxy'].keys():
                        featname = "_".join((use_cam, maStr+mbStr, feat)) # change m to str
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy'][feat](xa, ya, xb, yb) / dscale
                            if m == 0 and f == 0:
                                features_ordered.append('_'.join((use_cam, maStr+mbStr, feat)))

                    # two-mouse angle or ratio features. No unit conversion needed.
                    for feat in lam['xyxy_ang'].keys():
                        featname = "_".join((use_cam, maStr+mbStr, feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy_ang'][feat](xa, ya, xb, yb)
                            if m == 0 and f == 0:
                                features_ordered.append('_'.join((use_cam, maStr + mbStr, feat)))

                    # two-mouse velocity features. Lambda returns pix/frame, convert to cm/second.
                    for feat in lam['2mdt'].keys():
                        featname = "_".join((use_cam, maStr+mbStr, feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = \
                                lam['2mdt'][feat](xa, ya, xa0, ya0, xb, yb) * fps / dscale
                            if m == 0 and f == 0:
                                features_ordered.append('_'.join((use_cam, maStr + mbStr, featname)))

                # Bounding box features. No unit conversion needed so far.
                for feat in lam['bb'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['bb'][feat](boxa, boxb)
                        if m == 0 and f == 0:
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

                # environment-based features. Lambda returns pixels, convert to cm.
                for feat in lam['xybd'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xybd'][feat](xa, ya, xlims, ylims) / dscale
                        if m == 0 and f == 0:
                            features_ordered.append('_'.join((use_cam, maStr, feat)))

        # bar.finish()
        track['features'] = features_ordered
        return track

    except Exception as e:
        import linecache
        print("Error when extracting features:")
        exc_type, exc_obj, tb = sys.exc_info()
        filename = tb.tb_frame.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, tb.tb_lineno, tb.tb_frame.f_globals)
        print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, tb.tb_lineno, line.strip(), exc_obj))
        print(e)
        return []


def extract_features(project, progress_bar_sig='', sets_to_process = ['train', 'test', 'val']):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)

    # backwards compatibility, add feat_list to old projects and default to using all available features
    if 'feat_list' not in clf_params.keys() or not clf_params['feat_list']:
        clf_params['feat_list'] = []
        feats = generate_valid_feature_list(cfg)
        use_cam = list(feats.keys())[0]
        mouse_list = list(feats[use_cam].keys())
        for mouse in mouse_list:
            clf_params['feat_list'] = clf_params['feat_list'] + list(feats[use_cam][mouse].keys())
        with open(config_fid,'w') as f:
            yaml.dump(clf_params, f)

    for key in sets_to_process:
        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_data.json')) as f:
            data = json.load(f)

        feats = {'feature_names': [], 'sequences': {cfg['project_name']: {}}}
        keylist = list(data['sequences'][cfg['project_name']].keys())
        for i, k in enumerate(keylist):
            feats['sequences'][cfg['project_name']][k] = []
            for j, entry in enumerate(data['sequences'][cfg['project_name']][k]):
                print('%s video %i/%i: %s clip %i/%i' % (key, i+1, len(keylist), k, j+1, len(data['sequences'][cfg['project_name']][k])))
                feat_dict = run_feature_extraction(entry, cfg, use_grps=clf_params['feat_list'])
                if not feat_dict:
                    print('skipping for no feats, something went wrong')
                else:
                    feats['feature_names'] = feat_dict['features']
                    feats['vocabulary'] = data['vocabulary']
                    if 'annotations' in entry.keys():
                        insert_dict = {'features': feat_dict['data'].tolist(), 'annotations': entry['annotations']}
                    else:
                        insert_dict = {'features': feat_dict['data'].tolist()}
                    feats['sequences'][cfg['project_name']][k].append(insert_dict)

        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_features.json'), 'w') as f:
            json.dump(feats, f)
