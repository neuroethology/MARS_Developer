from __future__ import print_function,division
import sys, os
import warnings
import json, yaml
import numpy as np
warnings.filterwarnings('ignore')
from behavior_classification.MARS_feature_machinery import *


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
    mice = ['m'+str(i+1) for i in range(num_mice)]

    # TODO: replace hard-coded feature names with keypoints from the project config file.
    parts_top = ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base']
    parts_front = ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base', 'left_front_paw',
                   'right_front_paw', 'left_rear_paw', 'right_rear_paw']
    inferred_parts = ['centroid', 'centroid_head', 'centroid_body']

    feats = {'top': {'m1': {}, 'm2': {}}, 'front': {'m1': {}, 'm2': {}}}
    for cam in ['top', 'front']:
        for mouse in mice:
            feats[cam][mouse]['absolute_orientation'] = ['phi', 'ori_head', 'ori_body']
            feats[cam][mouse]['joint_angle'] = ['angle_head_body_l', 'angle_head_body_r']
            feats[cam][mouse]['fit_ellipse'] = ['major_axis_len', 'minor_axis_len', 'axis_ratio', 'area_ellipse']
            feats[cam][mouse]['distance_to_walls'] = ['dist_edge_x', 'dist_edge_y', 'dist_edge']
            feats[cam][mouse]['speed'] = ['speed', 'speed_centroid', 'speed_fwd', 'radial_vel', 'tangential_vel']
            feats[cam][mouse]['acceleration'] = ['acceleration_head', 'acceleration_body', 'acceleration_centroid']

        if num_mice > 1:  # multi-animal features. TODO: generalize this to >2 mice.
            feats[cam]['m1']['social_angle'] = ['angle_between', 'facing_angle', 'angle_social']
            feats[cam]['m1']['relative_size'] = ['area_ellipse_ratio']
            feats[cam]['m1']['social_distance'] = ['dist_centroid', 'dist_nose', 'dist_head', 'dist_body',
                                                   'dist_head_body', 'dist_gap', 'dist_scaled', 'overlap_bboxes']

    for cam, parts in zip(['top', 'front'], [parts_top, parts_front]):
        for mouse in mice:
            feats[cam][mouse]['raw_coordinates'] = [(p + c) for p in parts for c in ['_x', '_y']]
            [feats[cam][mouse]['raw_coordinates'].append(p + c) for p in inferred_parts for c in ['_x', '_y']]
            feats[cam][mouse]['intramouse_distance'] = [('dist_' + p + '_' + q) for p in parts for q in parts if q != p]
        if num_mice > 1:
            feats[cam]['m1']['intermouse_distance'] = [('dist_m1' + p + '_m2' + q) for p in parts for q in parts]

    return feats


def list_features(project):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    feats = generate_valid_feature_list(cfg)
    print('All available feature categories:')
    print("feats = ['" + "', '".join(list(feats['top']['m1'].keys())) + "']")
    print('\nFeatures included in each category:')
    for feat in feats['top']['m1'].keys():
        print(feat + ':')
        print("   {'" + "', '".join(feats['top']['m1'][feat]) + "'}")


def generate_lambdas():
    # define the lambdas for all the features, grouped by their required inputs.
    # all function definitions are in MARS_feature_machinery.
    # units for all lambdas are in pixels and frames. These must be converted to mouselengths and seconds
    # by the extract_features function.

    eps = np.spacing(1)
    parts_front = ['nose', 'right_ear', 'left_ear', 'neck', 'right_side', 'left_side', 'tail_base', 'left_front_paw',
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
    for i, p1 in enumerate(parts_front):
        for j, p2 in enumerate(parts_front):
            if p1 != p2:
                lam['xy']['dist_' + p1 + '_' + p2] = lambda x, y, ind1=i, ind2=j: \
                                                            np.linalg.norm([x[ind1] - x[ind2], y[ind1] - y[ind2]])
    for i, part in enumerate(parts_front):
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

    for i, p1 in enumerate(parts_front):
        for j, p2 in enumerate(parts_front):
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


def extract_features(sequence, cfg, use_grps=[], use_cam='top', use_mice=['m1','m2'],
                     smooth_keypoints=False, center_mouse=False):

    keypoints = [f for f in sequence['keypoints']]

    # currently not implemented smoothing:
    if smooth_keypoints:
        keypoints = smooth_keypoint_trajectories(keypoints)

    dscale = cfg['pixels_per_cm']
    fps = cfg['framerate']
    num_frames = len(keypoints)
    num_mice = len(cfg['animal_names'])*cfg['num_obj']
    parts = cfg['keypoints']
    num_parts = len(parts)

    feats = generate_valid_feature_list(cfg)
    lam = generate_lambdas()
    if not use_grps:
        use_grps = feats[use_cam]['m1'].keys()
    else:
        for grp in use_grps:
            if grp not in feats[use_cam]['m1'].keys():
                raise Exception(grp+' is not a valid feature group name.')
    features = flatten_feats(feats, use_grps=use_grps, use_cams=[use_cam], use_mice=use_mice)
    num_features = len(features)

    try:
        bar = progressbar.ProgressBar(widgets= [progressbar.FormatLabel('Feats frame %(value)d'), '/',
                                                progressbar.FormatLabel('%(max)d  '),
                                                progressbar.Percentage(), ' -- ', ' [', progressbar.Timer(), '] ',
                                                progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=num_frames)
        track = {'features': features,
                 'data': np.zeros((2, num_frames, num_features)),
                 'bbox': np.zeros((2, 4, num_frames)),
                 'keypoints': keypoints,
                 'fps': fps}

        # get some scaling parameters ###################################################
        mouse_length = np.zeros(num_frames)
        allx = []
        ally = []
        for f in range(num_frames):
            keypoints = sequence['keypoints'][f]
            xm1 = np.asarray(keypoints[0][0])
            ym1 = np.asarray(keypoints[0][1])
            xm2 = np.asarray(keypoints[1][0]) if num_mice > 1 else np.zeros()
            ym2 = np.asarray(keypoints[1][1])

            [allx.append(x) for x in (xm1, xm2)]
            [ally.append(y) for y in (ym1, ym2)]
            mouse_length[f] = np.linalg.norm((xm1[3] - xm1[6], ym1[3] - ym1[6]))

        # estimate the extent of our arena from tracking data
        allx = np.concatenate(allx).ravel()/dscale
        ally = np.concatenate(ally).ravel()/dscale
        xlims_0 = [np.percentile(allx, 1), np.percentile(allx, 99)]
        ylims_0 = [np.percentile(ally, 1), np.percentile(ally, 99)]

        # extract features ##############################################################
        mouse_list = ['m1', 'm2']
        pr = np.linspace(0, num_frames - 1, num_frames)  # for tracking progress
        bar.start()
        for f in range(num_frames):
            bar.update(pr[f])

            if f > 1:
                xm100 = xm10
                ym100 = ym10
                xm200 = xm20
                ym200 = ym20
            if f != 0:
                xm10 = xm1
                ym10 = ym1
                xm20 = xm2
                ym20 = ym2
            xm1 = np.asarray(sequence['keypoints'][f][0][0])
            ym1 = np.asarray(sequence['keypoints'][f][0][1])
            xm2 = np.asarray(sequence['keypoints'][f][1][0]) if num_mice > 1 else np.zeros(num_parts)
            ym2 = np.asarray(sequence['keypoints'][f][1][1]) if num_mice > 1 else np.zeros(num_parts)
            bboxes1 = np.asarray(sequence['bbox'][f])[0, :]
            bboxes2 = np.asarray(sequence['bbox'][f])[1, :] if num_mice > 1 else np.zeros(bboxes1.shape())
            if f == 0:
                xm10 = xm1
                ym10 = ym1
                xm20 = xm2
                ym20 = ym2
            if f <= 1:
                xm100 = xm10
                ym100 = ym10
                xm200 = xm20
                ym200 = ym20

            m1_vals = (xm1, ym1, xm2, ym2, xm10, ym10, xm100, ym100, bboxes1, bboxes2)
            m2_vals = (xm2, ym2, xm1, ym1, xm20, ym20, xm200, ym200, bboxes2, bboxes1)
            for m, (xa, ya, xb, yb, xa0, ya0, xa00, ya00, boxa, boxb) in enumerate([m1_vals, m2_vals]):
                if not mouse_list[m] in use_mice:
                    continue

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

                # single-mouse angle or ratio features. No unit conversion needed.
                for feat in lam['xy_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xy_ang'][feat](xa, ya)

                # ellipse-based features. Lambda returns pixels, convert to cm.
                ell = fit_ellipse(xa, ya)
                for feat in lam['ell'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell'][feat](ell) / dscale

                # ellipse-based angle or ratio features. No unit conversion needed.
                for feat in lam['ell_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell_ang'][feat](ell)

                # ellipse-based area features. Lambda returns pixels^2, convert to cm^2.
                for feat in lam['ell_area'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['ell_area'][feat](ell) / (dscale ** 2)

                # velocity features. Lambda returns pix/frame, convert to cm/second.
                for feat in lam['dt'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['dt'][feat](xa, ya, xa0, ya0) * fps / dscale

                # acceleration features. Lambda returns pix/frame^2, convert to cm/second^2.
                for feat in lam['d2t'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = \
                            lam['d2t'][feat](xa, ya, xa0, ya0, xa00, ya00) * fps * fps / dscale

                if num_mice > 1:
                    # two-mouse features. Lambda returns pixels, convert to cm.
                    for feat in lam['xyxy'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy'][feat](xa, ya, xb, yb) / dscale

                    # two-mouse angle or ratio features. No unit conversion needed.
                    for feat in lam['xyxy_ang'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy_ang'][feat](xa, ya, xb, yb)

                    # two-mouse velocity features. Lambda returns pix/frame, convert to cm/second.
                    for feat in lam['2mdt'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = \
                                lam['2mdt'][feat](xa, ya, xa0, ya0, xb, yb) * fps / dscale

                # Bounding box features. No unit conversion needed so far.
                for feat in lam['bb'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['bb'][feat](boxa, boxb)

                # environment-based features. Lambda returns pixels, convert to cm.
                for feat in lam['xybd'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xybd'][feat](xa, ya, xlims, ylims) / dscale

        bar.finish()
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

