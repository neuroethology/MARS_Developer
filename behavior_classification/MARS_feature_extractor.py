from __future__ import print_function,division
import sys, os
import copy
import warnings
import json, yaml
import numpy as np
warnings.filterwarnings('ignore')
from behavior_classification.MARS_feature_machinery import *
import behavior_classification.MARS_feature_lambdas as mars_lambdas
import pdb

flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


def load_pose(pose_fullpath):
    try:
        with open(pose_fullpath, 'r') as fp:
            pose = json.load(fp)
        return pose
    except Exception as e:
        raise e


def list_features(project):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    feats = mars_lambdas.generate_valid_feature_list(cfg)
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


def fit_ellipse(X, Y):
    data = [X.tolist(), Y.tolist()]
    mu = np.mean(data, axis=1)
    covariance = np.cov(data)
    rad = (-2 * np.log(1 - .75)) ** .5
    _, D, R = np.linalg.svd(covariance)
    normstd = np.sqrt(D)

    a = rad * normstd[0]
    b = rad * normstd[1]
    cx = mu[0]
    cy = mu[1]

    phi = (mh.atan2(np.mean(X[4:7]) - np.mean(X[:3]), np.mean(Y[4:7]) - np.mean(Y[:3])) + mh.pi / 2.) % (mh.pi * 2)
    theta_grid = np.linspace(-mh.pi, mh.pi, 200)
    xs = cx + a * np.cos(theta_grid) * np.cos(phi) + b * np.sin(theta_grid) * np.sin(phi)
    ys = cy - a * np.cos(theta_grid) * np.sin(phi) + b * np.sin(theta_grid) * np.cos(phi)

    # draw ori
    ori_vec_v = np.array([mh.cos(phi), -mh.sin(phi)]) * a
    ori_vec_h = np.array([mh.sin(phi), mh.cos(phi)]) * b
    ell = {'cx': cx, 'cy': cy, 'ra': a, 'rb': b, 'phi': phi,
           'xs': xs, 'ys': ys, 'ori_vec_v': ori_vec_v, 'ori_vec_h': ori_vec_h}

    return ell


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
    num_parts = 7  # for now, we're just supporting the MARS-style keypoints
    partorder = [nose, right_ear, left_ear, neck, right_side, left_side, tail]

    feats = mars_lambdas.generate_valid_feature_list(cfg)
    lam = mars_lambdas.generate_lambdas()
    if not use_grps:
        use_grps = []
        for mouse in mouse_list:
            use_grps = use_grps + list(feats[use_cam][mouse].keys())
    else:
        for grp in use_grps:
            if grp not in feats[use_cam][mouse_list[0]].keys():
                raise Exception(grp+' is not a valid feature group name.')
    use_grps.sort()
    features = flatten_feats(feats, use_grps=use_grps, use_cams=[use_cam], use_mice=mouse_list)
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
                bboxes.append(np.asarray(sequence['bbox'][f])[0, :])
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

                # single-mouse angle or ratio features. No unit conversion needed.
                for feat in lam['xy_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xy_ang'][feat](xa, ya)

                # sin/cos of single-mouse angle or ratio features. Unitless.
                for feat in lam['xy_ang_trig'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xy_ang_trig'][feat](xa, ya)

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

                    # two-mouse angle or ratio features. No unit conversion needed.
                    for feat in lam['xyxy_ang_trig'].keys():
                        featname = "_".join((use_cam, mouse_list[m], feat))
                        if featname in features:
                            track['data'][m, f, features.index(featname)] = lam['xyxy_ang_trig'][feat](xa, ya, xb, yb)

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

                # environment-based angle or ratio features. No unit conversion needed.
                for feat in lam['xybd_ang'].keys():
                    featname = "_".join((use_cam, mouse_list[m], feat))
                    if featname in features:
                        track['data'][m, f, features.index(featname)] = lam['xybd_ang'][feat](xa, ya, xlims, ylims)

        # bar.finish()
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


def extract_features(project, progress_bar_sig=''):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)

    # backwards compatibility, add feat_list to old projects and default to using all available features
    if 'feat_list' not in clf_params.keys() or not clf_params['feat_list']:
        clf_params['feat_list'] = []
        feats = mars_lambdas.generate_valid_feature_list(cfg)
        use_cam = list(feats.keys())[0]
        mouse_list = list(feats[use_cam].keys())
        for mouse in mouse_list:
            clf_params['feat_list'] = clf_params['feat_list'] + list(feats[use_cam][mouse].keys())
        with open(config_fid,'w') as f:
            yaml.dump(clf_params, f)

    for key in ['train', 'test', 'val']:
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
                    feats['sequences'][cfg['project_name']][k].append({'features': feat_dict['data'].tolist(), 'annotations': entry['annotations']})

        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_features.json'), 'w') as f:
            json.dump(feats, f)
