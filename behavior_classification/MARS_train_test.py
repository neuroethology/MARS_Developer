from __future__ import division
import os,sys,fnmatch
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import binarize
import dill
import json
import yaml
import time
from sklearn.ensemble import BaggingClassifier
from hmmlearn import hmm
from scipy import signal
import copy
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from behavior_classification import annotation_parsers as map
import gc
from behavior_classification import MARS_ts_util as mts
import joblib
from behavior_classification.behavior_helpers import *
from Util.seqIo import *
import scipy.io as sio
import pdb


# warnings.filterwarnings("ignore")
# plt.ioff()


lcat = lambda L: [i for j in L for i in j]
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


def clf_suffix(clf_params):
    if clf_params['clf_type'].lower() == 'mlp':
        suff = '_layers' + '-'.join(clf_params['hidden_layer_sizes']) if 'hidden_layer_sizes' in clf_params.keys() else ''
        suff = suff + '/'

    elif clf_params['clf_type'].lower() == 'xgb':
        suff = '_es' + str(clf_params['early_stopping']) if 'early_stopping' in clf_params.keys() else ''
        suff = suff + '_depth' + str(clf_params['max_depth']) if 'max_depth' in clf_params.keys() else suff
        suff = suff + '_child' + str(
            clf_params['min_child_weight']) if 'min_child_weight' in clf_params.keys() else suff

    else:  # defaults to xgb
        suff = '_es' + str(clf_params['early_stopping']) if 'early_stopping' in clf_params.keys() else ''
        suff = suff + '_depth' + str(clf_params['max_depth']) if 'max_depth' in clf_params.keys() else suff
        suff = suff + '_child' + str(clf_params['min_child_weight']) if 'min_child_weight' in clf_params.keys() else suff

    suff = suff + '_wnd' if clf_params['do_wnd'] else suff
    suff = suff + '_cwt' if clf_params['do_cwt'] else suff
    suff = suff + '_' + str(clf_params['user_suff']) if 'user_suff' in clf_params.keys() else suff
    suff = suff + '/'
    return suff


def unpack_params(clf_params, clf_type):
    with open('clf_defaults.yaml') as f:
        defaults = yaml.load(f, Loader=yaml.FullLoader)
    params = {}
    for k in defaults[clf_type].keys():
        if k in clf_params.keys():
            params[k] = clf_params[k]
        else:
            params[k] = defaults[clf_type][k]
    return params


def choose_classifier(clf_params):
    if clf_params['clf_type'].lower() == 'mlp':
        params = unpack_params(clf_params, 'mlp_defaults')
        mlp = MLPClassifier(**params)
        clf = BaggingClassifier(mlp, max_samples=0.1, n_jobs=3, random_state=7, verbose=0)

    elif clf_params['clf_type'].lower() == 'xgb':
        params = unpack_params(clf_params, 'xgb_defaults')
        clf = XGBClassifier(**params)

    else:
        print('Unrecognized classifier type %s, defaulting to XGBoost!' % clf_params['clf_type'])
        params = unpack_params(clf_params, 'xgb_defaults')
        clf = XGBClassifier(**params)
    return clf


def load_data(project, dataset, train_behaviors, drop_behaviors=[]):
    with open(os.path.join(project, 'project_config.yaml')) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(project, 'behavior', 'config_classifiers.yaml')) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(project, 'behavior', 'behavior_equivalences.yaml')) as f:
        equivalences = yaml.load(f, Loader=yaml.FullLoader)
        if equivalences is None:
            equivalences = {}

    if dataset in ['train', 'test', 'val']:
        with open(os.path.join(project, 'behavior', 'behavior_jsons', dataset + '_features.json')) as f:
            data = json.load(f)
    else:
        print('dataset must be train, test, or val.')
        return
    for label in train_behaviors:
        if label not in data['vocabulary']:
            print('Error: target behavior ' + label + ' not found in this dataset.\nAvailable labels:')
            print(list(data['vocabulary'].keys()))
            return

    keylist = list(data['sequences'][cfg['project_name']].keys())
    data_stack = []
    annot_raw = []
    for i, k in enumerate(keylist):
        if clf_params['verbose']:
            print('  preprocessing %s (%d/%d)' % (dataset, i+1, len(keylist)))
        feats = np.array(data['sequences'][cfg['project_name']][k]['features'])
        feats = np.swapaxes(feats, 0, 1)
        feats = mts.clean_data(feats)
        annots = data['sequences'][cfg['project_name']][k]['annotations']
        if len(annots) != feats.shape[0]:
            print('Length mismatch: %s %d %d' % (k, len(annots), d.shape[0]))
            print('Extra frames will be trimmed from the end of the sequence.')
            if len(annots) > feats.shape[0]:
                annots = annots[:feats.shape[0]]
            else:
                feats = feats[:len(annots), :, :]

        if clf_params['do_wnd']:
            feats = mts.apply_windowing(feats, cfg['framerate'])
        elif clf_params['do_cwt']:
            feats = mts.apply_wavelet_transform(feats)

        if drop_behaviors:
            if not isinstance(drop_behaviors, list):
                drop_behaviors = [drop_behaviors]
            drop_list = []
            for d in drop_behaviors:
                if d in equivalences.keys():
                    drop_list += [data['vocabulary'][i] for i in equivalences[d]]
                else:
                    drop_list.append(data['vocabulary'][d])
            keep_inds = [i for i,_annots in enumerate(annots) if _annots not in drop_list]
            annots = annots[keep_inds]
            feats = feats[keep_inds,:,:]
        annot_raw += annots
        data_stack.append(feats)
    if clf_params['verbose']:
        print('all sequences processed')
    data_stack = np.concatenate(data_stack, axis=0)

    if clf_params['verbose']:
        print('processing annotations...')
    annot_clean = {}
    for label_name in train_behaviors:
        if label_name in equivalences.keys():
            hit_list = [data['vocabulary'][i] for i in equivalences[label_name] if i in data['vocabulary']]
        else:
            hit_list = [data['vocabulary'][label_name]]
        annot_clean[label_name] = [1 if i in hit_list else 0 for i in annot_raw]
    print('done!\n')
    
    return data_stack, annot_clean


def assign_labels(all_predicted_probabilities, behaviors_used):
    # Assigns labels based on the provided probabilities.
    labels = []
    labels_num = []
    num_frames = all_predicted_probabilities.shape[0]
    # Looping over frames, determine which annotation label to take.
    for i in range(num_frames):
        # Get the [Nx2] matrix of current prediction probabilities.
        current_prediction_probabilities = all_predicted_probabilities[i]

        # Get the positive/negative labels for each behavior, by taking the argmax along the pos/neg axis.
        onehot_class_predictions = np.argmax(current_prediction_probabilities, axis=1)

        # Get the actual probabilities of those predictions.
        predicted_class_probabilities = np.max(current_prediction_probabilities, axis=1)

        # If every behavioral predictor agrees that the current_
        if np.all(onehot_class_predictions == 0):
            # The index here is one past any positive behavior --this is how we code for "other".
            beh_frame = 0
            # How do we get the probability of it being "other?" Since everyone's predicting it, we just take the mean.
            proba_frame = np.mean(predicted_class_probabilities)
            labels += ['other']
        else:
            # If we have positive predictions, we find the probabilities of the positive labels and take the argmax.
            pos = np.where(onehot_class_predictions)[0]
            max_prob = np.argmax(predicted_class_probabilities[pos])

            # This argmax is, by construction, the id for this behavior.
            beh_frame = pos[max_prob]
            proba_frame = predicted_class_probabilities[beh_frame]
            labels += [behaviors_used[beh_frame]]
            beh_frame += 1
        labels_num.append(beh_frame)

    return labels_num


def do_train(beh_classifier, X_tr, y_tr, X_ev, y_ev, savedir, verbose=0):

    beh_name = beh_classifier['beh_name']
    clf = beh_classifier['clf']
    clf_params = beh_classifier['params']

    # set some parameters for post-classification smoothing:
    kn = clf_params['smk_kn']
    blur_steps = clf_params['blur'] ** 2
    shift = clf_params['shift']

    # get the labels for the current behavior
    t = time.time()
    y_tr_beh = y_tr[beh_name]
    
    # scale the data
    if(verbose):
        print('fitting preprocessing parameters...')
    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    if not X_ev==[]:
        X_ev = scaler.transform(X_ev)

    # shuffle data
    X_tr, idx_tr = shuffle_fwd(X_tr)
    y_tr_beh = y_tr_beh[idx_tr]

    # fit the classifier!
    gc.collect()
    if (verbose):
        print('fitting clf for %s' % beh_name)
        print('    training set: %d positive / %d total (%d %%)' % (
            sum(y_tr_beh), len(y_tr_beh), 100*sum(y_tr_beh)/len(y_tr_beh)))
    if not X_ev==[]:
        eval_set = [(X_ev, y_ev[beh_name])]
        print('    eval set: %d positive / %d total (%d %%)' % (
            sum(y_ev[beh_name]), len(y_ev[beh_name]), 100 * sum(y_ev[beh_name]) / len(y_ev[beh_name])))
        if clf_params['early_stopping']:
            if verbose:
                print('  + early stopping')
            clf.fit(X_tr[::clf_params['downsample_rate'], :], y_tr_beh[::clf_params['downsample_rate']],
                    eval_set=eval_set, #eval_metric='aucpr',
                    early_stopping_rounds=clf_params['early_stopping'], verbose=True)
        else:
            clf.fit(X_tr[::clf_params['downsample_rate'], :], y_tr_beh[::clf_params['downsample_rate']],
                    eval_set=eval_set, eval_metric='aucpr', verbose=True)
        results = clf.evals_result()
    else:
        if verbose:
            print('  no validation set included')
        clf.fit(X_tr[::clf_params['downsample_rate'], :], y_tr_beh[::clf_params['downsample_rate']],
                eval_metric='aucpr', verbose=True)
        results = []

    # shuffle back
    X_tr = shuffle_back(X_tr, idx_tr)
    y_tr_beh = shuffle_back(y_tr_beh, idx_tr).astype(int)

    # evaluate on training set
    if (verbose):
        print('evaluating on the training set...')
    y_pred_proba = np.zeros((len(y_tr_beh), 2))
    gen = Batch(range(len(y_tr_beh)), lambda x: x % 1e5 == 0, 1e5)
    for i in gen:
        inds = list(i)
        pd_proba_tmp = (clf.predict_proba(X_tr[inds]))
        y_pred_proba[inds] = pd_proba_tmp
    y_pred_class = np.argmax(y_pred_proba, axis=1)

    # do hmm
    if (verbose):
        print('fitting HMM smoother...')
    hmm_bin = hmm.MultinomialHMM(n_components=2, algorithm="viterbi", random_state=42, params="", init_params="")
    hmm_bin.startprob_ = np.array([np.sum(y_tr_beh == i) / float(len(y_tr_beh)) for i in range(2)])
    hmm_bin.transmat_ = mts.get_transmat(y_tr_beh, 2)
    hmm_bin.emissionprob_ = mts.get_emissionmat(y_tr_beh, y_pred_class, 2)
    y_proba_hmm = hmm_bin.predict_proba(y_pred_class.reshape((-1, 1)))
    y_pred_hmm = np.argmax(y_proba_hmm, axis=1)

    # forward-backward smoothing with classes
    if (verbose):
        print('fitting forward-backward smoother...')
    len_y = len(y_tr_beh)
    z = np.zeros((3, len_y))
    y_fbs = np.r_[y_pred_hmm[range(shift, -1, -1)], y_pred_hmm, y_pred_hmm[range(len_y - 1, len_y - 1 - shift, -1)]]
    for s in range(blur_steps): y_fbs = signal.convolve(np.r_[y_fbs[0], y_fbs, y_fbs[-1]], kn / kn.sum(), 'valid')
    z[0, :] = y_fbs[2 * shift + 1:]
    z[1, :] = y_fbs[:-2 * shift - 1]
    z[2, :] = y_fbs[shift + 1:-shift]
    z_mean = np.mean(z, axis=0)
    y_pred_fbs = binarize(z_mean.reshape((-1, 1)), .5).astype(int).reshape((1, -1))[0]
    hmm_fbs = copy.deepcopy(hmm_bin)
    hmm_fbs.emissionprob_ = mts.get_emissionmat(y_tr_beh, y_pred_fbs, 2)
    y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
    y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)

    # print the results of training
    dt = (time.time() - t) / 60.
    print('training took %.2f mins' % dt)
    _, _, _ = prf_metrics(y_tr_beh, y_pred_class, beh_name)
    _, _, _ = prf_metrics(y_tr_beh, y_pred_hmm, beh_name)
    precision, recall, f_measure = prf_metrics(y_tr_beh, y_pred_fbs_hmm, beh_name)

    beh_classifier.update({'clf': clf,
                           'scaler': scaler,
                           'precision': precision,
                           'recall': recall,
                           'f_measure': f_measure,
                           'hmm_bin': hmm_bin,
                           'hmm_fbs': hmm_fbs})

    dill.dump(beh_classifier, open(savedir + 'classifier_' + beh_name, 'wb'))
    return results


def do_test(name_classifier, X_te, y_te, verbose=0, doPRC=0):

    classifier = joblib.load(name_classifier)

    # unpack the classifier
    beh_name = classifier['beh_name']
    clf = classifier['bag_clf']  if 'bag_clf' in classifier.keys() else classifier['clf']

    # unpack the smoother
    hmm_fbs = classifier['hmm_fbs']

    # unpack the smoothing parameters
    if 'params' in classifier.keys():
        scaler = classifier['scaler']
        clf_params = classifier['params']
        kn = clf_params['smk_kn']
        blur_steps = clf_params['blur'] ** 2
        shift = clf_params['shift']
    else:
        scaler = joblib.load(os.path.join(os.path.dirname(name_classifier),'scaler'))
        kn = classifier['k']
        blur_steps = classifier['blur_steps']
        shift = classifier['shift']
    
    # scale the data
    X_te = scaler.transform(X_te)

    t = time.time()
    len_y = len(y_te[beh_name])
    y_te_beh = y_te[beh_name]
    gt = y_te_beh

    # predict probabilities:
    if (verbose):
        print('predicting behavior probability')
    y_pred_proba = clf.predict_proba(X_te)
    proba = y_pred_proba
    
    if doPRC:
        # compute predictions as a function of threshold to make P-R curves!
        p_pos = np.squeeze(proba[:,1])
        proba_thr = np.zeros((p_pos.size,100))
        for thr in range(100):
            proba_thr[:,thr] = np.array([1 if i>(thr/100.) else 0 for i in p_pos])
    
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    preds = y_pred_class

    # forward-backward smoothing:
    if (verbose):
        print('forward-backward smoothing')
    
    if doPRC:
        y_pred_fbs_hmm_range = np.zeros(proba_thr.shape)
        for thr in range(100):
            y_pred_fbs = mts.do_fbs(y_pred_class=np.squeeze(proba_thr[:,thr]), kn=kn, blur=4, blur_steps=blur_steps, shift=shift)
            y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
            y_pred_fbs_hmm_range[:,thr] = np.argmax(y_proba_fbs_hmm, axis=1)
        
        sio.savemat(name_classifier  + '_results.mat', {'preds':y_pred_fbs_hmm_range,'gt': gt})
    
    y_pred_fbs = mts.do_fbs(y_pred_class=y_pred_class, kn=kn, blur=4, blur_steps=blur_steps, shift=shift)
    y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
    y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)
    preds_fbs_hmm = y_pred_fbs_hmm
    proba_fbs_hmm = y_proba_fbs_hmm
    dt = time.time() - t
    print('inference took %.2f sec' % dt)

    print('########## pd ##########')
    prf_metrics(y_te[beh_name], preds, beh_name)
    print('########## fbs ##########')
    prf_metrics(y_te[beh_name], preds_fbs_hmm, beh_name)

    return gt, proba, preds, preds_fbs_hmm, proba_fbs_hmm


def train_classifier(project, train_behaviors, drop_behaviors=[]):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # unpack user-provided classification parameters, and use default values for those not provided.
    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)

    if not (clf_params['downsample_rate']==int(clf_params['downsample_rate'])):
        print('Training set downsampling rate must be an integer; reverting to default value of 1.')
        clf_params['downsample_rate'] = 1

    # now create the classifier and give it an informative name:
    classifier = choose_classifier(clf_params)
    classifier_name = cfg['project_name'] + '_' + clf_params['clf_type'] + clf_suffix(clf_params)

    savedir = os.path.join(project, 'behavior', 'trained_classifiers', classifier_name)
    if not os.path.exists(savedir): os.makedirs(savedir)
    print('Training classifier: ' + classifier_name.upper())

    print('loading training data')
    X_tr, y_tr = load_data(project, 'train', train_behaviors, drop_behaviors=drop_behaviors)

    print('loading validation data')
    X_ev, y_ev = load_data(project, 'val', train_behaviors, drop_behaviors=drop_behaviors)

    print('loaded training data: %d X %d - %s ' % (X_tr.shape[0], X_tr.shape[1], list(y_tr.keys())))

    # train each classifier in a loop:
    for b, beh_name in enumerate(train_behaviors):
        print('######################### %s #########################' % beh_name)
        beh_classifier = {'beh_name': beh_name,
                          'beh_id': b + 1,
                          'clf': classifier,
                          'params': clf_params}
        results = do_train(beh_classifier, X_tr, y_tr, X_ev, y_ev, savedir, clf_params['verbose'])

    print('done training!')
    return results


def test_classifier(project, behs, video_path, test_videos, clf_params={}, ver=[7,8], verbose=0, doPRC=0):
    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)

    clf_type = clf_params['clf_type']
    feat_type = clf_params['feat_type']
    do_wnd = clf_params['do_wnd']
    do_cwt = clf_params['do_cwt']

    if clf_params['clf_path_hardcoded'] is not '':
        savedir = clf_params['clf_path_hardcoded']
    else:
        suff = clf_suffix(clf_type, clf_params)
        classifier_name = feat_type + '_' + clf_type + suff
        savedir = os.path.join('trained_classifiers','mars_v1_8',classifier_name)

    print('loading test data...')
    X_te_0, y_te, names = load_data(video_path, test_videos, behs,
                                  ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)
    print('loaded test data: %d X %d - %s ' % (X_te_0.shape[0], X_te_0.shape[1], list(set(y_te))))
    
    T = len(list(y_te.values())[0])
    n_classes = len(behs.keys())
    gt = np.zeros((T, n_classes)).astype(int)
    proba = np.zeros((T, n_classes, 2))
    preds = np.zeros((T, n_classes)).astype(int)
    preds_fbs_hmm = np.zeros((T, n_classes)).astype(int)
    proba_fbs_hmm = np.zeros((T, n_classes, 2))
    beh_list = list()

    for b, beh_name in enumerate(behs.keys()):
        print('predicting behavior %s...' % beh_name)
        beh_list.append(beh_name)
        name_classifier = savedir + 'classifier_' + beh_name
            
            
        print('loading classifier %s' % name_classifier)

        gt[:,b], proba[:, b, :], preds[:, b], preds_fbs_hmm[:, b], proba_fbs_hmm[:, b, :] = \
            do_test(name_classifier, X_te_0, y_te, verbose, doPRC)

    all_pred = assign_labels(proba, beh_list)
    all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, beh_list)

    print('Raw predictions:')
    score_info(gt, all_pred)
    print('Predictions after HMM and forward-backward smoothing:')
    score_info(gt, all_pred_fbs_hmm)
    P = {'0_G': gt,
         '0_Gc': y_te,
         '1_pd': preds,
         '2_pd_fbs_hmm': preds_fbs_hmm,
         '3_proba_pd': proba,
         '4_proba_pd_hmm_fbs': proba_fbs_hmm,
         '5_pred_ass': all_pred,
         '6_pred_fbs_hmm_ass': all_pred_fbs_hmm
         }
    dill.dump(P, open(savedir + 'results.dill', 'wb'))
    sio.savemat(savedir + 'results.mat', P)


def run_classifier(project, behs, video_path, test_videos, test_annot, clf_params={}, save_path=[], ver=[7,8], verbose=0):
    # this code actually saves *.annot files containing the raw predictions of the trained classifier,
    # instead of just giving you the precision and recall. You can load these *.annot files in Bento
    # along with the movies to inspect behavior labels by eye.
    #
    # Unlike test_classifier, this function runs classification on each video separately.

    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)

    clf_type = clf_params['clf_type']
    feat_type = clf_params['feat_type']
    do_wnd = clf_params['do_wnd']
    do_cwt = clf_params['do_cwt']

    suff = clf_suffix(clf_type, clf_params)
    classifier_name = feat_type + '_' + clf_type + suff
    savedir = os.path.join('trained_classifiers', classifier_name)

    for vid in test_videos:
        print('processing %s...' % vid)
        X_te_0, y_te, _ = load_data(video_path, [vid], behs,
                                    ver=ver, feat_type=feat_type, verbose=verbose, do_wnd=do_wnd, do_cwt=do_cwt)

        if not y_te:
            print('skipping this video...\n\n')
            continue

        T = len(list(y_te.values())[0])
        n_classes = len(behs.keys())
        gt = np.zeros((T, n_classes)).astype(int)
        proba = np.zeros((T, n_classes, 2))
        preds = np.zeros((T, n_classes)).astype(int)
        preds_hmm = np.zeros((T, n_classes)).astype(int)
        proba_hmm = np.zeros((T, n_classes, 2))
        preds_fbs_hmm = np.zeros((T, n_classes)).astype(int)
        proba_fbs_hmm = np.zeros((T, n_classes, 2))
        beh_list = list()

        for b, beh_name in enumerate(behs.keys()):
            print('predicting behavior %s...' % beh_name)
            beh_list.append(beh_name)
            name_classifier = savedir + 'classifier_' + beh_name

            gt[:, b], proba[:, b, :], preds[:, b], preds_hmm[:, b], proba_hmm[:, b, :], \
                preds_fbs_hmm[:, b], proba_fbs_hmm[:, b, :] = do_test(name_classifier, X_te_0, y_te, verbose)

        all_pred = assign_labels(proba, beh_list)
        all_pred_hmm = assign_labels(proba_hmm, beh_list)
        all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, beh_list)
        all_gt = assign_labels(gt, beh_list) if b>1 else np.squeeze(gt)

        vname,_ = os.path.splitext(os.path.basename(vid))
        if not save_path:
            save_path = video_path
        map.dump_labels_bento(all_pred, os.path.join(save_path, 'predictions_'+vname+'.annot'),
                              moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
        map.dump_labels_bento(all_pred_hmm, os.path.join(save_path, 'predictions_hmm_' + vname + '.annot'),
                              moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
        map.dump_labels_bento(all_pred_fbs_hmm, os.path.join(save_path, 'predictions_fbs_hmm_' + vname + '.annot'),
                              moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
        print('\n\n')
