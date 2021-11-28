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
import progressbar
import random
import pdb


# warnings.filterwarnings("ignore")
# plt.ioff()

import pdb
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

    with open(os.path.join('behavior_classification','clf_defaults.yaml')) as f:
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


def load_data(project, dataset, train_behaviors, drop_behaviors=[], drop_empty_trials=False, drop_movies=[], do_quicksave=False):
    with open(os.path.join(project, 'project_config.yaml')) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(project, 'behavior', 'config_classifiers.yaml')) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(project, 'behavior', 'behavior_equivalences.yaml')) as f:
        equivalences = yaml.load(f, Loader=yaml.FullLoader)
        if equivalences is None:
            equivalences = {}

    savestr = os.path.join(project, 'behavior', 'behavior_jsons', dataset + '_features')
    if clf_params['do_wnd']:
        savestr += '_wnd.json'
    elif clf_params['do_cwt']:
        savestr += '_cwt.json'
    if not do_quicksave or not os.path.isfile(savestr):
        if dataset in ['train', 'test', 'val']:
            with open(os.path.join(project, 'behavior', 'behavior_jsons', dataset + '_features.json')) as f:
                data = json.load(f)
        else:
            print('dataset must be train, test, or val.')
            return
        for label in train_behaviors:
            if label not in data['vocabulary']:
                print('Error: target behavior "' + label + '" not found in this dataset.\nAvailable labels:')
                print(list(data['vocabulary'].keys()))
                return np.array([]), [], []

        keylist = list(data['sequences'][cfg['project_name']].keys())
        data_stack = []
        annot_raw = []
        if clf_params['verbose']:
            print('applying filters...')
        if clf_params['do_wnd'] or clf_params['do_cwt']:
            bar = progressbar.ProgressBar(widgets=
                                          [progressbar.FormatLabel('Filtering sequence %(value)d'), '/',
                                           progressbar.FormatLabel('%(max)d  '), progressbar.Percentage(), ' -- ', ' [',
                                           progressbar.Timer(), '] ',
                                           progressbar.Bar(), ' (', progressbar.ETA(), ') '], maxval=len(keylist))
            bar.start()
        bump=0
        for i, k in enumerate(keylist):
            if k in drop_movies:
                continue
            for j, entry in enumerate(data['sequences'][cfg['project_name']][k]):
                feats = np.array(entry['features'])
                feats = np.swapaxes(feats, 0, 1)
                feats = mts.clean_data(feats)
                annots = entry['annotations']
                dropflag = False
                for label_name in train_behaviors:
                    if label_name in equivalences.keys():
                        hit_list = [data['vocabulary'][i] for i in equivalences[label_name] if i in data['vocabulary']]
                    else:
                        hit_list = [data['vocabulary'][label_name]]
                    if drop_empty_trials and not any([i in hit_list for i in annots]):
                        dropflag = True
                if dropflag:
                    continue
                if len(annots) != feats.shape[0]:
                    print('Length mismatch: %s %d %d' % (k, len(annots), feats.shape[0]))
                    print('Extra frames will be trimmed from the end of the sequence.')
                    if len(annots) > feats.shape[0]:
                        annots = annots[:feats.shape[0]]
                    else:
                        feats = feats[:len(annots), :, :]

                if np.shape(feats)[1] == 1:
                    feats = feats[:, 0, :]
                else:  # TODO: make >2 mouse case
                    feats = np.concatenate((feats[:, 0, :], feats[:, 1, :]), axis=1)

                if clf_params['do_wnd']:
                    windows = [int(np.ceil(w * cfg['framerate'])*2+1) for w in clf_params['windows']]
                    feats = mts.apply_windowing(feats, windows)
                elif clf_params['do_cwt']:
                    scales = [int(np.ceil(w * cfg['framerate'])) for w in clf_params['wavelets']]
                    feats = mts.apply_wavelet_transform(feats, scales)

                if drop_behaviors:
                    if not isinstance(drop_behaviors, list):
                        drop_behaviors = [drop_behaviors]
                    drop_list = []
                    for d in drop_behaviors:
                        if d in equivalences.keys():
                            drop_list += [data['vocabulary'][i] for i in equivalences[d]]
                        else:
                            drop_list.append(data['vocabulary'][d])
                    keep_inds = [i for i, _annots in enumerate(annots) if _annots not in drop_list]
                    annots = annots[keep_inds]
                    feats = feats[keep_inds, :]
                annot_raw.append(annots)
                data_stack.append(feats)
                bump += len(annots)
            # print('%s   %d' % (k, bump))
            if clf_params['do_wnd'] or clf_params['do_cwt']:
                bar.update(i)
        if clf_params['do_wnd'] or clf_params['do_cwt']:
            bar.finish()
        data_stack = np.concatenate(data_stack, axis=0)
        if do_quicksave:
            savedata = {'data_stack': data_stack.tolist(), 'annot_raw': annot_raw, 'vocabulary': data['vocabulary']}
            with open(savestr, 'w') as f:
                json.dump(savedata, f)
    else:
        with open(savestr) as f:
            data = json.load(f)
        annot_raw = data['annot_raw']
        data_stack = data['data_stack']

    annot_clean = {}
    # print(data['vocabulary'])
    for label_name in train_behaviors:
        annot_clean[label_name] = []
        if label_name in equivalences.keys():
            hit_list = [data['vocabulary'][i] for i in equivalences[label_name] if i in data['vocabulary']]
        else:
            hit_list = [data['vocabulary'][label_name]]
        for a in annot_raw:
            a_clean = [1 if i in hit_list else 0 for i in a]
            if 1 in a_clean:
                annot_clean[label_name] += a_clean
            else:
                annot_clean[label_name] += [-1]*len(a_clean)
    print('done!\n')
    return data_stack, annot_clean, data['vocabulary']


def assign_labels(all_predicted_probabilities, vocabulary):
    # Assigns labels based on the provided probabilities.
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

        # If every behavioral predictor agrees that the current behavior is "other"
        if np.all(onehot_class_predictions == 0):
            # The index here is one past any positive behavior --this is how we code for "other".
            beh_frame = 0
            # How do we get the probability of it being "other?" Since everyone's predicting it, we just take the mean.
            proba_frame = np.mean(predicted_class_probabilities)
            labels_num.append(vocabulary['other'])
        else:
            # If we have positive predictions, we find the probabilities of the positive labels and take the argmax.
            pos = np.where(onehot_class_predictions)[0]
            max_prob = np.argmax(predicted_class_probabilities[pos])

            # This argmax is, by construction, the id for this behavior.
            beh_frame = pos[max_prob]
            proba_frame = predicted_class_probabilities[beh_frame]
            labels_num.append(beh_frame)

    return labels_num


def handle_missing_trials(X, y, drop_empty_trials=False):
    mask = []
    if type(y) is dict:  # if y is a dictionary, all behaviors must be annotated for in a trial
        for k in list(y.keys()):
            mask = min(mask, y[k]) if mask != [] else y[k]
    else:
        mask = y
    if drop_empty_trials:
        X = X[[i != -1 for i in mask]]
        if type(y) is dict:
            for k in list(y.keys()):
                y[k] = np.array(y[k])
                y[k] = np.array(y[k][[i != -1 for i in mask]])
        else:
            y = np.array(y)
            y = np.array(y[[i != -1 for i in mask]])
    else:
        if type(y) is dict:
            for k in list(y.keys()):
                y[k] = np.array(y[k])
                y[k] = np.array([i if i != -1 else 0 for i in y[k]])
        else:
            y = np.array([i if i != -1 else 0 for i in y])  # remove the -1's
    return X, y


def do_train(beh_classifier, X_tr, y_tr_beh, X_ev, y_ev_beh, savedir, verbose=0):
    beh_name = beh_classifier['beh_name']
    clf = beh_classifier['clf']
    clf_params = beh_classifier['params']

    # downsample the data
    X_tr = X_tr[::clf_params['downsample_rate'], :]
    y_tr_beh = y_tr_beh[::clf_params['downsample_rate']]
    if not X_ev == []:
        X_ev = X_ev[::clf_params['downsample_rate'], :]
        y_ev_beh = y_ev_beh[::clf_params['downsample_rate']]

    # scale the data
    gc.collect()
    if(verbose):
        print('fitting preprocessing parameters...')
    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    if not X_ev == []:
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
    if not X_ev == []:
        eval_set = [(X_ev, y_ev_beh)]
        print('    eval set: %d positive / %d total (%d %%)' % (
            sum(y_ev_beh), len(y_ev_beh), 100 * sum(y_ev_beh) / len(y_ev_beh)))
        if clf_params['early_stopping']:
            if verbose:
                print('  + early stopping')
            clf.fit(X_tr, y_tr_beh, eval_set=eval_set,
                    early_stopping_rounds=clf_params['early_stopping'], verbose=True)
        else:
            clf.fit(X_tr, y_tr_beh, eval_set=eval_set, eval_metric='aucpr', verbose=True)
        results = clf.evals_result()
    else:
        if verbose:
            print('  no validation set included')
        clf.fit(X_tr, y_tr_beh, eval_metric='aucpr', verbose=True)
        results = []

    beh_classifier.update({'clf': clf,
                           'scaler': scaler})
    dill.dump(beh_classifier, open(os.path.join(savedir, 'classifier_' + beh_name), 'wb'))
    return results, beh_classifier


def do_train_smooth(beh_classifier, X_tr, y_tr_beh, savedir, verbose=False):
    beh_name = beh_classifier['beh_name']
    clf = beh_classifier['clf']
    scaler = beh_classifier['scaler']
    clf_params = beh_classifier['params']
    # set some parameters for post-classification smoothing:
    kn = clf_params['smk_kn']
    blur_steps = clf_params['blur'] ** 2
    shift = clf_params['shift']
    # get the labels for the current behavior
    t = time.time()

    # evaluate on training set
    if (verbose):
        print('evaluating on the training set...')
    y_pred_proba = np.zeros((len(y_tr_beh), 2))
    gen = Batch(range(len(y_tr_beh)), lambda x: x % 1e5 == 0, 1e5)
    for i in gen:
        inds = list(i)
        X_tr_s = scaler.transform(X_tr[inds])
        pd_proba_tmp = (clf.predict_proba(X_tr_s))
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
    for s in range(blur_steps):
        y_fbs = signal.convolve(np.r_[y_fbs[0], y_fbs, y_fbs[-1]], kn / kn.sum(), 'valid')
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
    print('performance on training set:')
    precision, recall, f_measure = prf_metrics(y_tr_beh, y_pred_fbs_hmm, beh_name)

    beh_classifier.update({'clf': clf,
                           'scaler': scaler,
                           'precision': precision,
                           'recall': recall,
                           'f_measure': f_measure,
                           'hmm_bin': hmm_bin,
                           'hmm_fbs': hmm_fbs})
    dill.dump(beh_classifier, open(os.path.join(savedir, 'classifier_' + beh_name), 'wb'))


def do_test(name_classifier, X_te, y_te_beh, verbose=0, doPRC=0):
    classifier = joblib.load(name_classifier)
    # unpack the classifier
    beh_name = classifier['beh_name']
    clf = classifier['bag_clf'] if 'bag_clf' in classifier.keys() else classifier['clf']
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
    len_y = len(y_te_beh)
    gt = y_te_beh

    # predict probabilities:
    y_pred_proba = clf.predict_proba(X_te)
    proba = y_pred_proba

    if doPRC:
        # compute predictions as a function of threshold to make P-R curves!
        p_pos = np.squeeze(proba[:, 1])
        proba_thr = np.zeros((p_pos.size, 100))
        for thr in range(100):
            proba_thr[:, thr] = np.array([1 if i > (thr/100.) else 0 for i in p_pos])

    y_pred_class = np.argmax(y_pred_proba, axis=1)
    preds = y_pred_class
    # forward-backward smoothing:
    if doPRC:
        y_pred_fbs_hmm_range = np.zeros(proba_thr.shape)
        for thr in range(100):
            y_pred_fbs = mts.do_fbs(y_pred_class=np.squeeze(proba_thr[:, thr]), kn=kn, blur=4, blur_steps=blur_steps, shift=shift)
            y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
            y_pred_fbs_hmm_range[:, thr] = np.argmax(y_proba_fbs_hmm, axis=1)
        sio.savemat(name_classifier + '_results.mat', {'preds': y_pred_fbs_hmm_range, 'gt': gt})

    y_pred_fbs = mts.do_fbs(y_pred_class=y_pred_class, kn=kn, blur=4, blur_steps=blur_steps, shift=shift)
    y_proba_fbs_hmm = hmm_fbs.predict_proba(y_pred_fbs.reshape((-1, 1)))
    y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)
    preds_fbs_hmm = y_pred_fbs_hmm
    proba_fbs_hmm = y_proba_fbs_hmm
    dt = time.time() - t
    print('inference took %.2f sec' % dt)

    return gt, proba, preds, preds_fbs_hmm, proba_fbs_hmm


def train_classifier(project, train_behaviors, drop_behaviors=[], drop_empty_trials=False,
                     drop_movies=[], do_quicksave=False):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # unpack user-provided classification parameters, and use default values for those not provided.
    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)
        if 'smk_kn' in clf_params.keys():
            clf_params['smk_kn'] = np.array(clf_params['smk_kn'])
    if not (clf_params['downsample_rate'] == int(clf_params['downsample_rate'])):
        print('Training set downsampling rate must be an integer; reverting to default value of 1.')
        clf_params['downsample_rate'] = 1
    clf_params['project_config'] = cfg

    print('loading training data...')
    X_tr, y_tr, vocab = load_data(project, 'train', train_behaviors,
                                  drop_behaviors=drop_behaviors,
                                  drop_empty_trials=drop_empty_trials,
                                  drop_movies=drop_movies,
                                  do_quicksave=do_quicksave)
    if X_tr.size == 0:
        return []
    print('loading validation data...')
    X_ev, y_ev, _ = load_data(project, 'val', train_behaviors,
                              drop_behaviors=drop_behaviors,
                              drop_empty_trials=drop_empty_trials,
                              drop_movies=drop_movies,
                              do_quicksave=do_quicksave)
    print('loaded training data: %d X %d - %s ' % (X_tr.shape[0], X_tr.shape[1], list(y_tr.keys())))

    # now create the classifier and give it an informative name:
    classifier = choose_classifier(clf_params)
    classifier_name = cfg['project_name'] + '_' + clf_params['clf_type'] + clf_suffix(clf_params)
    savedir = os.path.join(project, 'behavior', 'trained_classifiers', classifier_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print('Training classifier: ' + classifier_name.upper())
    # train each classifier in a loop:
    for beh_name in train_behaviors:
        print('######################### %s #########################' % beh_name)
        # drop trials missing annotations for our behavior of interest
        X_tr_beh, y_tr_beh = handle_missing_trials(X_tr, y_tr[beh_name], drop_empty_trials=drop_empty_trials)
        if X_ev != []:
            X_ev_beh, y_ev_beh = handle_missing_trials(X_ev, y_ev[beh_name], drop_empty_trials=drop_empty_trials)
        else:
            X_ev_beh = []
            y_ev_beh = []

        # shuffle in blocks of 2000 frames:
        blocksize = 2000
        num_blocks_tr = int(np.ceil(len(y_tr_beh) / blocksize))
        blockorder_tr = list(range(num_blocks_tr))
        random.shuffle(blockorder_tr)
        newinds_tr = list([j + blocksize * i for i in blockorder_tr for j in np.arange(blocksize)])
        X_tr_beh = np.array([X_tr_beh[i, :] for i in newinds_tr if i < len(y_tr_beh)])
        y_tr_beh = np.array([y_tr_beh[i] for i in newinds_tr if i < len(y_tr_beh)])
        if X_ev != []:
            num_blocks_ev = int(np.ceil(len(y_ev_beh) / blocksize))
            blockorder_ev = list(range(num_blocks_ev))
            random.shuffle(blockorder_ev)
            newinds_ev = list([j + blocksize * i for i in blockorder_ev for j in np.arange(blocksize)])
            X_ev_beh = np.array([X_ev_beh[i, :] for i in newinds_ev if i < len(y_ev_beh)])
            y_ev_beh = np.array([y_ev_beh[i] for i in newinds_ev if i < len(y_ev_beh)])

        # cut to our target training set size
        bouts_tr = sum([(i != 0 and j == 0) for i, j in zip(y_tr_beh[:-1], y_tr_beh[1:])])
        print('training using %d positive frames (%s bouts)' % (sum(y_tr_beh!=0), bouts_tr))

        beh_classifier = {'beh_name': beh_name,
                          'beh_id': vocab[beh_name],
                          'clf': classifier,
                          'params': clf_params}
        results = do_train(beh_classifier,
                           X_tr_beh, y_tr_beh,
                           X_ev_beh, y_ev_beh,
                           savedir, verbose=clf_params['verbose'])
        do_train_smooth(beh_classifier,
                        X_tr_beh, y_tr_beh,
                        savedir, verbose=clf_params['verbose'])

        print('done training!')
    return results


def test_classifier(project, test_behaviors, drop_behaviors=[], drop_empty_trials=False,
                    do_quicksave=False):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # unpack user-provided classification parameters, and use default values for those not provided.
    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)
        if 'smk_kn' in clf_params.keys():
            clf_params['smk_kn'] = np.array(clf_params['smk_kn'])

    print('loading test data...')
    X_te, y_te, vocab = load_data(project, 'test', test_behaviors,
                                    drop_behaviors=drop_behaviors, do_quicksave=do_quicksave)
    print('loaded test data: %d X %d - %s ' % (X_te.shape[0], X_te.shape[1], list(set(y_te))))
    X_te, y_te = handle_missing_trials(X_te, y_te, drop_empty_trials=drop_empty_trials)

    classifier_name = cfg['project_name'] + '_' + clf_params['clf_type'] + clf_suffix(clf_params)
    savedir = os.path.join(project, 'behavior', 'trained_classifiers', classifier_name)
    T = len(list(y_te.values())[0])
    n_classes = max([vocab[b] for b in list(vocab.keys())])+1
    gt = np.zeros((T, n_classes)).astype(int)
    proba = np.zeros((T, n_classes, 2))
    preds = np.zeros((T, n_classes)).astype(int)
    preds_fbs_hmm = np.zeros((T, n_classes)).astype(int)
    proba_fbs_hmm = np.zeros((T, n_classes, 2))
    print('loading classifiers from %s' % savedir)
    for b, beh_name in enumerate(test_behaviors):
        print('predicting %s...' % beh_name)
        name_classifier = os.path.join(savedir, 'classifier_' + beh_name)

        gt[:, vocab[beh_name]], proba[:, vocab[beh_name], :], preds[:, vocab[beh_name]],\
        preds_fbs_hmm[:, vocab[beh_name]], proba_fbs_hmm[:, vocab[beh_name], :] = \
            do_test(name_classifier, X_te, y_te[beh_name],
                    verbose=clf_params['verbose'], doPRC=True)
    all_pred = assign_labels(proba, vocab)
    all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, vocab)
    gt = np.argmax(gt, axis=1)

    print(' ')
    print('Classifier performance:')
    score_info(gt, all_pred_fbs_hmm, vocab)
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


# def run_classifier(project, test_behaviors):
#     # this code actually saves *.annot files containing the raw predictions of the trained classifier,
#     # instead of just giving you the precision and recall. You can load these *.annot files in Bento
#     # along with the movies to inspect behavior labels by eye.
#     # Unlike test_classifier, this function runs classification on each video separately.
#     config_fid = os.path.join(project, 'project_config.yaml')
#     with open(config_fid) as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
#     # unpack user-provided classification parameters, and use default values for those not provided.
#     config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
#     with open(config_fid) as f:
#         clf_params = yaml.load(f, Loader=yaml.FullLoader)
#         if 'smk_kn' in clf_params.keys():
#             clf_params['smk_kn'] = np.array(clf_params['smk_kn'])
#
#     classifier_name = cfg['project_name'] + '_' + clf_params['clf_type'] + clf_suffix(clf_params)
#     savedir = os.path.join('trained_classifiers', 'mars_v1_8', classifier_name)
#     print('loading test data...')
#     X_te_0, y_te, names = load_data(project, 'test', test_behaviors)
#     print('loaded test data: %d X %d - %s ' % (X_te_0.shape[0], X_te_0.shape[1], list(set(y_te))))
#
#     for vid in test_videos:
#         print('processing %s...' % vid)
#         X_te_0, y_te, _ = load_data(video_path, [vid], test_behaviors)
#         if not y_te:
#             print('skipping this video...\n\n')
#             continue
#
#         T = len(list(y_te.values())[0])
#         n_classes = len(test_behaviors.keys())
#         gt = np.zeros((T, n_classes)).astype(int)
#         proba = np.zeros((T, n_classes, 2))
#         preds = np.zeros((T, n_classes)).astype(int)
#         preds_hmm = np.zeros((T, n_classes)).astype(int)
#         proba_hmm = np.zeros((T, n_classes, 2))
#         preds_fbs_hmm = np.zeros((T, n_classes)).astype(int)
#         proba_fbs_hmm = np.zeros((T, n_classes, 2))
#         beh_list = list()
#         for b, beh_name in enumerate(test_behaviors.keys()):
#             print('predicting behavior %s...' % beh_name)
#             beh_list.append(beh_name)
#             name_classifier = savedir + 'classifier_' + beh_name
#             gt[:, b], proba[:, b, :], preds[:, b], preds_hmm[:, b], proba_hmm[:, b, :], \
#                 preds_fbs_hmm[:, b], proba_fbs_hmm[:, b, :] = do_test(name_classifier, X_te_0, y_te, clf_params['verbose'])
#         all_pred = assign_labels(proba, beh_list)
#         all_pred_hmm = assign_labels(proba_hmm, beh_list)
#         all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, beh_list)
#         all_gt = assign_labels(gt, beh_list) if b>1 else np.squeeze(gt)
#
#         vname,_ = os.path.splitext(os.path.basename(vid))
#         if not save_path:
#             save_path = video_path
#         map.dump_labels_bento(all_pred, os.path.join(save_path, 'predictions_'+vname+'.annot'),
#                               moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
#         map.dump_labels_bento(all_pred_hmm, os.path.join(save_path, 'predictions_hmm_' + vname + '.annot'),
#                               moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
#         map.dump_labels_bento(all_pred_fbs_hmm, os.path.join(save_path, 'predictions_fbs_hmm_' + vname + '.annot'),
#                               moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
#         print('\n\n')
