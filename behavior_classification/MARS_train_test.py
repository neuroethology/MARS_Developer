from __future__ import division
import os,sys,fnmatch
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import binarize
import dill
import time
from sklearn.ensemble import BaggingClassifier
from hmmlearn import hmm
from scipy import signal
import copy
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import MARS_annotation_parsers as map
import gc
import MARS_ts_util as mts
import joblib
from MARS_clf_helpers import *
from seqIo import *
import scipy.io as sio
import pdb


# warnings.filterwarnings("ignore")
# plt.ioff()


lcat = lambda L: [i for j in L for i in j]
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


def get_beh_dict(behavior):
    behs = {'sniff_face':       ['sniffface', 'snifface', 'sniff-face', 'sniff_face', 'head-investigation','facesniffing'],
            'sniff_genital':    ['sniffurogenital','sniffgenitals','sniff_genitals','sniff-genital','sniff_genital',
                                    'anogen-investigation'],
            'sniff_body':       ['sniff_body','sniffbody','bodysniffing','body-investigation','socialgrooming',
                                    'sniff-body','closeinvestigate','closeinvestigation','investigation'],
            'closeinvestigation': ['sniffface', 'snifface', 'sniff-face', 'sniff_face', 'head-investigation','facesniffing',
                                    'sniffurogenital','sniffgenitals','sniff_genitals','sniff-genital','sniff_genital',
                                    'anogen-investigation','sniff_body', 'sniffbody', 'bodysniffing', 'body-investigation',
                                    'socialgrooming','sniff-body', 'closeinvestigate', 'closeinvestigation', 'investigation',
                                    'investigate','first_inv','aggressive_investigation','attack_attempt','mount_attempt','dom_mount_attempt'],
            'mount':            ['mount','aggressivemount','intromission','dom_mount','ejaculate','ejuculation'],
            'attack':           ['attack','attempted_attack','chase']}
    
    if behavior.lower() in behs.keys():
        return {behavior: behs[behavior.lower()]}
    else:
        print('I didn''t recognize that behavior, aborting')
        return {}


def load_default_parameters():
    default_params = {'clf_type': 'xgb',
                      'feat_type': 'top',  # keep this to just top for now
                      'downsample_rate': 1,  # temporal downsampling applied to training data
                      'smk_kn': np.array([0.5, 0.25, 0.5]),
                      'blur': 4,
                      'shift': 4,
                      'do_wnd': False,
                      'do_cwt': False,
                      'early_stopping': 20, # set to zero to turn off early stopping
                      'clf_path_hardcoded': ''
                      }

    # in addition to these parameters, you can also store classifier-specific parameters in clf_params.
    # default values for those are defined below in choose_classifier.

    return default_params


def clf_suffix(clf_type='xgb', clf_params=dict()):
    
    if clf_type.lower() == 'mlp':
        suff = '_layers' + '-'.join(clf_params['hidden_layer_sizes']) if 'hidden_layer_sizes' in clf_params.keys() else ''
        suff = suff + '/'
    else:
        if not clf_type.lower() == 'xgb':
            print('Unrecognized classifier type %s, defaulting to XGBoost!' % clf_type)

        suff = '_es' + str(clf_params['early_stopping']) if 'early_stopping' in clf_params.keys() else ''
        suff = suff + '_depth' + str(clf_params['max_depth']) if 'max_depth' in clf_params.keys() else suff
        suff = suff + '_child' + str(clf_params['min_child_weight']) if 'min_child_weight' in clf_params.keys() else suff
        suff = suff + '_wnd' if clf_params['do_wnd'] else suff
        suff = suff + '_cwt' if clf_params['do_cwt'] else suff
        suff = suff + '_' + str(clf_params['user_suff']) if 'user_suff' in clf_params.keys() else suff
        suff = suff + '/'
    
    return suff


def choose_classifier(clf_type='xgb', clf_params=dict()):
    
    MLPdefaults = {'hidden_layer_sizes': (256, 512),
                   'learning_rate_init': 0.001,
                   'learning_rate': 'adaptive',
                   'max_iter': 100000,
                   'alpha': 0.0001}

    XGBdefaults = {'n_estimators': 2000,
                   'eta': 0.1,
                   'max_depth': 9,
                   'gamma': 1,
                   'min_child_weight': 4,
                   'subsample': 0.8,
                   'scale_pos_weight': 1,
                   'colsample_bytree': 0.8,
                   'max_bin': 256,
                   # 'objective': 'binary:logistic',
                   'tree_method': 'hist',
                   'silent': 1,
                   'seed': 33
                   }

    # insert defaults for other classifier types here!

    if clf_type.lower() == 'mlp':
        for k in MLPdefaults.keys():
            if not k in clf_params.keys():
                clf_params[k] = MLPdefaults[k]

        mlp = MLPClassifier(solver='adam',
                            alpha=clf_params['alpha'],
                            hidden_layer_sizes=clf_params['hidden_layer_sizes'],
                            learning_rate=clf_params['learning_rate'],
                            max_iter=clf_params['max_iter'],
                            learning_rate_init=clf_params['learning_rate_init'],
                            verbose=0,
                            random_state=1
                            )
        clf = BaggingClassifier(mlp, max_samples=.1, n_jobs=3, random_state=7, verbose=0)

    else:
        if not clf_type.lower() == 'xgb':
            print('Unrecognized classifier type %s, defaulting to XGBoost!' % clf_type)

        for k in XGBdefaults.keys():
            if not k in clf_params.keys():
                clf_params[k] = XGBdefaults[k]

        clf = XGBClassifier(n_estimators=clf_params['n_estimators'],
                            eta=clf_params['eta'],
                            max_depth=clf_params['max_depth'],
                            gamma=clf_params['gamma'],
                            min_child_weight=clf_params['min_child_weight'],
                            subsample=clf_params['subsample'],
                            scale_pos_weight=clf_params['scale_pos_weight'],
                            colsample_bytree=clf_params['colsample_bytree'],
                            max_bin=clf_params['max_bin'],
                            objective=clf_params['objective'],
                            tree_method=clf_params['tree_method'],
                            silent=clf_params['silent'],
                            seed=clf_params['seed']
                            )

    return clf


def quick_loader(filename, keep_labels):
    temp = np.load(filename, allow_pickle=True)
    data = temp['data']
    names = temp['names']
    behList = temp['behList']
    all_keep = []
    for i in keep_labels.keys():
        all_keep += keep_labels[i] 
    labels = []
    for beh in behList:
        labels += map.merge_channels(beh['behs_bout'], beh['keys'], len(beh['behs_frame']), target_behaviors = all_keep)
    return data, names, labels


def load_data(video_path, video_list, keep_labels, ver=[7, 8], feat_type='top', verbose=0, do_wnd=False, do_cwt=False, save_after=True):
    data = []
    labels = []
    behList = []
    feature_savefile = '_'.join(list(set([os.path.dirname(i) for i in video_list]))) + '_' + feat_type
    feature_savefile = feature_savefile+'_wnd' if do_wnd else feature_savefile+'_cwt' if do_cwt else feature_savefile
    feature_savefile = feature_savefile + '_v1_' + str(ver[-1])

    if os.path.exists(os.path.join(video_path, feature_savefile + '.npz')):
        if verbose:
            print('    quick-loading from file')
        data, names, labels = quick_loader(os.path.join(video_path, feature_savefile + '.npz'), keep_labels)

    else:
        for v in video_list:
            vbase = os.path.basename(v)
            vbase2 = '_'.join(vbase.split('_')[:-1])
            vid = []
            seq = []

            for file in os.listdir(os.path.join(video_path, v)):
                if (fnmatch.fnmatch(file, '*.txt') and not fnmatch.fnmatch(file, '*OutputLikelihood.txt')) or fnmatch.fnmatch(file, '*.annot'):
                    ann = file
                elif fnmatch.fnmatch(file, '*.seq'):
                    seq = os.path.join(video_path, v, file)

            # we load exact frame timestamps for *.annot files to make sure we get the time->frame conversion correct
            #if fnmatch.fnmatch(ann, '*.annot') and seq:
            #    sr = seqIo_reader(seq)
            #    timestamps = sr.getTs()
            #else:
            timestamps = []

            for version in ver:
                fstr = os.path.join(video_path, v, vbase + '_raw_feat_%s_v1_%d.npz' % (feat_type, version))
                fstr2 = os.path.join(video_path, v, vbase2 + '_raw_feat_%s_v1_%d.npz' % (feat_type, version))
                if os.path.isfile(fstr):
                    vid = np.load(open(fstr, 'rb'))
                    if verbose:
                        print('loaded file: ' + os.path.basename(fstr))
                elif os.path.isfile(fstr2):
                    vid = np.load(open(fstr2, 'rb'))
                    if verbose:
                        print('loaded file: ' + os.path.basename(fstr2))

            if not vid:
                print('Feature file not found for %s' % vbase)
            else:
                names = vid['features'].tolist()
                if 'data_smooth' in vid.keys():
                    d = vid['data_smooth']
                    d = mts.clean_data(d)
                    d = mts.normalize_pixel_data(d,'top')
                    d = mts.clean_data(d)
                    n_feat = d.shape[2]

                    # we remove some features that have the same value for both mice (hardcoded for now, shaaame)
                    # featToKeep = lcat([range(39), range(49, 58), [59, 61, 62, 63], range(113, n_feat)])
                    featToKeep = list(flatten([range(39), range(42, 58), 59, 61, 62, 63, range(113, n_feat)]))

                    d = np.hstack((d[0, :, :], d[1, :, featToKeep].transpose()))
                    names_r = np.array(['1_' + f for f in names])
                    names_i = np.array(['2_' + names[f] for f in featToKeep])
                    names = np.concatenate((names_r, names_i)).tolist()

                    # for this project, we also remove raw pixel-based features to keep things simple
                    # d = mts.remove_pixel_data(d, 'top')
                    # names = mts.remove_pixel_data(names, 'top')

                else: # this is for features created with MARS_feature_extractor (which currently doesn't build data_smooth)
                    d = vid['data']
                    d = mts.clean_data(d)

                if do_wnd:
                    d = mts.apply_windowing(d)
                    feats_wnd_names=[]
                    fn = ['min','max','mean','std']
                    win=[3,11,21]
                    for f in names:
                        for w in win:
                            for x in fn:
                                feats_wnd_names.append('_'.join([f,str(w),x]))
                    names = feats_wnd_names
                    
                elif do_cwt:
                    d = mts.apply_wavelet_transform(d)
                data.append(d)
                
                beh = map.parse_annotations(os.path.join(video_path, v, ann), timestamps=timestamps)
                if len(beh['behs_frame']) == 1+d.shape[0]: # this happens sometimes?
                    beh['behs_frame'] = beh['behs_frame'][:-1]
                all_keep = []
                for i in keep_labels.keys():
                    all_keep += keep_labels[i] 
                labels += map.merge_channels(beh['behs_bout'], beh['keys'], len(beh['behs_frame']), target_behaviors = all_keep)
                behList += [beh]

                if len(beh['behs_frame']) != d.shape[0]:
                    print('Length mismatch: %s %d %d' % (v, len(beh['behs_frame']), d.shape[0]))
        if not data:
            print('No feature files found')
            return [], [], []
        if (verbose):
            print('all files loaded')

        data = np.concatenate(data, axis=0)

    if verbose:
        print('    processing annotation files')
    y = {}
    for label_name in keep_labels.keys():
        y_temp = np.array([]).astype(int)
        for i in labels:
            y_temp = np.append(y_temp, 1) if i in keep_labels[label_name] else np.append(y_temp, 0)
        y[label_name] = y_temp

    if save_after and not os.path.exists(os.path.join(video_path, feature_savefile + '.npz')):
        if verbose:
            print('    saving processed data for future use')
        saveData = {'data': data, 'names': names, 'behList': behList}
        np.savez(os.path.join(video_path, feature_savefile), **saveData)

    print('done!\n')

    return data, y, names


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


def train_classifier(behs, video_path, train_videos, eval_videos=[], clf_params={}, ver=[7, 8], verbose=0):

    # unpack user-provided classification parameters, and use default values for those not provided.
    default_params = load_default_parameters()
    for k in default_params.keys():
        if k not in clf_params.keys():
            clf_params[k] = default_params[k]

    # determine which classifier type we're training, which features we're using, and what windowing to use:
    clf_type = clf_params['clf_type']
    feat_type = clf_params['feat_type']
    do_wnd = clf_params['do_wnd']
    do_cwt = clf_params['do_cwt']

    if not (clf_params['downsample_rate']==int(clf_params['downsample_rate'])):
        print('Training set downsampling rate must be an integer; reverting to default value.')
        clf_params['downsample_rate'] = default_params['downsample_rate']

    # now create the classifier and give it an informative name:
    classifier = choose_classifier(clf_type, clf_params)
    suff = clf_suffix(clf_type, clf_params)
    classifier_name = feat_type + '_' + clf_type + suff
    folder = 'mars_v1_' + str(ver[-1])
    savedir = os.path.join('trained_classifiers',folder, classifier_name)
    if not os.path.exists(savedir): os.makedirs(savedir)
    print('Training classifier: ' + classifier_name.upper())

    f = open(savedir + '/log_selection.txt', 'w')

    print('loading training data')
    X_tr, y_tr, features = load_data(video_path, train_videos, behs, ver=ver, feat_type=feat_type,
                                     verbose=verbose,do_wnd=do_wnd, do_cwt=do_cwt)

    if eval_videos:
        print('loading validation data')
        X_ev, y_ev, features = load_data(video_path, eval_videos, behs, ver=ver, feat_type=feat_type,
                                         verbose=verbose,do_wnd=do_wnd, do_cwt=do_cwt)
    else:
        X_ev = []
        y_ev = []
    print('loaded training data: %d X %d - %s ' % (X_tr.shape[0], X_tr.shape[1], list(y_tr.keys())))

    # train each classifier in a loop:
    for b,beh_name in enumerate(behs.keys()):
        print('######################### %s #########################' % beh_name)
        beh_classifier = {'beh_name': beh_name,
                          'beh_id': b + 1,
                          'clf': classifier,
                          'params': clf_params}
        results = do_train(beh_classifier, X_tr, y_tr, X_ev, y_ev, savedir, verbose)

    print('done training!')
    return results


def test_classifier(behs, video_path, test_videos, clf_params={}, ver=[7,8], verbose=0, doPRC=0):

    default_params = load_default_parameters()
    for k in default_params.keys():
        if k not in clf_params.keys():
            clf_params[k] = default_params[k]

    clf_type = clf_params['clf_type']
    feat_type = clf_params['feat_type']
    do_wnd = clf_params['do_wnd']
    do_cwt = clf_params['do_cwt']
    print(clf_params)

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


def run_classifier(behs, video_path, test_videos, test_annot, clf_params={}, save_path=[], ver=[7,8], verbose=0):
    # this code actually saves *.annot files containing the raw predictions of the trained classifier,
    # instead of just giving you the precision and recall. You can load these *.annot files in Bento
    # along with the *.seq movies to inspect behavior labels by eye.
    #
    # Unlike test_classifier, this function runs classification on each video separately.

    default_params = load_default_parameters()
    for k in default_params.keys():
        if k not in clf_params.keys():
            clf_params[k] = default_params[k]

    clf_type = clf_params['clf_type']
    feat_type = clf_params['feat_type']
    do_wnd = clf_params['do_wnd']
    do_cwt = clf_params['do_cwt']
    print(clf_params)

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
