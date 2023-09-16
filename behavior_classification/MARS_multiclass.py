from __future__ import division
import glob
import os
import pdb
import xgboost
import dill

from behavior_classification.MARS_feature_extractor import *
from behavior_classification.MARS_train_test import *
from behavior_classification.behavior_helpers import *


def extract_all_features(project, progress_bar_sig=''):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # generate all valid features for the dataset so we can pull the ones we need for each classifier.
    # in the future we could just pull the ones we want from the list of classifiers in classifier_directory.
    feats = generate_valid_feature_list(cfg)
    use_cam = list(feats.keys())[0]
    mouse_list = list(feats[use_cam].keys())
    feat_list=[]
    for mouse in mouse_list:
        feat_list = feat_list + list(feats[use_cam][mouse].keys())

    for key in ['train', 'test', 'val']:
        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_data.json')) as f:
            data = json.load(f)

        feats = {'feature_names': [], 'sequences': {cfg['project_name']: {}}}
        keylist = list(data['sequences'][cfg['project_name']].keys())
        for i, k in enumerate(keylist):
            feats['sequences'][cfg['project_name']][k] = []
            for j, entry in enumerate(data['sequences'][cfg['project_name']][k]):
                print('%s video %i/%i: %s clip %i/%i' % (key, i+1, len(keylist), k, j+1, len(data['sequences'][cfg['project_name']][k])))
                feat_dict = run_feature_extraction(entry, cfg, use_grps=feat_list)
                if not feat_dict:
                    print('skipping for no feats, something went wrong')
                else:
                    feats['feature_names'] = feat_dict['features']
                    feats['vocabulary'] = data['vocabulary']
                    feats['sequences'][cfg['project_name']][k].append({'features': feat_dict['data'].tolist(), 'annotations': entry['annotations']})

        with open(os.path.join(project, 'behavior', 'behavior_jsons', key + '_features.json'), 'w') as f:
            json.dump(feats, f)


def train_multiclass(project, classifier_directory, drop_behaviors=[], redo_feature_extraction=True):
    # global proba_train, proba_eval, proba_val

    if redo_feature_extraction:
        extract_all_features(project)

    # load information in the project config file
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # first, run binary classifiers on all the training and validation data---------------------------------------------
    savedir = os.path.join(project, 'behavior', 'trained_classifiers', classifier_directory)
    trained_classifiers = glob.glob(os.path.join(savedir, 'classifier_*'))
    trained_behaviors = [x.replace(os.path.join(savedir, 'classifier_'), '') for x in trained_classifiers if 'results' not in x]
    feat_order = []
    T = []

    print('loading classifiers from %s' % savedir)
    print(trained_behaviors)

    for beh_name in trained_behaviors:
        print('\npredicting %s...' % beh_name)
        name_classifier = os.path.join(savedir, 'classifier_' + beh_name)

        # load all three datasets, making sure features are in the right order
        clf = joblib.load(name_classifier)
        clf_params = clf['params']
        if feat_order != clf_params['feature_order']:
            print('reorganizing data...')
            feats_tr, y_tr, vocab, _ = load_data(project, 'train', trained_behaviors, drop_behaviors=drop_behaviors, do_quicksave=False, target_feature_order=clf_params['feature_order'])
            feats_ev, y_ev, _, _ = load_data(project, 'val', trained_behaviors,
                                         drop_behaviors=drop_behaviors, do_quicksave=False,
                                         target_feature_order=clf_params['feature_order'])
            feats_te, y_te, _, _ = load_data(project, 'test', trained_behaviors,
                                         drop_behaviors=drop_behaviors, do_quicksave=False,
                                         target_feature_order=clf_params['feature_order'])
            print('reorganized')
            print(vocab)

        X_tr = feats_tr
        X_ev = feats_ev
        X_te = feats_te

        # initialize the arrays that will track our labels and our classifier output
        if not T:
            n_classes = max(vocab.values())+1
            # initialize for training set
            T = len(list(y_tr.values())[0])
            gt_train = np.zeros((T, n_classes)).astype(int)
            proba_train = np.zeros((T, n_classes, 2))
            # initialize for validation set
            T = len(list(y_ev.values())[0])
            gt_eval = np.zeros((T, n_classes)).astype(int)
            proba_eval = np.zeros((T, n_classes, 2))
            # initialize for test set
            T = len(list(y_te.values())[0])
            gt_test = np.zeros((T, n_classes)).astype(int)
            proba_test = np.zeros((T, n_classes, 2))

        # do_test actually runs the classifier- call it once for the training set and once for validation.
        # > gt_train/gt_ev/gt_te are the actual ground truth annotations for each non-omitted frame of the training,
        #   validation, and test sets, respectively. They're one-hot encoded, meaning they are matrices of (time x
        #   behaviors) where you have gt_train[time,b] = 1 if behavior number <b> happens on frame <time>.
        #
        # > proba_train/proba_eval/proba_test are the classifier-predicted probabilities that each given behavior
        #   occurred in the training, validation, and test sets, respectively. These start out as matrices of
        #   (time x behaviors x 2), but the last dimension is just to store both p and (1-p), so we get rid of it below.


        print('running classifier...')
        b = vocab[beh_name]
        gt_train[:, b], proba_train[:, b, :], _, _, _ = do_test(name_classifier, X_tr, y_tr[beh_name], verbose=False, doPRC=False)
        gt_eval[:, b], proba_eval[:, b, :], _, _, _     = do_test(name_classifier, X_ev, y_ev[beh_name], verbose=False, doPRC=False)
        gt_test[:, b], proba_test[:, b, :], _, _, _ = do_test(name_classifier, X_te, y_te[beh_name], verbose=False, doPRC=False)

    # we don't need to keep both probability channels
    proba_train = proba_train[:, :, 1]
    proba_eval    = proba_eval[:, :, 1]
    proba_test    = proba_test[:, :, 1]

    # and let's make our one-hot encodings into vectors
    for b in range(n_classes):
        gt_train[:, b] = list(np.array(gt_train[:, b]) * b)
        gt_eval[:, b]  = list(np.array(gt_eval[:, b]) * b)
        gt_test[:, b]  = list(np.array(gt_test[:, b]) * b)
    gt_train = list(np.amax(np.array(gt_train), axis=1))
    gt_eval  = list(np.amax(np.array(gt_eval), axis=1))
    gt_test  = list(np.amax(np.array(gt_test), axis=1))

    print('applying windowing...')
    # if you wanted to, you could add some temporal windowing here to help classifier performance-----------------------
    windows = [int(np.ceil(w * cfg['framerate']) * 2 + 1) for w in clf_params['windows']]
    proba_train = mts.apply_windowing(proba_train, windows)
    proba_eval = mts.apply_windowing(proba_eval, windows)
    proba_test = mts.apply_windowing(proba_test, windows)

    # shuffle the training and validation sets--------------------------------------------------------------------------
    print('Training multi-class classifier!')

    # shuffle training set in blocks of 2000 frames:
    blocksize = 2000
    num_blocks_tr = int(np.ceil(len(gt_train) / blocksize))
    blockorder_tr = list(range(num_blocks_tr))
    random.shuffle(blockorder_tr)
    newinds_tr = list([j + blocksize * i for i in blockorder_tr for j in np.arange(blocksize)])
    # apply shuffle
    X_train_multiclass = np.array([proba_train[i, :] for i in newinds_tr if i < len(gt_train)])
    y_train_multiclass = np.array([gt_train[i] for i in newinds_tr if i < len(gt_train)])

    # shuffle validation set in blocks of 2000 frames:
    num_blocks_eval = int(np.ceil(len(gt_eval) / blocksize))
    blockorder_eval = list(range(num_blocks_eval))
    random.shuffle(blockorder_eval)
    newinds_eval = list([j + blocksize * i for i in blockorder_eval for j in np.arange(blocksize)])
    # apply shuffle
    X_eval_multiclass = np.array([proba_eval[i, :] for i in newinds_eval if i < len(gt_eval)])
    y_eval_multiclass = np.array([gt_eval[i] for i in newinds_eval if i < len(gt_eval)])

    # (we don't need to shuffle the test set since ordering of test entries doesn't matter for performance.)

    # and this is the code that actually does the training--------------------------------------------------------------

    # train classifier on (X_tr_multiclass, y_tr_multiclass) and test it on (X_ev_multiclass, y_ev_multiclass)
    eval_set = [(X_eval_multiclass, y_eval_multiclass)]
    pdb.set_trace()
    params = {'n_estimators': 500, 'max_depth': 5}  # you can change these parameter settings if you want, see XGBClassifier documentation
    clf = XGBClassifier(**params)  # you can also use a different classifier here if you'd like
    clf.fit(X_train_multiclass, y_train_multiclass, eval_set=eval_set, early_stopping_rounds=20, verbose=True)

    print('done training!')

    results = clf.predict(proba_test)
    score_info(gt_test, results, vocab)  # this should output precision/recall info for your classifier, or you can look at results manually

    classifier = {'clf': clf, 'vocab': vocab, 'merged_behaviors': trained_behaviors}
    dill.dump(classifier, open(os.path.join(savedir, 'class_merger'), 'wb'))
    return gt_test, results