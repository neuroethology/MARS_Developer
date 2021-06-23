from __future__ import division
import os,sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import binarize
from collections import Counter
from sklearn.ensemble import BaggingClassifier
from hmmlearn import hmm
import scipy
from scipy import signal
import MARS_annotation_parsers as map
import pywt
from scipy.signal import medfilt
import progressbar
import multiprocessing as mp


flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


def clean_data(data):
    """Eliminate the NaN and Inf values by taking the last value that was neither."""
    idx = np.where(np.isnan(data) | np.isinf(data))
    if idx[0].size>0:
        for j in range(len(idx[0])):
            if idx[0][j] == 0:
                data[idx[0][j], idx[1][j],idx[2][j]] = 0.
            else:
                data[idx[0][j], idx[1][j],idx[2][j]] = data[idx[0][j] - 1, idx[1][j],idx[2][j]]
    return data


def apply_wavelet_transform(starter_features):
    wave1 = pywt.ContinuousWavelet('gaus8')
    # wave2 = pywt.ContinuousWavelet('gaus7')
    scales = [1, 3, 5, 10, 30, 90, 270]
    dims = np.shape(starter_features)
    nfeat = dims[1]

    transformed_features = np.zeros((dims[0], nfeat + nfeat * len(scales)))#*2))
    transformed_features[:, range(0, nfeat)] = starter_features

    for f, feat in enumerate(starter_features.swapaxes(0, 1)):
        for w, wavelet in enumerate([wave1]): #, wave2]):
            for i, s in enumerate(scales):
                a, _ = pywt.cwt(medfilt(feat), s, wavelet)
                a[0][:s] = 0
                a[0][-s:] = 0
                inds = nfeat + f + nfeat*i + w*nfeat*len(scales)
                transformed_features[:, inds] = a

    return transformed_features


def compute_win_feat(starter_feature, windows=[3, 11, 21]):
    """This function computes the window features from a given starter feature."""
    # Define the functions being computed.
    fxns = [np.min, np.max, np.mean, np.std]
    num_fxns = len(fxns)

    # Get the number of frames
    number_of_frames = np.shape(starter_feature)[0]

    # Count the number of windows.
    num_windows = len(windows)

    # Number of additional features should simply be (the number of windows)*(the number of fxns) --in our case, 12.
    num_feats = num_windows * num_fxns

    # Create a placeholder for the features.d
    features = np.zeros((number_of_frames, num_feats))

    # Loop over the window sizes
    for window_num, w in enumerate(windows):
        # Iterate with a given window size.
        # Get the space where we should put the newly computed features.
        left_endpt = window_num * num_fxns
        right_endpt = window_num * num_fxns + num_fxns
        
        # Compute the features and store them.
        features[:, left_endpt:right_endpt] = get_JAABA_feats(starter_feature=starter_feature, window_size=w)

    return features


def get_JAABA_feats(starter_feature, window_size=3):
    # Get the number of frames.
    number_of_frames = np.shape(starter_feature)[0]
    # Get the radius of the window.
    radius = (window_size - 1) / 2
    radius = int(radius)
    r = int(radius)
    row_placeholder = np.zeros(window_size)
    column_placeholder = np.zeros(number_of_frames)
    
    row_placeholder[:r] = np.flip(starter_feature[1:(radius + 1)], 0)
    row_placeholder[r:] = starter_feature[:(radius + 1)]
    
    column_placeholder[:-radius] = starter_feature[radius:]
    column_placeholder[-radius:] = np.flip(starter_feature[-(radius + 1):-1], 0)
    
    # Create the matrix that we're going to compute on.
    window_matrix = scipy.linalg.toeplitz(column_placeholder, row_placeholder)
    
    # Set the functions.
    fxns = [np.min, np.max, np.mean, np.std]
    num_fxns = len(fxns)
    
    # Make a placeholder for the window features we're computing.
    window_feats = np.zeros((number_of_frames, num_fxns))

    # Do the feature computation.
    for fxn_num, fxn in enumerate(fxns):
        if (window_size <= 3) & (fxn == np.mean):
            window_feats[:, fxn_num] = starter_feature
        else:
            window_feats[:, fxn_num] = fxn(window_matrix, axis=1)
    return window_feats


def apply_windowing(starter_features):
    windows = [3, 11, 21]
    total_feat_num = np.shape(starter_features)[1]

    window_features = np.array([])
    for i in range(total_feat_num):
        feat_temp = compute_win_feat(starter_features[:, i], windows)
        if i==0:
            window_features = feat_temp
        else:
            window_features = np.concatenate((window_features,feat_temp), axis=1)
    
    return window_features


def normalize_pixel_data(data,view):
    if view == 'top':fd = [range(40, 49)]
    elif view == 'front': fd = [range(47,67)]
    elif view == 'top_pcf': fd = [range(40,57)]
    fd = list(flatten(fd))
    md = np.nanmedian(data[:, :, fd], 1, keepdims=True)
    data[:, :, fd] /= md
    return data


def remove_pixel_data(data, view):
    if view == 'top': fd = [range(40, 49)]
    elif view == 'front': fd = [range(47,67)]
    elif view == 'top_pcf': fd = [range(40,57)]
    fd = list(flatten(fd))
    if type(data) == np.ndarray:
        data = np.delete(data, fd, 1)
    else:
        data = [i for j, i in enumerate(data) if j not in fd]
    return data


def get_transmat(gt, n_states):
    # Count transitions between states
    cs = [Counter() for _ in range(n_states)]
    prev = gt[0]
    for row in gt[1:]:
        cs[prev][row] += 1
        prev = row

    # Convert to probabilities
    transitions = np.zeros((n_states, n_states))
    for x in range(n_states):
        for y in range(n_states):
            transitions[x, y] = float(cs[x][y]) / float(sum(cs[x].values()))
    return transitions


def get_emissionmat(gt,pred,n_states):
    # The emissions are the translations from ground truth to predicted
    # Count emissions
    cs = [Counter() for _ in range(n_states)]
    for i, row in enumerate(gt):
        cs[row][pred[i]] += 1

    # Compute probabilities
    emissions = np.zeros((n_states, n_states))
    for x in range(n_states):
        for y in range(n_states):
            emissions[x, y] = float(cs[x][y]) / float(sum(cs[x].values()))
    return emissions


def do_fbs(y_pred_class, kn, blur, blur_steps, shift):
    """Does forward-backward smoothing."""
    len_y = len(y_pred_class)

    # fbs with classes
    z = np.zeros((3, len_y))  # Make a matrix to hold the shifted predictions --one row for each shift.

    # Create mirrored start and end indices for extending the length of our prediction vector.
    mirrored_start = range(shift, -1, -1)  # Creates indices that go (shift, shift-1, ..., 0)
    mirrored_end = range(len_y - 1, len_y - 1 - shift, -1)  # Creates indices that go (-1, -2, ..., -shift)

    # Now we extend the predictions to have a mirrored portion on the front and back.
    extended_predictions = np.r_[ y_pred_class[mirrored_start], y_pred_class, y_pred_class[mirrored_end] ]

    # Do our blurring.
    for s in range(blur_steps):
        extended_predictions = signal.convolve(np.r_[extended_predictions[0], extended_predictions, extended_predictions[-1]],
                                               kn / kn.sum(),  # The kernel we are convolving.
                                               'valid')  # Only use valid conformations of the filter.
        # Note: this will leave us with 2 fewer items in our signal each iteration, so we append on both sides.

    z[0, :] = extended_predictions[2 * shift + 1:]
    z[1, :] = extended_predictions[:-2 * shift - 1]
    z[2, :] = extended_predictions[shift + 1:-shift]

    z_mean = np.mean(z, axis=0)  # Average the blurred and shifted signals together.
    y_pred_fbs = binarize(z_mean.reshape((-1, 1)), .5).astype(int).reshape((1, -1))[0]  # Anything that has a signal strength over 0.5, is taken to be positive.
    return y_pred_fbs


def do_hmm(gt,pd):
    hmm_bin = hmm.MultinomialHMM(n_components=2, algorithm="viterbi", random_state=42, params="", init_params="")
    hmm_bin.startprob_ = np.array([np.sum(gt == i) / float(len(gt)) for i in range(2)])
    hmm_bin.transmat_ = get_transmat(gt, 2)
    hmm_bin.emissionprob_ = get_emissionmat(gt, pd, 2)
    return hmm_bin
