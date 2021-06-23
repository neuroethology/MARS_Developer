import yaml
import json
import os
import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pose_annotation_tools.json_util import *


def plot_frame(project, fr, markersize=8, figsize=[15, 10]):
    # plots annotations from all workers plus the worker median for an example frame.

    config_fid = os.path.join(project ,'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)

    plt.figure(figsize=figsize)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    markers = 'vosd*p'

    im = mpimg.imread(D[fr]['image'])
    plt.imshow(im, cmap='gray')

    # plot the labels from each individual worker:
    for mouse in config['animal_names']:
        for w, [x, y] in enumerate(zip(D[fr]['ann_' + mouse]['X'], D[fr]['ann_' + mouse]['Y'])):
            for i, [px, py] in enumerate(zip(x, y)):
                plt.plot(px * D[fr]['width'], py * D[fr]['height'],
                         colors[i % 9], marker=markers[w % 6], markersize=markersize)

        for i, [px, py] in enumerate(zip(D[fr]['ann_' + mouse]['med'][1], D[fr]['ann_' + mouse]['med'][0])):
            plt.plot(np.array(px) * D[fr]['width'], np.array(py) * D[fr]['height'],
                     'k', marker='o', markeredgecolor='w', markeredgewidth=math.sqrt(markersize), markersize=markersize)
    plt.show()


def compute_human_PCK(project, animal_names=None, xlim=None, pixel_units=False):

    config_fid = os.path.join(project ,'project_config.yaml')
    with open(config_fid) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['pixels_per_cm'] and not config['pixels_per_cm'] == 'None':
        if pixel_units:
            pixels_per_cm = 1.0
        else:
            pixels_per_cm = config['pixels_per_cm']
    else:  # no pixels/cm conversion, default to pixels.
        pixels_per_cm = 1.0

    if animal_names is None:
        animal_names = config['animal_names']

    dictionary_file_path = os.path.join(project, 'annotation_data', 'processed_keypoints.json')
    if not os.path.exists(dictionary_file_path):
        make_annot_dict(project)
    with open(dictionary_file_path, 'r') as fp:
        D = json.load(fp)

    bins = 10000
    if xlim:
        binrange = [-1 / bins, xlim[1] * 2 + 1 / bins]
    else:
        binrange = [-1 / bins, max(D[0]['width'], D[0]['height']) + 1 / bins]
    nSamp = len(D)
    nKpts = len(D[0]['ann_label'])

    fields = ['min', 'max', 'mean', 'med']
    ptNames = D[0]['ann_label']
    ptNames = ['all'] + ptNames

    counts = {a: {p: {n: [] for n in fields} for p in ptNames} for a in animal_names}
    super_counts = {p: {n: np.zeros(bins) for n in fields} for p in ptNames}
    for cnum, animal in enumerate(animal_names):
        dMean = np.zeros((nKpts, nSamp))  # average worker-gt distance
        dMedian = np.zeros((nKpts, nSamp))  # median worker-gt distance
        dMin = np.zeros((nKpts, nSamp))  # performance of best worker on a given frame
        dMax = np.zeros((nKpts, nSamp))  # performance of worst worker on a given frame

        for fr, frame in enumerate(D):
            X = np.array(frame['ann_' + animal]['X']) * D[0]['width']
            Y = np.array(frame['ann_' + animal]['Y']) * D[0]['height']
            trial_dists = []
            for i, [pX, pY] in enumerate(zip(X, Y)):
                mX = np.median(np.delete(X, i, axis=0), axis=0)
                mY = np.median(np.delete(Y, i, axis=0), axis=0)
                trial_dists.append(np.sqrt(np.square(mX - pX) + np.square(mY - pY)))
            trial_dists = np.array(trial_dists)

            dMean[:, fr] = np.mean(trial_dists, axis=0)
            dMedian[:, fr] = np.median(trial_dists, axis=0)
            dMin[:, fr] = np.min(trial_dists, axis=0)
            dMax[:, fr] = np.max(trial_dists, axis=0)

        for c, use in enumerate([dMin, dMax, dMean, dMedian]):
            for p, pt in enumerate(use):
                counts[animal][ptNames[p+1]][fields[c]], usedbins = np.histogram(pt, bins, range=binrange)
                counts[animal][ptNames[p+1]][fields[c]] = counts[animal][ptNames[p+1]][fields[c]]\
                                                          / sum(counts[animal][ptNames[p+1]][fields[c]])
                super_counts[ptNames[p+1]][fields[c]] += counts[animal][ptNames[p+1]][fields[c]] / len(animal_names)

            counts[animal]['all'][fields[c]], _ = np.histogram(np.mean(use,axis=0), bins, range=binrange)
            counts[animal]['all'][fields[c]] = counts[animal]['all'][fields[c]] / sum(counts[animal]['all'][fields[c]])
            super_counts['all'][fields[c]] += counts[animal]['all'][fields[c]] / len(animal_names)

        if not pixel_units:
            usedbins = usedbins / pixels_per_cm
        binctrs = usedbins[1:]  # (usedbins[1:] + usedbins[:-1]) / 2.0

    return counts, super_counts, binctrs


def plot_human_PCK(project, animal_names=None, xlim=None, pixel_units=False, combine_animals=False):
    # plot inter-worker variability

    counts, super_counts, binctrs = compute_human_PCK(project, xlim=xlim, pixel_units=pixel_units)

    nKpts = len(super_counts.keys())
    ptNames = super_counts.keys()
    animal_names = counts.keys()
    thr = 0.85
    fig, ax = plt.subplots(math.ceil(nKpts /4), 4, figsize=(15, 4* math.ceil(nKpts / 4)))
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    if not combine_animals:
        for cnum, animal in enumerate(animal_names):
            cutoff = 0
            for p, pt in enumerate(ptNames):
                objs = ax[int(p / 4), p % 4].stackplot(binctrs, counts[animal][pt]['max'].cumsum(),
                                                       (counts[animal][pt]['min'].cumsum() -
                                                        counts[animal][pt]['max'].cumsum()),
                                                       color=colors[cnum], alpha=0.25)
                objs[0].set_alpha(0)
                ax[int(p / 4), p % 4].plot(binctrs, counts[animal][pt]['med'].cumsum(),
                                           '--', color=colors[cnum], label=animal+' median')
                cutoff = max(cutoff, sum((counts[animal][pt]['med'].cumsum()) < thr))
            for p, label in enumerate(ptNames):
                ax[int(p / 4), p % 4].set_title(label)
                xlim = xlim if xlim is not None else [0, binctrs[cutoff]]
                ax[int(p / 4), p % 4].set_xlim(xlim)
            ax[int(p / 4), p % 4].legend()
    else:
        cutoff = 0
        for p, pt in enumerate(ptNames):
            objs = ax[int(p / 4), p % 4].stackplot(binctrs, super_counts[pt]['max'].cumsum(),
                                                   (super_counts[pt]['min'].cumsum() -
                                                    super_counts[pt]['max'].cumsum()),
                                                   color=colors[0], alpha=0.25)
            objs[0].set_alpha(0)
            ax[int(p / 4), p % 4].plot(binctrs, super_counts[pt]['med'].cumsum(), '--', color=colors[0])
            cutoff = max(cutoff, sum((super_counts[pt]['med'].cumsum()) < thr))
        for p, label in enumerate(ptNames):
            ax[int(p / 4), p % 4].set_title(label)
            xlim = xlim if xlim is not None else [0, binctrs[cutoff]]
            ax[int(p / 4), p % 4].set_xlim(xlim)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel('percent correct keypoints')
    if pixel_units:
        plt.xlabel('error radius (pixels)')
    else:
        plt.xlabel('error radius (cm)')