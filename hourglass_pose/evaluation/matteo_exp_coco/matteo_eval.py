import os, sys
import cPickle as pickle
import json
from json import encoder

sys.path.append('../')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from pycocotools.cocoanalyze import COCOanalyze

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io


def show_dets(coco_dts, coco_gts, img_path, sigmas):
    if len(coco_dts) == 0 and len(coco_gts) == 0:
        return 0

    skeleton = [[1,2],[2,3],[1,3],[4,5],[4,6],[5,7],[6,7]]
    # sigmas = 2*np.array([ 0.02104489,  0.02295525,  0.02300004,  0.02369522,  0.0304542 , 0.03030587,  0.02386336])

    oks = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    a = np.sqrt(-2 * np.log(oks))
    NUM_KEYPOINTS = 7

    I = io.imread(img_path)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(I,cmap='gray')

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []

    for i,ann in enumerate(coco_gts):
        if i == 0:
            color = 'yellow'
        else:
            color = 'green'

        if 'keypoints' in ann and type(ann['keypoints']) == list:
            # turn skeleton into zero-based index
            sks = np.array(skeleton) - 1
            kp = np.array(ann['keypoints'])
            x = kp[0::3]
            y = kp[1::3]
            for sk in sks:
                plt.plot(x[sk], y[sk], linewidth=3, color=color)

            plt.plot(x, y, 'o', markersize=2, markerfacecolor=color,
                     markeredgecolor=color, markeredgewidth=3)

            for x1, y1, sigma1 in zip(x, y, sigmas):
                r = 2*sigma1 * (np.sqrt(ann["area"]) + np.spacing(1))
                # r = sigma1 * (ann["area"]+np.spacing(1))
                circle = plt.Circle((x1, y1), a[0] * r, fc=(1, 0, 0, 0.4), ec='k')
                ax.add_patch(circle)

                circle = plt.Circle((x1, y1), a[7] * r, fc=(0, 0, 0, 0), ec='k')
                ax.add_patch(circle)

    for i,ann in enumerate(coco_dts):
        if i == 0:
            color = 'red'
        else:
            color = 'blue'

        sks = np.array(skeleton) - 1
        kp = np.array(ann['keypoints'])
        x = kp[0::3]
        y = kp[1::3]
        for sk in sks:
            plt.plot(x[sk], y[sk], linewidth=3, color=color)

        for kk in xrange(len(sigmas)):
            plt.plot(x[kk], y[kk], 'o', markersize=5, markerfacecolor=color,
                             markeredgecolor=color, markeredgewidth=3)

    for g,d in zip(coco_gts,coco_dts):
        bbox  = g['bbox']
        score = d['score']
        oks = 0.
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor=[1, .6, 0], linewidth=3)

        dt_kpt_arr = np.array(d['keypoints'])
        gt_kpt_arr = np.array(g['keypoints'])

        xd = dt_kpt_arr[0::3]
        yd = dt_kpt_arr[1::3]
        xg = gt_kpt_arr[0::3]
        yg = gt_kpt_arr[1::3]

        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((len(sigmas)))

        dx = xd - xg
        dy = yd - yg

        e = (dx ** 2 + dy ** 2) / (sigmas * 2) ** 2 / (g['area'] + np.spacing(1)) / 2
        oks = np.sum(np.exp(-e)) / e.shape[0]
        print(oks)

        # ax.annotate("[%.3f][%.3f]" % (score, oks), (bbox[0] + bbox[2] / 2.0, bbox[1] + 15),
        #             color=[1, .6, 0], weight='bold', fontsize=12, ha='center', va='center')
        # ax.add_patch(rect)

    #plt.show()
    plt.savefig('error.pdf',bbox_inches='tight')

path = '../cocoEval_old.pkl'
with open(path, 'r') as fp:
    results = pickle.load(fp)

# correct annotations to right format
gt_anns = results[0]['annotations']
for i,g in enumerate(gt_anns):
    g['num_keypoints'] = int(g['num_keypoints'])
    g['area']          = float(g['area'])
    g['bbox']          = map(float,g['bbox'])
    g['keypoints']     = map(float, g['keypoints'])
    g['image_old']     =int(g['image_old'][0])

    if i==0:
        g['image_id'] = 1
        continue

    if i%2 != 0:
        g['image_id']  = gt_anns[i-1]['image_id']
    else:
        g['image_id']  = gt_anns[i-1]['image_id']+1


results[0]['annotations'] = gt_anns
results[0]['images'] = [{'id' : i+1} for i in range(len(gt_anns)/2)]
with open('./gt_data.json','wb') as fp:
    json.dump(results[0], fp)

for i,d in enumerate(results[1]):
    d['keypoints']     = map(float, d['keypoints'])

    if i==0:
        d['image_id'] = 1
        continue

    if i%2 != 0:
        d['image_id']  = results[1][i-1]['image_id']
    else:
        d['image_id']  = results[1][i-1]['image_id']+1

with open('./dt_data.json','wb') as fp:
    json.dump(results[1], fp)

COCO_GT  = './gt_data.json'
COCO_DT  = './dt_data.json'
SAVE_DIR = './'

coco_gt = COCO( COCO_GT )

print("{:15}[{}] instances in [{}] images.".format('Ground-truth:',
                                                   len(coco_gt.getAnnIds()),
                                                   len(coco_gt.getImgIds())))
## create imgs_info dictionary
with open(COCO_GT,'rb') as fp:
    data = json.load(fp)
imgs_info = {i['id']:{'id':i['id'] ,
                      'width':1024,
                      'height':570}
                       for i in data['images']}
assert(len(coco_gt.getImgIds())==len(imgs_info))

## load team detections
with open(COCO_DT,'rb') as fp: team_dts = json.load(fp)
team_dts     = [d for d in team_dts if d['image_id'] in imgs_info]
team_img_ids = set([d['image_id'] for d in team_dts])
print("{:15}[{}] instances in [{}] images.".format('Detections:',
                                                   len(team_dts),
                                                   len(team_img_ids)))

coco_dt   = coco_gt.loadRes(team_dts)
coco_analyze = COCOanalyze(coco_gt,coco_dt,'keypoints')
sigmas =np.array([ 0.02104489,  0.02295525,  0.02300004,  0.02369522,  0.0304542 , 0.03030587,  0.02386336])
img_path ='./images/00643.jpeg'
show_dets(coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=643)), coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=643)),img_path, sigmas)

# thresholds to use for analysis
coco_analyze.params.oksThrs       = [.95,.75]

# smallest threshold considered a localization error
coco_analyze.params.oksLocThrs    = .1
# threshold limits for jitter errors
coco_analyze.params.jitterOksThrs = [.5,.85]

# area ranges for evaluation
coco_analyze.params.areaRng       = [[0 ** 2, 1e5 ** 2]]
coco_analyze.params.areaRngLbl    = ['all']

# sigmas
coco_analyze.params.sigmas = sigmas

coco_analyze.params.kpts_name = \
            [u'nose',
             u'right_ear', u'left_ear',
             u'neck',
             u'right_side', u'left_side',
             u'tail']

coco_analyze.params.inv_kpts_name = \
            [u'nose',
             u'left_ear', u'right_ear',
             u'tail',
             u'left_side', u'right_side',
             u'neck']

coco_analyze.params.num_kpts = len(coco_analyze.params.kpts_name)
coco_analyze.params.inv_idx      = [  coco_analyze.params.inv_kpts_name.index(coco_analyze.params.kpts_name[i]) for i in xrange(coco_analyze.params.num_kpts)]

coco_analyze.evaluate(verbose=True, makeplots=True)

coco_analyze.params.err_types = ['jitter','inversion','swap','miss']
coco_analyze.analyze(check_kpts=True,check_scores=True,check_false=True)
coco_analyze.summarize(makeplots=True, savedir='./', team_name='test')

corrected_dts = coco_analyze.corrected_dts