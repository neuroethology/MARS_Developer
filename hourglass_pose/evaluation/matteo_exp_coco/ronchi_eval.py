import os, sys
import cPickle as pickle
import json
from json import encoder
from jinja2 import Template
import jinja2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
sys.path.append('coco-analyze-release/')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
from pycocotools.cocoanalyze import COCOanalyze

## Analysis API imports
from analysisAPI.errorsAPImpact import errorsAPImpact
from analysisAPI.localizationErrors import localizationErrors
from analysisAPI.scoringErrors import scoringErrors
from analysisAPI.backgroundFalsePosErrors import backgroundFalsePosErrors
from analysisAPI.backgroundFalseNegErrors import backgroundFalseNegErrors
from analysisAPI.occlusionAndCrowdingSensitivity import occlusionAndCrowdingSensitivity
from analysisAPI.sizeSensitivity import sizeSensitivity
from analysisAPI.utilities import *

def show_dets(coco_dts, coco_gts, img_path,  sigmas):
    if len(coco_dts) == 0 and len(coco_gts) == 0:
        return 0

    if not os.path.exists(SAVE_DIR+'/coco_sigma/'): os.makedirs(SAVE_DIR+'/coco_sigma/')
    # skeleton = [[1,2],[2,3],[1,3],[4,5],[4,6],[5,7],[6,7]]
    skeleton = [[1,2],[2,3],[1,3],[4,5],[4,6],[5,7],[6,7],[4,8],[4,9],[5,10],[6,11]]
    # sigmas = 2*np.array([ 0.02104489,  0.02295525,  0.02300004,  0.02369522,  0.0304542 , 0.03030587,  0.02386336])
    sigmas = np.array([0.03465136, 0.0344434, 0.03461611, 0.0362902, 0.04186067, 0.04204246, 0.03432178, 0.04125073, 0.04113029, 0.04016324, 0.04140253])  # front

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
        print oks

        # ax.annotate("[%.3f][%.3f]" % (score, oks), (bbox[0] + bbox[2] / 2.0, bbox[1] + 15),
        #             color=[1, .6, 0], weight='bold', fontsize=12, ha='center', va='center')
        # ax.add_patch(rect)

    #plt.show()

    plt.savefig(SAVE_DIR+'/coco_sigma/error_%06d.pdf' % id,bbox_inches='tight')
    plt.savefig(SAVE_DIR+'/coco_sigma/error_%06d.png' % id,bbox_inches='tight')
    plt.close()
view='front'
root = '/media/cristina/data_lnx/cristina/mars/results_keypoints_j/'
path = root + 'top_fixed/' if view=='top' else root+'front_moreaug/'
SAVE_DIR =  path +'cocoanalyze'
if not os.path.exists(SAVE_DIR):os.makedirs(SAVE_DIR)
COCO_GT  = path +'gt_data.json'
COCO_DT  = path +'dt_data.json'

############ reformat data from my pr curve prep data
cocoeval_file = path +'cocoEval.pkl'
with open(cocoeval_file, 'r') as fp: results = pickle.load(fp)

# correct gt annotations to right format
# consecutive detection leads to double the images, we need 1 images for two mice so
# correcting the ids
gt_anns = results[0]['annotations']
for i,g in enumerate(gt_anns):
    g['num_keypoints'] = int(g['num_keypoints'])
    g['area']          = float(g['area'])
    g['bbox']          = map(float,g['bbox'])
    g['keypoints']     = map(float, g['keypoints'])
    # g['image_old']     =int(g['image_old'][0])
    if i==0:
        g['image_id'] = 1
        continue
    if i%2 != 0:
        g['image_id']  = gt_anns[i-1]['image_id']
    else:
        g['image_id']  = gt_anns[i-1]['image_id']+1
results[0]['annotations'] = gt_anns
results[0]['images'] = [{'id' : i+1} for i in range(len(gt_anns)/2)]
with open(path +'gt_data.json','wb') as fp:  json.dump(results[0], fp)
# same for predictions
for i,d in enumerate(results[1]):
    d['keypoints']     = map(float, d['keypoints'])
    if i==0:
        d['image_id'] = 1
        continue
    if i%2 != 0:
        d['image_id']  = results[1][i-1]['image_id']
    else:
        d['image_id']  = results[1][i-1]['image_id']+1
with open(path +'dt_data.json','wb') as fp:  json.dump(results[1], fp)

############## plots prediction and gt with sigmas
coco_gt = COCO( COCO_GT )
print("{:15}[{}] instances in [{}] images.".format('Ground-truth:',
                                                   len(coco_gt.getAnnIds()),
                                                   len(coco_gt.getImgIds())))
## create imgs_info dictionary
with open(COCO_GT,'rb') as fp:  data = json.load(fp)
imgs_info = {i['id']:{'id':i['id'] ,
                      'width':1024 if view=='top' else 1280,
                      'height':570 if view=='top' else 500}
                       for i in data['images']}
assert(len(coco_gt.getImgIds())==len(imgs_info))

## load team detections
with open(COCO_DT,'rb') as fp: team_dts = json.load(fp)
team_dts     = [d for d in team_dts if d['image_id'] in imgs_info]
team_img_ids = set([d['image_id'] for d in team_dts])
print("{:15}[{}] instances in [{}] images.".format('Detections:',
                                                   len(team_dts),
                                                   len(team_img_ids)))
print("Loaded [{}] instances in [{}] images.".format(len(team_dts),len(imgs_info)))

coco_dt   = coco_gt.loadRes(team_dts)
coco_analyze = COCOanalyze(coco_gt,coco_dt,'keypoints')
if not os.path.exists('figures'):os.makedirs('figures')
sigmas= np.array([0.02349794, 0.02573029, 0.02574004, 0.02459145, 0.03146337, 0.03146769, 0.02510474]) if view=='top' else \
    np.array([0.03465136, 0.0344434 , 0.03461611, 0.0362902 , 0.04186067, 0.04204246, 0.03432178, 0.04125073,0.04113029, 0.04016324, 0.04140253]) #front
plt.ioff()
for id in range(1,len(team_img_ids)):
    print(id)
    img_filename = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=id))[0]['filename']
    if view=='top':
        if os.path.exists('../../A_annotation/selected_frames_5k_miniscope_top/' + img_filename):
            img_path ='../../A_annotation/selected_frames_5k_miniscope_top/' + img_filename
        elif os.path.exists('../../A_annotation/selected_frames_10K/' + img_filename):
            img_path ='../../A_annotation/selected_frames_10K/' + img_filename
        else:
            print('image not found')
    else:
        if os.path.exists('../../A_annotation/selected_frames_5k_miniscope_front/' + img_filename):
            img_path ='../../A_annotation/selected_frames_5k_miniscope_front/' + img_filename
        elif os.path.exists('../../A_annotation/selected_frames_10K_front/' + img_filename):
            img_path ='../../A_annotation/selected_frames_10K_front/' + img_filename
        else:
            print('image not found')
    show_dets(coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[id])), coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[id])),img_path, sigmas)
    # show_dets(coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[id,id+1])), coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[id,id+1])),img_path, sigmas)


########################################################################################################################
gt_data = json.load(open(COCO_GT,'rb'))
coco_gt = COCO( COCO_GT )

imgs_info = {i['id']:{'id':i['id'] ,
                      'width':1024 if view=='top' else 1280,
                      'height':570 if view=='top' else 500,
                      'coco_url':
                          '../../A_annotation/selected_frames_5k_miniscope_top/' + coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=i['id']))[0]['filename'] if
                          os.path.exists('../../A_annotation/selected_frames_5k_miniscope_top/' + coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=i['id']))[0]['filename']) else
                          '../../A_annotation/selected_frames_10K/' + coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=i['id']))[0]['filename'] if view =='top' else '../../A_annotation/selected_frames_5k_miniscope_top/' + coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=i['id']))[0]['filename'] if
                          os.path.exists('../../A_annotation/selected_frames_5k_miniscope_front/' + coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=i['id']))[0]['filename']) else
                          '../../A_annotation/selected_frames_10K_front/' + coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=i['id']))[0]['filename']}
                       for i in gt_data['images']}

team_dts = json.load(open(COCO_DT,'rb'))
team_dts = [d for d in team_dts if d['image_id'] in imgs_info]
team_img_ids = set([d['image_id'] for d in team_dts])
print("Loaded [{}] instances in [{}] images.".format(len(team_dts),len(imgs_info)))

## initialize COCO detections api
coco_dt   = coco_gt.loadRes( team_dts )

## initialize COCO analyze api
coco_analyze = COCOanalyze(coco_gt, coco_dt, 'keypoints')

# coco_analyze.params.oksThrs       = [.5,.75,.85,.95]
coco_analyze.params.oksThrs       = [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]
# smallest threshold considered a localization error
coco_analyze.params.oksLocThrs    = .1
# threshold limits for jitter errors
coco_analyze.params.jitterOksThrs = [.5,.75,.85, 0.9]
# area ranges for evaluation
coco_analyze.params.areaRng       = [[0 ** 2, 1e5 ** 2],[32 ** 2, 96 ** 2]]#[96 ** 2, 1e5 ** 2],[32 ** 2, 96 ** 2]
coco_analyze.params.areaRngLbl    = ['all']
coco_analyze.params.maxDets = [2]  # 'large','medium'
# sigmas
sigmas= np.array([0.02349794, 0.02573029, 0.02574004, 0.02459145, 0.03146337, 0.03146769, 0.02510474]) if view=='top' else \
    np.array([0.03465136, 0.0344434 , 0.03461611, 0.0362902 , 0.04186067, 0.04204246, 0.03432178, 0.04125073,0.04113029, 0.04016324, 0.04140253]) #front

coco_analyze.params.sigmas = sigmas
coco_analyze.params.kpts_name =  [u'nose',u'right_ear', u'left_ear',u'neck',u'right_side', u'left_side',u'tail'] if view=='top' else \
[u'nose',u'right_ear', u'left_ear',u'neck',u'right_side', u'left_side',u'tail', u'right_front_paw',u'left_front_paw',u'right_rear_paw',u'left_rear_paw']
coco_analyze.params.inv_kpts_name = [u'neck',u'left_ear', u'right_ear', u'nose', u'left_side', u'right_side',u'tail'] if view=='top' else \
    [u'neck',u'left_ear', u'right_ear', u'nose', u'left_side', u'right_side',u'tail', u'left_front_paw', u'right_front_paw',u'left_rear_paw',u'right_rear_paw', u'left_rear_paw']
coco_analyze.params.num_kpts = len(coco_analyze.params.kpts_name)
coco_analyze.params.inv_idx  = [ coco_analyze.params.inv_kpts_name.index(coco_analyze.params.kpts_name[i])
                                 for i in xrange(coco_analyze.params.num_kpts)]
# use analyze() method for advanced error analysis
# input arguments:
#  - check_kpts   : analyze keypoint localization errors for detections with a match (default: True)
#                 : default errors types are ['jitter','inversion','swap','miss']
#  - check_scores : analyze optimal score (maximizing oks over all matches) for every detection (default: True)
#  - check_bkgd   : analyze background false positives and false negatives (default: True)
coco_analyze.evaluate(verbose=True, makeplots=True,savedir=SAVE_DIR)
coco_analyze.analyze(check_kpts=True, check_scores=True, check_bckgd=True)
coco_analyze.summarize(makeplots=True,savedir=SAVE_DIR)

#######################################################################################################3

latex_jinja_env = jinja2.Environment(
        block_start_string    = '\BLOCK{',
        block_end_string      = '}',
        variable_start_string = '\VAR{',
        variable_end_string   = '}',
        comment_start_string  = '\#{',
        comment_end_string    = '}',
        line_statement_prefix = '%%',
        line_comment_prefix   = '%#',
        trim_blocks           = True,
        autoescape            = False,
        loader                = jinja2.FileSystemLoader(os.path.abspath('coco-analyze-release/latex/'))
    )
template = latex_jinja_env.get_template('report_template.tex')
template_vars  = {}
template_vars['num_dts'] = len(team_dts)
template_vars['num_imgs_dts'] = len(set([d['image_id'] for d in team_dts]))
template_vars['num_imgs'] = len(imgs_info)
coco_analyze.evaluate(verbose=True, makeplots=True, savedir=SAVE_DIR)
template_vars['overall_prc_large'] = '%s/prc_[None][large][%d].pdf' % (SAVE_DIR,  coco_analyze.params.maxDets[0])
template_vars['overall_prc_all'] = '%s/prc_[None][all][%d].pdf' % (SAVE_DIR,  coco_analyze.params.maxDets[0])
## analyze imapct on AP of all error types
paths = errorsAPImpact( coco_analyze, SAVE_DIR )
template_vars.update(paths)

## analyze breakdown of localization errors
paths = localizationErrors( coco_analyze, imgs_info, SAVE_DIR )
template_vars.update(paths)

## analyze background false positives
paths = backgroundFalsePosErrors( coco_analyze, imgs_info, SAVE_DIR )
template_vars.update(paths)

# analyze background false negatives
paths = backgroundFalseNegErrors( coco_analyze, imgs_info, SAVE_DIR )
template_vars.update(paths)

## analyze scoring errors
paths = scoringErrors( coco_analyze, 0.75, imgs_info, SAVE_DIR )
template_vars.update(paths)

## analyze sensitivity to occlusion and crowding of instances
paths = occlusionAndCrowdingSensitivity( coco_analyze, .75, SAVE_DIR )
template_vars.update(paths)

## analyze sensitivity to size of instances
paths = sizeSensitivity( coco_analyze, .75, SAVE_DIR )
template_vars.update(paths)

output_report = open(SAVE_DIR +'/performance_report.tex', 'w')
output_report.write( template.render(template_vars) )
output_report.close()




##############################################################################

# coco_analyze.params.err_types = ['miss','swap','inversion','jitter']
coco_analyze.params.err_types = ['jitter','miss','inversion','swap']
coco_analyze.analyze(check_kpts=True, check_scores=True, check_bckgd=True)
coco_analyze.summarize(makeplots=True, savedir=SAVE_DIR, team_name='test')

corrected_dts = coco_analyze.corrected_dts
corrected_dts = coco_analyze.corrected_dts['all']

i = 6
# info on keypoint detection localization error
print 'good: %s'%corrected_dts[i]['good']
print 'miss: %s'%corrected_dts[i]['miss']
print 'swap: %s'%corrected_dts[i]['swap']
print 'inv.: %s'%corrected_dts[i]['inversion']
print 'jit.: %s\n'%corrected_dts[i]['jitter']

# corrected keypoint locations
print 'predicted keypoints:\n %s'%corrected_dts[i]['keypoints']
print 'corrected keypoints:\n %s\n'%corrected_dts[i]['opt_keypoints']

# optimal detection score
print 'original score: %s'%corrected_dts[i]['score']
print 'optimal score:  %s\n'%corrected_dts[i]['opt_score']

false_pos_dts = coco_analyze.false_pos_dts
false_neg_gts = coco_analyze.false_neg_gts
for oks in coco_analyze.params.oksThrs:
    print "Oks:[%.2f] - Num.FP:[%d] - Num.FN:[%d]"%(oks,len(false_pos_dts['all',str(oks)]),len(false_neg_gts['all',str(oks)]))
