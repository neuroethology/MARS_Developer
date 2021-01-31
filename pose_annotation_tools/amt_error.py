from __future__ import division
import os,sys
import json
import dill
import cPickle as pkl
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# read gt keypoints structure
view = 'front'
path5 ='../tf_dataset/front/AMT5K_csv.pkl'
path10 ='../tf_dataset/front/AMT10K_csv.pkl'

#load extracted info from annotations
with open(path10,'rb') as fp:    D = pkl.load(fp)
with open(path5,'rb') as fp:    Dm = pkl.load(fp)
# with open(path5old,'rb') as fp:    Dm_old = pkl.load(fp)
D = D + Dm
w = D[0]['width']
h = D[0]['height']
num_points = len(D[0]['ann_label'])
parts = D[0]['ann_label']
print 'load'

with open('../tf_dataset/front/AMT15_front_csv.pkl','wb') as  fp: pkl.dump(D,fp)

im_path = './'

# #prepare a dict with the info needed for the next step of preparing the tf records
# idg = 1
# v_infob = []
# for i in range(len(D)):
#     print i
#     # image name and id
#     # bbox/label allows to separate between black or white mouse
#     # from the annotation 0 is the black mouse, 1 is the white mouse
#
#     B = D[i]['ann_B']['bbox']
#     W = D[i]['ann_W']['bbox']
#     Bp = D[i]['ann_B']['med']
#     Wp = D[i]['ann_W']['med']
#     i_frame = {'filename': im_path + D[i]['frame_name'],
#                'id': format(idg, '06d'),
#                "class": {
#                    "label": 0,
#                    "text": '',
#                },
#                'width': D[i]['width'],
#                'height': D[i]['height'],
#                'object': {
#                    'id':[0],
#                    'area':[ D[i]['ann_B']['area']],
#                    'bbox': {
#                        'xmin': [B[0]],
#                        'xmax': [B[1]],
#                        'ymin': [B[2]],
#                        'ymax': [B[3]],
#                        'label': [0],
#                        'count': 1,
#                        'score':[1]},
#                    'parts':{
#                         'x':Bp[1][:7].tolist(),
#                         'y':Bp[0][:7].tolist(),
#                         'v':[2]*(len(Bp[0][:7])),
#                         'count': [len(Bp[0][:7])]
#                     }}}
#
#     v_infob.append(i_frame)
#     idg += 1
#
# idg = 1
# v_infow = []
# for i in range(len(D)):
#     print i
#     # image name and id
#     # bbox/label allows to separate between black or white mouse
#     # from the annotation 0 is the black mouse, 1 is the white mouse
#
#     B = D[i]['ann_B']['bbox']
#     W = D[i]['ann_W']['bbox']
#     Bp = D[i]['ann_B']['med']
#     Wp = D[i]['ann_W']['med']
#     i_frame = {'filename': im_path + D[i]['frame_name'],
#                'id': format(idg, '06d'),
#                "class": {
#                    "label": 0,
#                    "text": '',
#                },
#                'width': D[i]['width'],
#                'height': D[i]['height'],
#                'object': {
#                    'id':[0],
#                    'area':[ D[i]['ann_W']['area']],
#                    'bbox': {
#                        'xmin': [W[0]],
#                        'xmax': [W[1]],
#                        'ymin': [W[2]],
#                        'ymax': [W[3]],
#                        'label': [0],
#                        'count': 1,
#                        'score':[1]},
#                    'parts':{
#                         'x':Wp[1][:7].tolist(),
#                         'y':Wp[0][:7].tolist(),
#                         'v':[2]*(len(Wp[0][:7])),
#                         'count': [len(Wp[0][:7])]
#                     }}}
#
#     v_infow.append(i_frame)
#     idg += 1
#
#
# #save the dict
# with open('../tf_dataset_detection/AMT_data/top/dict15k_tfrecords_black.pkl','wb') as fp:    pkl.dump(v_infob,fp)
# with open('../tf_dataset_detection/AMT_data/top/dict15k_tfrecords_white.pkl','wb') as fp:    pkl.dump(v_infow,fp)
# print ('done ')

savedir = 'paper_plots/AMT_analysis/'
if not os.path.exists(savedir):os.makedirs(savedir)

#check changes fixed
# for f in range(len(Dm)):
#
#     d=Dm[f]
#     im = np.asarray(Image.open(d['frame_name']))
#     id = d['frame_id']
#     idx = [o for o in range(len(Dm)) if Dm_old[o]['frame_name']==d['frame_name']][0]
#     do = Dm_old[idx]
#     id_old = do['frame_id']
#
#     # median points
#     mXB = d['ann_B']['med'][0] * h
#     mYB = d['ann_B']['med'][1] * w
#
#     XB = d['ann_B']['X'] * h
#     YB = d['ann_B']['Y'] * w
#
#     mXBo = do['ann_B']['med'][0] * h
#     mYBo = do['ann_B']['med'][1] * w
#
#     XBo = do['ann_B']['X'] * h
#     YBo = do['ann_B']['Y'] * w
#
#     plt.clf()
#     plt.subplot(1,2,1)
#     plt.imshow(im,cmap='gray')
#     plt.plot(mYB[0],mXB[0],'.b')
#     plt.plot(mYBo[0],mXBo[0],'.m')
#     plt.subplot(1,2,2)
#     plt.imshow(im,cmap='gray')
#     plt.plot(YB[:,0],XB[:,0],'.b')
#     plt.pause(2)


def unpack_raw_annotations(annotations_dict):
    """
    Unpacks the raw annotations of a pickled annotation file and produces the X's and Y's, with each annotator in a row.
    """
    raw_X = annotations_dict['Y']
    raw_Y = annotations_dict['X']
    return raw_X, raw_Y

def unpack_gt_annotations(annotation_dict):
    medians = annotation_dict['med']
    gt_X = medians[1,:]
    gt_Y = medians[0,:]
    return gt_X, gt_Y

def calc_pixel_error(gt_X, gt_Y, raw_X, raw_Y, x_scale=1024., y_scale=570.):
    """Calculates the pixel error for a given example."""
    # print(raw_X)
    # print(gt_X)
    err_x = raw_X - gt_X
    err_y = raw_Y - gt_Y
    # print(err_x)
    err_x *= x_scale
    err_y *= y_scale

    # print(err_x)
    # raw_input()
    pixelwise_error = np.sqrt(err_x**2 + err_y**2)
    return pixelwise_error

def annotations_to_error(annotation_dict):
    gt_x, gt_y = unpack_gt_annotations(annotation_dict)
    raw_x, raw_y = unpack_raw_annotations(annotation_dict)
    px_error = calc_pixel_error(gt_x,gt_y, raw_x, raw_y)
    return px_error

# Initialize the lists to store errors.
blk_error = []
b_err = []
white_error = []
w_err = []

# Loop over the size of the annotation set.
for i in range(len(D)):
    # Give an update on how far we are.
    print i

    # Parse out the annotations.
    black_annotations = D[i]['ann_B']
    white_annotations = D[i]['ann_W']

    # Convert annotations to error.
    px_error_black = annotations_to_error(black_annotations) # for each annotator x # for each k
    px_error_white = annotations_to_error(white_annotations)

    # Add the error to the list.
    blk_error.append(px_error_black.tolist())
    b_err.append(px_error_black)
    white_error.append(px_error_white.tolist())
    w_err.append(px_error_white)

# Bind the errors together.
errors = [blk_error,white_error]
errs=[b_err]; errs.extend(w_err)
# Save them as a json.
with open(savedir + 'top_AMT_error.json', 'wb') as fp: json.dump(errors, fp)

err_b=[[] for _ in range(num_points)]
err_w=[[] for _ in range(num_points)]

for k in range(num_points):
    for f in range(len(D)):
        err_b[k].extend(b_err[f][:,k])
        err_w[k].extend(w_err[f][:,k])

err_b=np.asarray(err_b)
err_w=np.asarray(err_w)
err_bw = np.hstack((err_b,err_w))

# plot curve all keypoints
fig=plt.figure()
plt.plot(np.sort(err_bw.ravel()),np.linspace(0,1,len(err_bw.ravel())),'b-',lw=2)
plt.title('Top view, All Keypoints')
plt.xlim([0,100])
plt.xlabel('Error radiud (px)')
plt.xlabel('Fraction of ground truth within a given error radius')

##### each keypoint
fig=plt.figure()
for k in range(num_points):
    plt.subplot(3,3,k+1)
    plt.plot(np.sort(err_bw[k].ravel()), np.linspace(0, 1, len(err_bw[k].ravel())), 'k-', lw=2,label='both')
    plt.plot(np.sort(err_b[k].ravel()), np.linspace(0, 1, len(err_b[k].ravel())), 'b-', lw=2,label='resident')
    plt.plot(np.sort(err_w[k].ravel()), np.linspace(0, 1, len(err_w[k].ravel())), 'r-', lw=2,label='intruder')
    plt.xlim([0,100])
    plt.title(parts[k])
    if k==0: plt.legend(loc='best')
fig.suptitle('Top view, Keypoints')
fig.text(0.5, 0.04, 'Error radius (px)', ha='center')
fig.text(0.0, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical')

##########################################################################################################
# wort best avg all keypoints
def get_err_type(cell_err,type,num_points):
    err_type = np.zeros((len(cell_err),num_points))
    for f in range(len(b_err)):
        curr_err= b_err[f]
        metric = curr_err**2
        avg_err = np.mean(metric,1)
        if type=='min':
            min_err = np.min(curr_err,0)
            err_type[f,:]=min_err
        if type=='minind':
            min_idx = np.argmin(avg_err)
            err = curr_err[min_idx]
            err_type[f,:]=err
        if type=='maxind':
            max_idx = np.argmax(avg_err)
            err = curr_err[max_idx]
            err_type[f,:]=err
        if type=='max':
            max_err = np.max(curr_err, 0)
            err_type[f,:]=max_err
        if type=='avg':
            med_idx = np.argmin(np.abs(avg_err-np.median(avg_err)))
            err = curr_err[med_idx]
            err_type[f,:]=err
    return err_type

b_maxi = get_err_type(b_err,'maxind',num_points)
w_maxi = get_err_type(w_err,'maxind',num_points)
bw_maxi = get_err_type(errs,'maxind',num_points)

b_avg = get_err_type(b_err,'avg',num_points)
w_avg = get_err_type(w_err,'avg',num_points)
bw_avg = get_err_type(errs,'avg',num_points)

b_mini = get_err_type(b_err,'minind',num_points)
w_mini = get_err_type(w_err,'minind',num_points)
bw_mini = get_err_type(errs,'minind',num_points)

b_min = get_err_type(b_err,'min',num_points)
w_min = get_err_type(w_err,'min',num_points)
bw_min = get_err_type(errs,'min',num_points)

b_max = get_err_type(b_err,'max',num_points)
w_max = get_err_type(w_err,'max',num_points)
bw_max = get_err_type(errs,'max',num_points)

compare = [b_mini,b_maxi,b_avg,b_min,b_max,err_b]
type = ['Best Individual','Worst Individual','AVG','Best Possible','Worst Possible','All']
pix_per_cm = 37.8
for c in range(len(compare)):
    curr_se = compare[c]
    sorted_curr_se = np.sort(curr_se.ravel())/pix_per_cm
    curr_ls = np.linspace(0,1,len(sorted_curr_se))
    plt.plot(sorted_curr_se,curr_ls,lw=1,label=type[c])
plt.xlim([0,3])
plt.ylim([0,1])
plt.legend(loc='best')
plt.title('Top view Resident mouse: AMT workers error VS ground truth')
plt.xlabel('Error radius (cm)')
plt.ylabel('Fraction of predictions within a given error radius')
plt.savefig(savedir+'AMT_B_err.png')
plt.savefig(savedir+'AMT_B_err.pdf')

compare = [w_mini,w_maxi,w_avg,w_min,w_max,err_w]
pix_per_cm = 37.8
for c in range(len(compare)):
    curr_se = compare[c]
    sorted_curr_se = np.sort(curr_se.ravel())/pix_per_cm
    curr_ls = np.linspace(0,1,len(sorted_curr_se))
    plt.plot(sorted_curr_se,curr_ls,lw=1,label=type[c])
plt.xlim([0,3])
plt.ylim([0,1])
plt.legend(loc='best')
plt.title('Top view Intruder mouse: AMT workers error VS ground truth')
plt.xlabel('Error radius (cm)')
plt.ylabel('Fraction of predictions within a given error radius')
plt.savefig(savedir+'AMT_W_err.png')
plt.savefig(savedir+'AMT_W_err.pdf')

compare = [bw_avg,bw_min,bw_max,err_bw]
type = ['AVG','Best','Worst','All']
pix_per_cm = 37.8
for c in range(len(type)):
    curr_se = compare[c]
    sorted_curr_se = np.sort(curr_se.ravel())/pix_per_cm
    curr_ls = np.linspace(0,1,len(sorted_curr_se))
    plt.plot(sorted_curr_se,curr_ls,lw=1,label=type[c])
cmin = np.sort(compare[1].ravel())/pix_per_cm
cmax = np.sort(compare[2].ravel())/pix_per_cm
ls = np.linspace(0,1,len(cmin))
plt.fill_betweenx(ls,cmin,cmax,where=cmax>=cmin,facecolor='gray',interpolate=True,alpha=0.1)
plt.xlim([0,3])
plt.ylim([0,1])
plt.legend(loc='best')
plt.title('Top view: AMT workers performance VS ground truth')
plt.xlabel('Error radius (cm)')
plt.ylabel('Fraction of predictions within a given error radius')
plt.savefig(savedir+'top_AMT_BW_err.png')
plt.savefig(savedir+'top_AMT_BW_err.pdf')

compare = [b_mini,b_maxi,b_avg,b_min,b_max]
type = ['Best Individual','Worst Individual','AVG','Best Possible','Worst Possible']
fig=plt.figure(figsize=(15,8))
plt.subplots_adjust(wspace =.3,hspace=.5)
for c in range(len(type)):
# worst best avg each keypoint
    for k in range(num_points):
        plt.subplot(3,3,k+1)
        curr_se = compare[c]
        sorted_curr_se = np.sort(curr_se[:,k])/pix_per_cm
        curr_ls = np.linspace(0,1,len(sorted_curr_se))
        plt.plot(sorted_curr_se,curr_ls,lw=1,label=type[c])
        plt.xlim([0,1])
        if c==0:plt.title(parts[k])
    if c==len(compare)-1:plt.legend(loc='best')
fig.suptitle('Top view: AMT workers error VS ground truth')
fig.text(0.5, 0.04, 'Error radius (cm)', ha='center')
fig.text(0.04, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical')


compare = [w_mini,w_avg,w_maxi,w_min,w_max,err_w]
fig=plt.figure(figsize=(15,8))
plt.subplots_adjust(wspace =.3,hspace=.5)
for c in range(len(type)):
# worst best avg each keypoint
    for k in range(num_points):
        plt.subplot(3,3,k+1)
        curr_se = compare[c]
        sorted_curr_se = np.sort(curr_se[:,k])/pix_per_cm
        curr_ls = np.linspace(0,1,len(sorted_curr_se))
        plt.plot(sorted_curr_se,curr_ls,lw=1,label=type[c])
        plt.xlim([0,1])
        if c==0:plt.title(parts[k])
    if c==len(compare)-1:plt.legend(loc='best')
fig.suptitle('Top view: AMT workers error VS ground truth')
fig.text(0.5, 0.04, 'Error radius (cm)', ha='center')
fig.text(0.04, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical')


compare = [bw_mini,bw_avg,bw_maxi,bw_min,bw_max]
type = ['Best Individual','AVG','Worst Individual','Best Possible','Worst Possible']
fig=plt.figure(figsize=(15,15))
plt.subplots_adjust(wspace =.3,hspace=.5)
for c in range(len(type)):
# worst best avg each keypoint
    for k in range(num_points):
        plt.subplot(5,3,k+1)
        curr_se = compare[c]
        sorted_curr_se = np.sort(curr_se[:,k])/pix_per_cm
        curr_ls = np.linspace(0,1,len(sorted_curr_se))
        plt.plot(sorted_curr_se,curr_ls,lw=1,label=type[c])
        plt.xlim([0,2])
        if c==0:plt.title(parts[k])
plt.legend(bbox_to_anchor=(1.2,.55))
fig.suptitle('Front view: AMT workers performance VS ground truth')
fig.text(0.5, 0.04, 'Error radius (cm)', ha='center')
fig.text(0.04, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical')
plt.savefig(savedir+'front_AMT_BW_err_part.png')
plt.savefig(savedir+'front_AMT_BW_err_part.pdf')


compare = [bw_mini,bw_avg,bw_maxi,bw_min,bw_max]
type = ['Best Individual','AVG','Worst Individual','Best Possible','Worst Possible']
fig=plt.figure(figsize=(15,15))
plt.subplots_adjust(wspace =.3,hspace=.5)
for k in range(num_points):
    plt.subplot(5, 3, k + 1)
    for c in range(len(type)):# worst best avg each keypoint
        curr_se = compare[c]
        sorted_curr_se = np.sort(curr_se[:,k])/pix_per_cm
        curr_ls = np.linspace(0,1,len(sorted_curr_se))
        plt.plot(sorted_curr_se,curr_ls,lw=1,label=type[c])
        plt.xlim([0,2])
        if c==0:plt.title(parts[k])
    cmin = np.sort(compare[2][:,k])/pix_per_cm
    cmax = np.sort(compare[0][:,k])/pix_per_cm
    ls = np.linspace(0,1,len(cmin))
    plt.fill_betweenx(ls,cmin,cmax,where=cmax<=cmin,facecolor='gray',interpolate=True,alpha=0.5)
plt.legend(bbox_to_anchor=(1.2,.55))
fig.suptitle('Front view: AMT workers performance VS ground truth')
fig.text(0.5, 0.04, 'Error radius (cm)', ha='center')
fig.text(0.04, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical')
plt.savefig(savedir+'front_AMT_BW_err_part.png')
plt.savefig(savedir+'front_AMT_BW_err_part.pdf')