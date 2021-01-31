###############################################################################################
# Plot annotation on image frame
###############################################################################################
# Given the dict of all frames and a frame number it plots the image frame, annotated keypoints, median keypoints,
# the skeleton, bboxes, ellipses, oriented bboxes
#usage: plot_annotations(filename,im_path,0,0,1,0,0,1,0,0)


import os
import sys
from matplotlib import pyplot as plt
import Image
import pdb
import sys
import cPickle as pickle




def plot_annotations(D,f, plt_keypoints=0, plt_med=0, plt_skel=0,plt_bbox=0,plt_ellipse0=0,plt_ellipse1=0,plt_ori1=0,plt_ori2=0, plt_tbox=0,plt_ntbox=0):

    # with open(filename,'rb') as fp:
    #     D = pickle.load(fp)
    # print 'load'

    F = D[f]
    h = F['height']
    w = F['width']

    #keypints
    XB = F['ann_B']['X']*h
    YB = F['ann_B']['Y']*w
    XW = F['ann_W']['X']*h
    YW = F['ann_W']['Y']*w

    #median points
    mXB = F['ann_B']['med'][0]*h
    mYB = F['ann_B']['med'][1]*w
    mXW = F['ann_W']['med'][0]*h
    mYW = F['ann_W']['med'][1]*w

    fig, ax = plt.subplots(1)
    ax.imshow(Image.open('./' + F['frame_name']), cmap='Greys_r')

    #plot all points for all annotators
    if plt_keypoints:
        ax.plot(YB,XB,'.')
        ax.plot(YW,XW,'.')

    #plot median points
    if plt_med:
        ax.plot(mYB,mXB,'.',c='b',markersize=12)
        ax.plot(mYW,mXW,'.',c='r',markersize=12)

    #plot segments skeleton
    if plt_skel:
        ax.plot([mYB[0],mYB[1],mYB[2],mYB[0]],[mXB[0],mXB[1],mXB[2],mXB[0]],c='blue')
        ax.plot([mYB[3],mYB[4],mYB[6],mYB[5],mYB[3]],[mXB[3],mXB[4],mXB[6],mXB[5],mXB[3]],c='blue')
        ax.plot([mYB[6],mYB[7],mYB[8]],[mXB[6],mXB[7],mXB[8]],c='blue')

        ax.plot([mYW[0],mYW[1],mYW[2],mYW[0]],[mXW[0],mXW[1],mXW[2],mXW[0]],c='red')
        ax.plot([mYW[3],mYW[4],mYW[6],mYW[5],mYW[3]],[mXW[3],mXW[4],mXW[6],mXW[5],mXW[3]],c='red')
        ax.plot([mYW[6],mYW[7],mYW[8]],[mXW[6],mXW[7],mXW[8]],c='red')

    #plot boxes
    if plt_bbox:
        xmin = F['ann_B']['bbox'][0] * w
        xmax = F['ann_B']['bbox'][1] * w
        ymin = F['ann_B']['bbox'][2] * h
        ymax = F['ann_B']['bbox'][3] * h
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')

        xmin = F['ann_W']['bbox'][0] * w
        xmax = F['ann_W']['bbox'][1] * w
        ymin = F['ann_W']['bbox'][2] * h
        ymax = F['ann_W']['bbox'][3] * h
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')

    # cB, raB, rbB, alphaB, xbtB, ybtB, xbB, ybB, xsB, ysB, ori_vec_vB, ori_vec_hB = fit_ellipse(mYB[:7], mXB[:7])
    # cW, raW, rbW, alphaW, xbtW, ybtW, xbW, ybW, xsW, ysW, ori_vec_vW, ori_vec_hW = fit_ellipse(mYW[:7], mXW[:7])

    # if plt_ellipse1:
    #     # plot center
    #     ax.plot(cB[0], cB[1], 'bd')
    #     ax.plot(cW[0], cW[1], 'rd')
    #
    #     # plot ellipse
    #     ax.plot(xsB, ysB,'b-')
    #     ax.plot(xsW, ysW,'r-')
    #
    #     if plt_ori1:
    #         # plot orientation
    #         ax.plot([cB[0], cB[0] + ori_vec_vB[0]], [cB[1], cB[1] + ori_vec_vB[1]], 'b-')
    #         ax.plot([cB[0], cB[0] + ori_vec_hB[0]], [cB[1], cB[1] + ori_vec_hB[1]], 'b-')
    #         ax.plot([cW[0], cW[0] + ori_vec_vW[0]], [cW[1], cW[1] + ori_vec_vW[1]], 'r-')
    #         ax.plot([cW[0], cW[0] + ori_vec_hW[0]], [cW[1], cW[1] + ori_vec_hW[1]], 'r-')
    #     pdb.set_trace()
    #     if plt_ori2:
    #         ax.plot([cB[0],mYB[0]], [cB[1], mXB[0]], 'b-')
    #         ax.plot([cW[0],mYW[0]], [cW[1], mXB[0]], 'r-')


    # if plt_ntbox:
    #     # plot bbox non tilted
    #     ax.plot(xbB, ybB, 'b-')
    #     ax.plot(xbW, ybW, 'r-')

    # if plt_tbox:
    #     # plot bbox tilted
    #     ax.plot(xbtB, ybtB, 'b-')
    #     ax.plot(xbtW, ybtW, 'r-')

    #plot other ellipse function
    # cRowB, cColB, raB, rbB, phiB, xsB, ysB, oxB, oyB = get_param_ellipse(mXB[:7],mYB[:7])
    # cRowW, cColW, raW, rbW, phiW, xsW, ysW, oxW, oyW = get_param_ellipse(mXW[:7],mYW[:7])
    # if plt_ellipse0:
    #     ax.plot(xsB,ysB)
    #     ax.plot(xsW,ysW)
    #
    #     ax.plot(oxB,oyB)
    #     ax.plot(oxW,oyW)
    #
    #     # ax.plt(orixvB,oriyvB); ax.plt(orixhB,oriyhB)
    #     # ax.plt(orixvW,oriyvW); ax.plt(orixhW,oriyhW)

    # fig.savefig('/media/cristina/2.1TB Data/cristina/python_scripts/tensorflow_play/input/annotation/plot_medoid/'+ F['frame_id'] + '_' + format(f, '04d'))
    # plt.close('all')

filename = '../tf_dataset_detection/top/AMT10K_csv_actions.pkl'
im_path = './'

with open(filename, 'rb') as fp:
    D = pickle.load(fp)
print 'load'
plt.ion()

for i in range(10):
    print i
    plot_annotations(D, i, 0,1,0,1)



