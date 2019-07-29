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
import argparse
import json
sys.path.append('.')
from config import parse_config_file


def get_err_btw_kps(gt,pd,x_scale,y_scale,num_points):
    medians = gt['med']
    gt_X = medians[1, :num_points]*x_scale
    gt_Y = medians[0, :num_points]*y_scale

    pd_X = np.array(pd[::3])[:num_points]
    pd_Y = np.array(pd[1::3])[:num_points]

    err_x = pd_X - gt_X
    err_y = pd_Y - gt_Y

    pixelwise_error = np.sqrt(err_x**2 + err_y**2)
    return pixelwise_error

def annotations_to_error(annotation_dict,x_scale,y_scale,num_points):
    raw_X = annotation_dict['Y'][:,:num_points]
    raw_Y = annotation_dict['X'][:,:num_points]
    medians = annotation_dict['med']
    gt_X = medians[1,:num_points]
    gt_Y = medians[0,:num_points]

    err_x = raw_X - gt_X
    err_y = raw_Y - gt_Y
    err_x *= x_scale
    err_y *= y_scale

    pixelwise_error = np.sqrt(err_x ** 2 + err_y ** 2)

    return pixelwise_error

def get_err_type(cell_err,type,num_points):
        err_type = np.zeros((len(cell_err),num_points))
        for f in range(len(cell_err)):
            curr_err= cell_err[f]
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

def plot_pck(cell,type):
    pix_per_cm=37.8
    for k in range(len(cell)):
        curr_se = cell[k]
        sorted_curr_se = np.sort(curr_se.ravel())/pix_per_cm
        curr_ls = np.linspace(0,1.,len(sorted_curr_se))
        if 'mars' in type[k].lower():
            plt.plot(sorted_curr_se,curr_ls,label=type[k],linestyle='-.',lw=1.5)
        else:
            plt.plot(sorted_curr_se,curr_ls,label=type[k])

def plot_pck_fill(cell,type):
    pix_per_cm=37.8
    for k in range(len(cell)):
        curr_se = cell[k]
        sorted_curr_se = np.sort(curr_se.ravel())/pix_per_cm
        curr_ls = np.linspace(0,1.,len(sorted_curr_se))
        if 'mars' in type[k].lower():
            plt.plot(sorted_curr_se,curr_ls,label=type[k],linestyle='-.',lw=1.5)
        else:
            plt.plot(sorted_curr_se,curr_ls,label=type[k])
    cmin = np.sort(cell[0].ravel())/pix_per_cm
    cmax = np.sort(cell[2].ravel())/pix_per_cm
    ls = np.linspace(0,1,len(cmin))
    plt.fill_betweenx(ls,cmin,cmax,where=cmax>=cmin,facecolor='gray',interpolate=True,alpha=0.1)
    plt.xlim([0,3])
    plt.ylim([0,1])

def plot_pck_kpt(cell,type,parts,view):

    pix_per_cm=37.8
    for c in range(len(type)):
        for k in range(len(parts)):
            plt.subplot(2 if view.lower()=='top' else 3,4,k+1)
            curr_se = cell[c]
            sorted_curr_se = np.sort(curr_se[:,k])/pix_per_cm
            curr_ls = np.linspace(0,1.,len(sorted_curr_se))
            if 'mars' in type[c].lower():
                plt.plot(sorted_curr_se, curr_ls, label=type[c], linestyle='-.', lw=1.5)
            else:
                plt.plot(sorted_curr_se, curr_ls, label=type[c])
            plt.xlim([0,3])
            if c==0:plt.title(parts[k])
        if c==len(cell)-1:plt.legend(loc='best')

def plot_pck_kpt_fill(cell,type,parts,view):
    pix_per_cm=37.8
    for k in range(len(parts)):
        plt.subplot(2 if view.lower()=='top' else 3, 4, k + 1)
        for c in range(len(type)):# worst best avg each keypoint
            curr_se = cell[c]
            sorted_curr_se = np.sort(curr_se[:,k])/pix_per_cm
            curr_ls = np.linspace(0,1,len(sorted_curr_se))
            if 'mars' in type[c].lower():
                plt.plot(sorted_curr_se, curr_ls, label=type[c], linestyle='-.', lw=1.5)
            else:
                plt.plot(sorted_curr_se, curr_ls, label=type[c])
            plt.xlim([0,3])
            if c==0:plt.title(parts[k])
        cmin = np.sort(cell[2][:,k])/pix_per_cm
        cmax = np.sort(cell[0][:,k])/pix_per_cm
        ls = np.linspace(0,1,len(cmin))
        plt.fill_betweenx(ls,cmin,cmax,where=cmax<=cmin,facecolor='gray',interpolate=True,alpha=0.4)


DATASET_DIR='../tf_dataset_keypoints/top_allset_j/'
EXPERIMENT_DIR='../results_keypoints_j/top_fixed/'



amt_file =  '../tf_dataset_detection/AMT_data/top/AMT15_top_csv.pkl'
#load extracted info from annotations


def get_perfs(res_file,amt_file,savedir, view,num_points):

    D = pkl.load(open(amt_file, 'rb')) # normalized to image size
    # with open(path5old,'rb') as fp:    Dm_old = pkl.load(fp)
    w = D[0]['width']
    h = D[0]['height']
    parts = np.array(D[0]['ann_label'])[:num_points]

    res_file = res_file + 'results_pose_test.pkl'
    results = pkl.load(open(res_file,'rb')) # image scale already
    pd = results[0]
    pd_b = pd[::2]
    pd_w = pd[1::2]

    # Initialize the lists to store errors.
    b_err = []
    w_err = []
    pd_b_err=[]
    pd_w_err=[]

    # Loop over the size of the annotation set.
    for i in range(len(pd_b)):
        # Give an update on how far we are.
        filename = pd_b[i]['filename']
        idx = [f for f in range(len(D)) if D[f]['frame_name'].split('/')[-1]==filename][0]
        d=D[idx]
        # Parse out the annotations.
        black_annotations = d['ann_B']
        white_annotations = d['ann_W']

        black_pd = pd_b[i]['keypoints']
        white_pd = pd_w[i]['keypoints']

        # amt error
        # Convert annotations to error.
        b_err.append(annotations_to_error(black_annotations,w,h,num_points))# for each annotator x # for each k
        w_err.append(annotations_to_error(white_annotations,w,h,num_points))

        #amt vs pred
        pd_b_err.append(get_err_btw_kps(black_annotations,black_pd,w,h,num_points))
        pd_w_err.append(get_err_btw_kps(white_annotations,white_pd,w,h,num_points))

    # Bind the errors together.
    err_amt=b_err; err_amt.extend(w_err) # for workers
    err_b=[[] for _ in range(num_points)]
    err_w=[[] for _ in range(num_points)]
    for k in range(num_points):
        for f in range(len(pd_b)):
            err_b[k].extend(b_err[f][:,k])
            err_w[k].extend(w_err[f][:,k])

    err_b=np.asarray(err_b)
    err_w=np.asarray(err_w)
    err_bw = np.hstack((err_b,err_w))

    err_b_pd = np.array(pd_b_err).T
    err_w_pd = np.array(pd_w_err).T
    err_bw_pd=pd_b_err; err_bw_pd.extend(pd_w_err)
    err_bw_pd=np.array(err_bw_pd).T

    # wort best avg all keypoints

    b_maxi = get_err_type(b_err,'maxind',num_points)
    w_maxi = get_err_type(w_err,'maxind',num_points)
    bw_maxi = get_err_type(err_amt,'maxind',num_points)

    b_avg = get_err_type(b_err,'avg',num_points)
    w_avg = get_err_type(w_err,'avg',num_points)
    bw_avg = get_err_type(err_amt,'avg',num_points)

    b_mini = get_err_type(b_err,'minind',num_points)
    w_mini = get_err_type(w_err,'minind',num_points)
    bw_mini = get_err_type(err_amt,'minind',num_points)

    b_min = get_err_type(b_err,'min',num_points)
    w_min = get_err_type(w_err,'min',num_points)
    bw_min = get_err_type(err_amt,'min',num_points)

    b_max = get_err_type(b_err,'max',num_points)
    w_max = get_err_type(w_err,'max',num_points)
    bw_max = get_err_type(err_amt,'max',num_points)

    ##########################################################################################################

    # plot curve all keypoints
    fig=plt.figure()
    plt.plot(np.sort(err_bw.ravel()),np.linspace(0,1,len(err_bw.ravel())),'b-',lw=2,label='AMT annotations')
    plt.plot(np.sort(err_bw_pd.ravel()),np.linspace(0,1,len(err_bw_pd.ravel())),'r-',lw=2,label='MARS predictions')
    plt.title(view[0].upper() + view[1:] + ' view, PCK: AMT vs MARS ')
    plt.xlim([0,100])
    plt.xlabel('Error radius (px)')
    plt.ylabel('Fraction of ground truth within a given error radius')
    plt.legend()
    plt.savefig(savedir + view +'_amt_vs_mars_BW.png')
    plt.savefig(savedir + view +'_amt_vs_mars_BW.pdf')

    ##### each keypoint
    fig=plt.figure(figsize=(15,15))
    for k in range(num_points):
        plt.subplot(3,3,k+1)
        plt.plot(np.sort(err_bw[k].ravel()), np.linspace(0, 1, len(err_bw[k].ravel())), 'k-',label='AMT both')
        plt.plot(np.sort(err_bw_pd[k].ravel()), np.linspace(0, 1, len(err_bw_pd[k].ravel())), 'k--',label='MARS both')
        plt.plot(np.sort(err_b[k].ravel()), np.linspace(0, 1, len(err_b[k].ravel())), 'b-', label='AMT resident')
        plt.plot(np.sort(err_b_pd[k].ravel()), np.linspace(0, 1, len(err_b_pd[k].ravel())), 'b--', label='MARS resident')
        plt.plot(np.sort(err_w[k].ravel()), np.linspace(0, 1, len(err_w[k].ravel())), 'r-', label='AMT intruder')
        plt.plot(np.sort(err_w_pd[k].ravel()), np.linspace(0, 1, len(err_w_pd[k].ravel())), 'r--', label='MARS intruder')
        plt.xlim([0,100])
        plt.title(parts[k])
        if k==0: plt.legend(loc='best')
    fig.suptitle(view[0].upper() + view[1:] +' view, PCK keypoint breakdown: AMT vs MARS ')
    fig.text(0.5, 0.02, 'Error radius (px)', ha='center',fontsize=12)
    fig.text(0.06, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical',fontsize=12)
    plt.savefig(savedir + view +'_amt_vs_mars_BW_part.png')
    plt.savefig(savedir + view +'_amt_vs_mars_BW_part.pdf')

    # compare = [b_mini,b_avg,b_maxi,err_b_pd]
    # type = ['AMT Best Individual','AMT Avg','AMT Worst Individual','MARS All']
    # plot_pck(compare,type)
    # plt.xlim([0,3])
    # plt.ylim([0,1])
    # plt.legend(loc='best')
    # plt.title(view[0].upper() + view[1:] +' view, Resident mouse: AMT VS MARS PCK')
    # plt.xlabel('Error radius (cm)')
    # plt.ylabel('Fraction of predictions within a given error radius')
    # plt.savefig(savedir+view +'_amt_mars_B_err_bestworst.png')
    # plt.savefig(savedir+view +'_amt_mars_B_err_bestworst.pdf')
    #
    # compare = [w_mini,w_avg,w_maxi,err_w_pd]
    # plot_pck(compare,  type)
    # plt.xlim([0,3])
    # plt.ylim([0,1])
    # plt.legend(loc='best')
    # plt.title(view[0].upper() + view[1:] +' view, Intruder mouse: AMT VS MARS PCK')
    # plt.xlabel('Error radius (cm)')
    # plt.ylabel('Fraction of predictions within a given error radius')
    # plt.savefig(savedir+view+'_amt_mars_W_err_bestworst.png')
    # plt.savefig(savedir+view + '_amt_mars_W_err_bestworst.pdf')

    # compare = [b_mini,b_min,b_avg,b_maxi,b_max,err_b,err_b_pd]
    # type = ['AMT Best Individual', 'AMT Best Possible','AMT Avg','AMT Worst Individual','AMT Worst Possible','AMT All','MARS All']
    # plot_pck(compare,type)
    # plt.xlim([0,3])
    # plt.ylim([0,1])
    # plt.legend(loc='best')
    # plt.title(view[0].upper() + view[1:] +' view, Resident mouse: AMT VS MARS PCK')
    # plt.xlabel('Error radius (cm)')
    # plt.ylabel('Fraction of predictions within a given error radius')
    # plt.savefig(savedir+view +'_amt_mars_B_err_bestworst.png')
    # plt.savefig(savedir+view +'_amt_mars_B_err_bestworst.pdf')

    # compare = [w_mini,w_min,w_avg,w_maxi,w_max,err_w,err_w_pd]
    # plot_pck(compare,  type)
    # plt.xlim([0,3])
    # plt.ylim([0,1])
    # plt.legend(loc='best')
    # plt.title(view[0].upper() + view[1:] +' view Intruder mouse: AMT workers error VS MARS')
    # plt.xlabel('Error radius (cm)')
    # plt.ylabel('Fraction of predictions within a given error radius')
    # plt.savefig(savedir+view+'_amt_mars_W_err.png')
    # plt.savefig(savedir+view + '_amt_mars_W_err.pdf')

    fig=plt.figure()
    compare = [bw_mini,bw_avg,bw_maxi,err_bw_pd]
    type = ['AMT Best','AMT Avg','AMT Worst','MARS']
    plot_pck_fill(compare, type)
    plt.legend(loc='best')
    plt.title(view[0].upper() + view[1:]+  ' view PCK: AMT vs MARS ')
    plt.xlabel('Error radius (cm)')
    plt.ylabel('Fraction of predictions within a given error radius')
    plt.savefig(savedir+view +'_amt_mars_BW_err_bestworst.png')
    plt.savefig(savedir+view +'_amt_mars_BW_err_bestworst.pdf')

    # for each part
    # compare = [b_mini,b_avg,b_maxi,err_b_pd.T]
    # type = ['AMT Best','AMT AVG','AMT Worst','MARS']
    # fig = plt.figure(figsize=(15, 15))
    # plt.subplots_adjust(wspace=.3, hspace=.5)
    # plot_pck_kpt(compare,type,parts,view)
    # fig.suptitle(view[0].upper() + view[1:]+ ' view Resident: AMT VS MARS PCK single keypoint')
    # fig.text(0.5, 0.04, 'Error radius (cm)', ha='center',fontsize=12)
    # fig.text(0.04, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical',fontsize=12)
    # plt.savefig(savedir+view +'_amt_mars_B_err_part_bestworst.png')
    # plt.savefig(savedir+view +'_amt_mars_B_err_part_bestworst.pdf')
    #
    # compare = [w_mini,w_avg,w_maxi,err_w_pd.T]
    # fig = plt.figure(figsize=(15, 15))
    # plt.subplots_adjust(wspace=.3, hspace=.5)
    # plot_pck_kpt(compare, type, parts,view)
    # fig.suptitle(view[0].upper() + view[1:] + ' view Intruder: AMT VS MARS PCK')
    # fig.text(0.5, 0.04, 'Error radius (cm)', ha='center',fontsize=12)
    # fig.text(0.04, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical',fontsize=12)
    # plt.savefig(savedir + view + '_amt_mars_W_err_part.png')
    # plt.savefig(savedir + view + '_amt_mars_W_err_part.pdf')
    fig=plt.figure()
    compare = [bw_mini,bw_avg,bw_maxi,err_bw_pd.T]
    fig=plt.figure(figsize=(18,10))
    plt.subplots_adjust(wspace=.3, hspace=.5)
    plot_pck_kpt(compare, type, parts,view)
    fig.suptitle(view[0].upper() + view[1:] + ' view, PCK keypoint breakdown: AMT vs MARS')
    plt.legend(bbox_to_anchor=(1.5,.55))
    fig.text(0.5, 0.04, 'Error radius (cm)', ha='center',fontsize=12)
    fig.text(0.04, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical',fontsize=12)
    plt.savefig(savedir+view +'_amt_mars_BW_err_bestworst_part.png')
    plt.savefig(savedir+view +'_amt_mars_BW_err_bestworst_part.pdf')

    fig=plt.figure(figsize=(18,10))
    compare = [bw_mini,bw_avg,bw_maxi,err_bw_pd.T]
    type = ['AMT Best','AMT AVG','AMT Worst','MARS']
    plt.subplots_adjust(wspace =.3,hspace=.5)
    plot_pck_kpt_fill(compare,type,parts,view)
    plt.legend(bbox_to_anchor=(1.5,.55))
    fig.suptitle(view[0].upper() + view[1:] + ' view, PCK keypoint breakdown: AMT vs MARS ')
    fig.text(0.5, 0.04, 'Error radius (cm)', ha='center',fontsize=12)
    fig.text(0.04, 0.5, 'Fraction of predictions within a given error radius', va='center', rotation='vertical',fontsize=12)
    plt.savefig(savedir+view +'_amt_mars_BW_err_fill_bestworst_part.png')
    plt.savefig(savedir+view +'_amt_mars_BW_err_fill_bestworst_part.pdf')


def parse_args():
    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    #file generate with my_prcurve_prep_data
    parser.add_argument('--res_path', dest='res_path',
                        help='paths to results files', type=str,
                        required=True)

    parser.add_argument('--save_dir', dest='savedir',
                        help='the fullpath (without extension) to save the file to', type=str,
                        required=True)

    parser.add_argument('--view', dest='view',
                        help='top or front view', type=str,
                        required=True)

    parser.add_argument('--amt_path', dest='amt_path',
                        help='configuration file ', type=str,
                        required=True)

    parser.add_argument('--num_points', dest='num_points',
                        help='number of points ', type=int,
                        required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    get_perfs(args.res_path,args.amt_path, args.savedir,args.view,args.num_points)
