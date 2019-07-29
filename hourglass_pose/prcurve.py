import cPickle as pickle
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dist
import pdb
from scipy.stats import gaussian_kde
eps = np.spacing(1)
from scipy import stats



# with open ('./results_pose_test_white.pkl','r')  as fp:
#   results = pickle.load(fp)
#
# with open ('./results_pose_test_black.pkl','r')  as fp:
#   results = pickle.load(fp)
def plot_pr(f, figtitle, savename, NUM_PARTS):
    fname = ''.join([f, '/results_pose_test.pkl'])
    with open (fname,'r')  as fp:
      results = pickle.load(fp)

    dt = results[0]
    gt = results[1]
    # NUM_PARTS = 7
    ## turn the keypoints dictionaries into lists of [1,2]-arrays, then flatten them.
    gt_keys = [(np.asarray(gt[k]['keypoints']).reshape(NUM_PARTS,3)[:,:2]).tolist() for k in range(len(gt))]
    gt_keys_flat = np.array([gt_keys[i][k] for i in range(len(gt)) for k in range(NUM_PARTS)])

    dt_keys = [(np.asarray(dt[k]['keypoints']).reshape(NUM_PARTS,3)[:,:2]).tolist() for k in range(len(dt))]
    dt_keys_flat = [dt_keys[i][k] for i in range(len(dt)) for k in range(NUM_PARTS)]

    confs = np.array([dt[i]['score'][k] for i in range(len(dt)) for k in range(NUM_PARTS)])
    distp = np.asarray([dist.euclidean(gt_keys_flat[i],dt_keys_flat[i]) for i in range(len(dt_keys_flat))])
    print 'GT'
    print(dist.euclidean(gt_keys_flat[1],dt_keys_flat[1]))
    print len(gt_keys_flat)
    print len(dt_keys_flat)
    print 'done'

    plt.figure(1)

    #pr curve
    drange =[5.0, 10.0, 15.0, 20.0]
    conf_thr = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)

    P = np.zeros((len(conf_thr)+1,len(drange)))     # tp / pred positives
    R = np.zeros((len(conf_thr)+1,len(drange)))     # tp / positives

    for d in xrange(len(drange)):
        valid = np.less_equal(distp,drange[d]) #the distances are within some threshold
        conf_dist_ok = np.copy(confs[valid]) # get the scores for all the above distances
        # no_valid = np.greater(distp,drange[d])
        # conf_dist_no = np.copy(confs[no_valid])
        pos = np.copy(gt_keys_flat[valid])
        count_pos = len(gt_keys_flat) # There are gt keypoints in the dataset that are not visible; this should not count them
        count_neg = len(gt_keys_flat) - count_pos # As it is, this is identically zero...

        for t in xrange(len(conf_thr)):
            y_ = np.greater_equal(conf_dist_ok,conf_thr[t]) # get all the dt scores that are actually accurate, and above some threshold;
            y = np.greater_equal(confs,conf_thr[t]) # get all the parts that the detector thinks are accurate, and are above some threshold
            # y_no = np.less(conf_dist_no,conf_thr[t]) # tn dist no conf no
            # yd_no = np.greater_equal(conf_dist_no,conf_thr[t]) # fp dist no conf y
            # yc_no = np.less(conf_dist_ok,conf_thr[t]) # fn  dist y conf no
            # count_tn = np.sum(y_no)
            # count_fn = count_neg -  count_tn
            count_tp = np.sum(y_)
            # print t
            # print count_tp
            count_pred = np.sum(y)
            count_fp = count_pred - count_tp

            P[t+1,d] = count_tp / (count_pred + eps)
            R[t,d] = count_tp / (count_pos + eps)
        plt.plot(R[:,d], P[:,d], label='d=' + str(drange[d]))
        for t in range(0,len(conf_thr),10):
            plt.text(R[t,d], P[t,d], str(np.round(conf_thr[t],2)),rotation='vertical',size=8)

    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    plt.legend(loc='right', ncol=1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve %s view' % figtitle)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.savefig(f + savename+'.png')
    plt.savefig(f + savename+'.pdf')

def parse_args():
    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--f', dest='f',
                        help='folder containing the results_pose.pkl file',
                        required=True, type=str)
    parser.add_argument('--t', dest='title',
                        help='title of the pr_curve figure',
                        required=False, type=str, default='')
    parser.add_argument('--s', dest='savename',
                        help='folder containing the results_pose.pkl file',
                        required=False, type=str, default='')
    parser.add_argument('--n', dest='num_parts',
                        help='the number of parts there are',
                        required=False, type=int, default=11)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # print "Configurations:"
    # print pprint.pprint(cfg)

    plot_pr(args.f, args.title, args.savename, args.num_parts)