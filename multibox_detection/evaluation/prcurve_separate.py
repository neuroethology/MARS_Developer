import sys, os
sys.path.append('./evaluation')

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

path ='../../results_detection/top_separate_allset/black/'
save_name = path +'pr_curve'
with open(path + 'cocoEval.pkl', 'rb') as fp: cocoEval = pickle.load(fp)
print 'load'

#### PR CURVE
rs_mat = cocoEval.params.recThrs #[0:.01:1] R=101 recall thresholds for evaluation
ps_mat  = cocoEval.eval['precision']
iou_mat = cocoEval.params.iouThrs
arearng_mat = cocoEval.params.areaRng


jet = cmx = plt.get_cmap('jet')
cNorm = colors.Normalize(vmin = 0,vmax = len(iou_mat))
scalarMap = cm.ScalarMappable(norm=cNorm,cmap= jet)

show=[.5,.75,.85,.8999999999999999,.95]
fig,ax = plt.subplots(1)
for i in range(len(iou_mat)):
    if iou_mat[i] in show:
        colorVal = scalarMap.to_rgba(i)
        ax.plot(rs_mat,ps_mat[i,:,:,0,1],c=colorVal,ls='-',lw=2,label = 'IoU = %s' % np.round(iou_mat[i],2))
plt.grid()
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(loc='best')
# plt.legend(bbox_to_anchor=(1.1, 0.85),fontsize=12)
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.savefig(save_name+'.png')
plt.savefig(save_name+'.pdf')
plt.show()


