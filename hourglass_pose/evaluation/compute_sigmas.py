import os, sys
import cPickle as pickle
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist


with open('../tf_dataset_detection/AMT_data/front/AMT15_front_csv.pkl','rb') as fp:
    D = pickle.load(fp)

# med_x = np.zeros((1,9,20000))
# med_y =np.zeros((1,9,20000))
# area_vec = []
# X = np.zeros((5,9,20000))
# Y= np.zeros((5,9,20000))
#
# for i in range(0,20000,2):
#     X[:,:,i] =D[i/2]['ann_B']['X'][:5]*1024
#     X[:,:,i+1] =D[i/2]['ann_W']['X'][:5]*1024
#     Y[:,:,i] =D[i/2]['ann_B']['Y'][:5]*570
#     Y[:,:,i+1] =D[i/2]['ann_W']['Y'][:5] * 570
#
#     #mu for all points and b w mouse and x y
#     med_x[:,:,i]=D[i/2]['ann_B']['med'][0]*1024
#     med_x[:,:,i+1]=D[i/2]['ann_W']['med'][0]*1024
#     med_y[:,:,i]=D[i/2]['ann_B']['med'][1]*570
#     med_y[:,:,i+1]=D[i/2]['ann_W']['med'][1]*570
#     area_vec.append(np.sqrt(D[i/2]['ann_B']['area']))
#     area_vec.append(np.sqrt(D[i/2]['ann_W']['area']))
#
# area_vec = np.array(area_vec)
#
# d = np.zeros((len(X),7))
# for i in range(len(X)):
#     for j in range(7):
#         for k in range(5):
#             d[i,j] += (((X[k, j,i]-med_x[0,j,i])**2 + (Y[k,j,i]-med_y[0,j,i])**2))
#     d[i,:]/= np.sqrt(area_vec[i])
#
# std = np.zeros(7)
# for j in range(7):
#     top_d = d[:,j]
#     top_d.sort()
#     top_d = top_d[:int(len(top_d)*.99)]
#     std[j] = np.sqrt(np.mean(top_d))


# sigmas = [0.18613165,  0.23771644,  0.2125576 ,  0.24509726,  0.6945994 , 0.53104378,  0.20537745]




##################################################################################################################

area_vec = []
X = []
Y= []

for j in range(9):
    x = []
    y = []
    for i in range(len(D)):
        x.append((D[i]['ann_B']['X'][:,j]*1024).tolist())
        x.append((D[i]['ann_W']['X'][:,j]*1024).tolist())
        y.append((D[i]['ann_B']['Y'][:,j]*570).tolist())
        y.append((D[i]['ann_W']['Y'][:,j]*570).tolist())
    X.append(x)
    Y.append(y)

for i in range(len(D)):
    area_vec.append(D[i]['ann_B']['area'])
    area_vec.append(D[i]['ann_W']['area'])

X = np.asarray(X)
Y = np.asarray(Y)
area_vec = np.asarray(area_vec)

D = np.zeros((len(X),len(X[0])))
sigma = np.zeros(len(X))
for j in range(len(X)):
    for i in range(len(X[0])):
        xy = np.asarray([X[j][i], Y[j][i]]).reshape((2,len(X[j][i]))).T
        D[j][i] = np.mean(dist.pdist(xy)) / area_vec[i]
    sigma[j] = np.mean(D[j])

squared_sigma = np.sqrt(sigma)
sigma2 = 2 * squared_sigma

array([ 0.04208979,  0.04591051,  0.04600008,  0.04739044,  0.0609084 ,
        0.06061174,  0.04772672,  0.06143189,  0.06792349])



##############################front
area_vec = []
X = []
Y= []

for j in range(13):
    x = []
    y = []
    for i in range(len(D)):
        x.append((D[i]['ann_B']['X'][:,j]*1280).tolist())
        x.append((D[i]['ann_W']['X'][:,j]*1280).tolist())
        y.append((D[i]['ann_B']['Y'][:,j]*500).tolist())
        y.append((D[i]['ann_W']['Y'][:,j]*500).tolist())
    X.append(x)
    Y.append(y)

for i in range(len(D)):
    area_vec.append(D[i]['ann_B']['area'])
    area_vec.append(D[i]['ann_W']['area'])

X = np.asarray(X)
Y = np.asarray(Y)
area_vec = np.asarray(area_vec)

D = np.zeros((len(X),len(X[0])))
sigma = np.zeros(len(X))
for j in range(len(X)):
    for i in range(len(X[0])):
        xy = np.asarray([X[j][i], Y[j][i]]).reshape((2,len(X[j][i]))).T
        D[j][i] = np.mean(dist.pdist(xy)) / area_vec[i]
    sigma[j] = np.mean(D[j])

squared_sigma = np.sqrt(sigma)
sigma2 = 2 * squared_sigma

array([0.03465136, 0.0344434 , 0.03461611, 0.0362902 , 0.04186067,
       0.04204246, 0.03432178, 0.04125073,0.04113029, 0.04016324, 0.04140253])
