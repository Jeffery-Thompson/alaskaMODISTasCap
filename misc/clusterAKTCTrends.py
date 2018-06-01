#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:23:43 2018

script to cluster the Tasseled Cap trends image

@author: jeth6160
"""

import time

import numpy as np
from scipy import ndimage
import matplotlib
import rasterio
from matplotlib import pyplot as plt
from numpy import random
import matplotlib.image as mpimg

#from scipy.cluster.hierarchy import dendrogram, linkage

#from sklearn import manifold, datasets
#from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans

# define sample size for timing machine learing
s_size = [0.001, 0.01, 0.1, 0.15, 0.2]

# set input output directories
iDir = '/Users/jeth6160/Desktop/permafrost/Arctic/WHRC/output/'
iDir2 = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'
iDir3 = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'

iFile = 'TC_TrendsAll_v3'
fExt = '.tif'

inFile = iFile + fExt 

with rasterio.open(iDir3 +  inFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    fIn = src.read()
    fMask = src.read_masks(1)
    tcTrends = fIn.squeeze()
    rP,cP = np.where(src.read_masks(1) )
    crsOut = src.crs
    #rT,cT = np.where(src.read_masks(1) > -9999)
    traOut = src.transform
    affOut = src.affine 
    
bands,rows, cols = tcTrends.shape

#tcFlat = np.concatenate((np.reshape(tcTrends[0,:,:],[rows*cols,1]),np.reshape(tcTrends[1,:,:],[rows*cols,1]),np.reshape(tcTrends[2,:,:],[rows*cols,1])),axis=1)

tcFlat = tcTrends.reshape(1,bands*rows*cols,order='F').reshape(rows*cols,bands)




tc_i = np.where(tcFlat==-9999)
tcFlat[tc_i]=np.nan

tc_valid = np.argwhere(np.isfinite(tcFlat[:,0])).squeeze()
dataClst=tcFlat[tc_valid,:]
dataClst= dataClst.astype(int)
#sample_i = np.random.choice(dataClst.shape[0],size=np.round(dataClst.shape[0]*0.01).astype(int),replace=False)

batch_sz = np.round(dataClst.shape[0]*0.2).astype(int)
#EtestCl = dataClst[sample_i,:].reshape(sample_i.shape[0]*3,1)


mBkMeans = MiniBatchKMeans(n_clusters=9,init='random',max_iter=20,
                           batch_size=batch_sz,verbose=True,
                           compute_labels=True,random_state=42)

#mbkmean = MiniBatchKMeans(n_clusters=9)
[sTime, sClock] = time.time(),time.clock() 
clusterFit= mBkMeans.fit(dataClst)
#mbk_fit = mbkmean.fit(dataClst)

#ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(testCl)
[eTime, eClock] = time.time(),time.clock() 

elapsed_time = eTime -sTime
elapsed_clock = eClock -sClock
print('Elapased time is:',elapsed_time)


clMems = clusterFit.labels_

clImg = tcFlat[:,0]
#clImg=np.ndarray([rows,cols],dtype=int)
#clImg[:,:] = -9999
clImg[tc_valid] = clMems
clImg = clImg.reshape(rows,cols,order='F')

cl_nan = np.argwhere(np.isnan(clImg))
clImg[cl_nan]=-9999
clImg = clImg.astype('int16')
imgplot= plt.imshow(clImg)


with rasterio.open(iDir3 + 'TCTrends_clust'+'.tif', 'w', driver='GTiff', height=rows,
                       width=cols, count=1, dtype='int16',
                       crs=crsOut, transform=traOut, nodata=-9999) as dst:
    dst.write(clImg, 1)








tc_valid = np.argwhere(np.isfinite(tcFlat[:,0]))

sample_i = np.random.choice(tc_valid.shape[0],size=np.round(tc_valid.shape[0]*0.1).astype(int),replace=False)

#tcFlat =[]
#tcFlat = np.reshape(tcTrends,[bands,rows*cols],order='')
for band in range(0,bands):
    t = np.array(tcTrends[band].flatten('C'))
    tcFlat.append(t)

for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(n_clusters=10,affinity='euclidean',linkage='ward')
    t0 = time()
    clustering.fit(tcFlat)
    print("%s : %.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)


plt.show()


f, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.set_title('Green vs Bright')
ax1.scatter(dataClst[:,0],dataClst[:,1])

ax2.set_title('Wet vs Bright')
ax2.scatter(dataClst[:,0],dataClst[:,2])

ax3.set_title('Wet vs Green')
ax3.scatter(dataClst[:,1],dataClst[:,2])
