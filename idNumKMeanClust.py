#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:02:12 2018

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
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm 

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

# set the fraction of the total dataset 
s_size = 0.2

# flatten for clustering
tcFlat = tcTrends.reshape(1,bands*rows*cols,order='F').reshape(rows*cols,bands)
tc_i = np.where(tcFlat==-9999)
tcFlat[tc_i]=np.nan

tc_valid = np.argwhere(np.isfinite(tcFlat[:,0])).squeeze()

dataClst=tcFlat[tc_valid,:]
#dataClst= dataClst.astype(int)
batch_sz = np.round(dataClst.shape[0]*s_size).astype(int)

# do the clusting
nCl = range(1, 20)
inertias=[]

mBkMeans = [MiniBatchKMeans(n_clusters=i,init='random',max_iter=20,
                           batch_size=batch_sz,verbose=False,
                           compute_labels=True,random_state=42)for i in nCl]

clScores = [mBkMeans[i].fit(dataClst).score(dataClst) for i in range(len(mBkMeans))]

# plot the resutls
plt.plot(nCl,clScores)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')  


# 5 clusters seems about where the information starts to become less useful    
mBkMeans5cl = MiniBatchKMeans(n_clusters=5,init='random',max_iter=20,
                           batch_size=batch_sz,verbose=True,
                           compute_labels=True,random_state=42)
clusterFit= mBkMeans5cl.fit(dataClst)
clMems = mBkMeans5cl.labels_

clImg = tcFlat[:,0]
#clImg=np.ndarray([rows,cols],dtype=int)
#clImg[:,:] = -9999
clImg[tc_valid] = clMems
cl_nan = np.argwhere(np.isnan(clImg))
clImg[cl_nan]=-9999
clImg = clImg.reshape(rows,cols,order='F')
clImg = clImg.astype('int16')
imgplot= plt.imshow(clImg)
imgplot.set_cmap('nipy_spectral')

with rasterio.open(iDir3 + 'TCTrends_clust_5cl'+'.tif', 'w', driver='GTiff', height=rows,
                       width=cols, count=1, dtype='int16',
                       crs=crsOut, transform=traOut, nodata=-9999) as dst:
    dst.write(clImg, 1)

"""


# from web, code to estimate number of clusters

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]

score

pl.plot(Nc,score)

pl.xlabel('Number of Clusters')

pl.ylabel('Score')

pl.title('Elbow Curve')

pl.show()
"""