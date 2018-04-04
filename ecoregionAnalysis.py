#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 17:21:41 2018

@author: jeth6160
"""
import numpy as np
import rasterio
from matplotlib import path
import geopandas as gpd
from affine import Affine 
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from tqdm import tqdm 
from sklearn.cluster import MiniBatchKMeans

iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/EPA/'
fIn = 'AK_export_albers.shp'

iDir2 = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
fIn2 = 'TC_TrendsAll_v3.tif'

shpIn = gpd.read_file(iDir+fIn)

# FID for northern most shapefils 6,11,14
roi = shpIn.iloc[6]

x,y = roi['geometry'].exterior.coords.xy
x = np.array(x)
y = np.array(y)

verts = np.concatenate((x,y),axis=0).reshape(x.shape[0],2,order='F')
shape_path = path.Path(verts)


with rasterio.open(iDir2 + fIn2, 'r', driver='GTiff') as src: 
    cIn = src.read()
    cMask = src.read_masks(1)
    frp = cIn.squeeze()
    iR,iC = np.where(src.read_masks(1) > 0)
    cTra = src.transform
    cAff = src.affine

cXYCent = cAff * Affine.translation(0.5,0.5)    
rc2ll_course = lambda r, c: (c, r) * cXYCent
ll2rc_course = lambda y, x: (x,y) * ~cXYCent

xImg, yImg = np.vectorize(rc2ll_course, otypes=[np.float, np.float])(iR,iC) 
imgPts = np.concatenate((xImg,yImg),axis=0).reshape(xImg.shape[0],2,order='F')

imgInPoly = shape_path.contains_points(imgPts,transform=None,radius=0.0)

imgTemp = np.zeros([cIn.shape[1],cIn.shape[2]],dtype='int8')
imgInPoly_i = np.where(imgInPoly>0)
imgTemp[iR[imgInPoly_i],iC[imgInPoly_i]]=1

#imgTemp2 = imgTemp.reshape(iR,iC,order='F')

#ptsCount = imgInPoly.sum()

#tempArray = cIn[:,iR[imgInPoly_i],iC[imgInPoly_i]].transpose()


s_size = 0.2
nCl = range(1, 20)
inertias=[]
for i in tqdm(nCl):
    #print('MiniBatchKMeans with clusters = ',i)
    mBkMeans = MiniBatchKMeans(n_clusters=i,init='random',max_iter=20,
                           batch_size=np.round(cIn[:,iR[imgInPoly_i],iC[imgInPoly_i]].transpose().shape[0]*s_size).astype(int),verbose=False,
                           compute_labels=True,random_state=42)
    
    mBkMeans.fit(cIn[:,iR[imgInPoly_i],iC[imgInPoly_i]].transpose())
    #clScores = mBkMeans[i].fit(dataClst).score(dataClst)
    inertias.append(mBkMeans.inertia_)

plt.figure()
plt.plot(nCl,inertias)
plt.xticks(nCl)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')  

mBkMeans4cl = MiniBatchKMeans(n_clusters=4,init='random',max_iter=20,
                           batch_size=np.round(cIn[:,iR[imgInPoly_i],iC[imgInPoly_i]].transpose().shape[0]*s_size).astype(int),verbose=False,
                           compute_labels=True,random_state=42)
clusterFit= mBkMeans4cl.fit(cIn[:,iR[imgInPoly_i],iC[imgInPoly_i]].transpose())
clMems = mBkMeans4cl.labels_


imgTemp3 = imgTemp
imgTemp3[iR[imgInPoly_i],iC[imgInPoly_i]]=(clMems+1).transpose()
imgTemp4 = imgTemp3.reshape(iR,iC,order='F')

plt.figure()
plt.imshow(imgTemp3)

"""
ptsContained = shape_path.contains_points(imgPoints)


p = path.Path([(0,0), (0, 1), (1, 1), (1, 0)])
points = np.array([.5, .5]).reshape(1, 2)
p.contains_points(points)
"""