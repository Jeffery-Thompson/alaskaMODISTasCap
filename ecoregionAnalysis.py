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

iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/EPA/'
fIn = 'AK_export_albers.shp'

iDir2 = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
fIn2 = 'TC_TrendsAll_v3.tif'

shpIn = gpd.read_file(iDir+fIn)

# FID for northern most shapefils 6,11,14
shape = shpIn[6]

x,y = shape['geometry'].exterior.coords.xy
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
"""
imgInPoly = shape_path.contains_points(imgPts,transform=None,radius=0.0)

imgTemp = np.zeros([cIn.shape[1],cIn.shape[2]],dtype='int8')
imgInPoly_i = np.where(imgInPoly>0)
imgTemp[imgInPoly_i]=1

imgTemp2 = imgTemp.reshape(iR,iC,order='F')

ptsContained = shape_path.contains_points(imgPoints)


p = path.Path([(0,0), (0, 1), (1, 1), (1, 0)])
points = np.array([.5, .5]).reshape(1, 2)
p.contains_points(points)
"""