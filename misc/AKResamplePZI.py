#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:23:30 2018

@author: jeth6160
"""
import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from affine import Affine 
import numpy.ma as ma
from skimage.transform import rescale,resize

#from scipy.ndimage import zoom
#from scipy.interpolate import griddata
#from scipy.misc import imresize

iDir = '/Users/jeth6160/Desktop/permafrost/Arctic/WHRC/output/'
iDir2 = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'
iDir3 = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'

courseFile = 'AK_PZI.tif'
with rasterio.open(iDir3 +  courseFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    cIn = src.read()
    cMask = src.read_masks(1)
    pzi = cIn.squeeze()
    rP,cP = np.where(src.read_masks(1) )
    #sRTC,sCTC = np.where(test_img.read_mask(1))
    #rT,cT = np.where(src.read_masks(1) > -9999)
    cTra = src.transform
    cAff = src.affine 
    cCrs = src.crs


fineFile = 'TC_TrendsAll_v3.tif'
with rasterio.open(iDir2 +  fineFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    fIn = src.read()
    fMask = src.read_masks(1)
    tassCap = fIn.squeeze()
    rTC,cTC = np.where(src.read_masks(1) )
    #rT,cT = np.where(src.read_masks(1) > -9999)
    fTra = src.transform
    fAff = src.affine  
    fCrs = src.crs
    

#resamImg = imresize(pzi,tassCap.shape[1:],interp='nearest')    
#resamImg = griddata(pzi,tassCap.shape[1:],interp='nearest')
resamImg = resize(pzi,tassCap.shape[1:],order=0,preserve_range=True)
resamImg = resamImg.astype('float32')


with rasterio.open(iDir2 + 'AK_PZI_resamp'+'.tif', 'w', driver='GTiff', height=resamImg.shape[1],
                       width=resamImg.shape[0], count=1, dtype='float32',
                       crs=fCrs, transform=fTra, nodata=-9999.) as dst:
    dst.write(resamImg, 1)



with rasterio.open(iDir2 + 'AK_PZI_resample'+'.tif', 'w', driver='GTiff', height=fIn.shape[1],
                       width=fIn.shape[2], count=1, dtype='float64',
                       crs=cCrs, transform=cTra, nodata=0) as dst:
    dst.write(pziLowImg, 1)
