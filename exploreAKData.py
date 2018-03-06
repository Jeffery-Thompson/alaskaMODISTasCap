#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:34:10 2018

@author: jeth6160
"""

import numpy as np
import rasterio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from affine import Affine 
import numpy.ma as ma

#import rasterio.transform as transform

# set input output directories
iDir = '/Users/jeth6160/Desktop/permafrost/Arctic/WHRC/output/'
iDir2 = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'
iDir3 = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'

courseFile = 'AK_PctBurned_v2.tif'
with rasterio.open(iDir +  courseFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    cIn = src.read()
    cMask = src.read_masks(1)
    pctBurned = cIn.squeeze()
    rB,cB = np.where(src.read_masks(1) )
    #rB,cB = np.where(src.read_masks(1) > -9999)
    cTra = src.transform
    cAff = src.affine

cXYCent = cAff * Affine.translation(0.5,0.5)    
rc2ll_course = lambda r, c: (c, r) * cXYCent
ll2rc_course = lambda y, x: (x,y) * ~cXYCent

bLon, bLat = np.vectorize(rc2ll_course, otypes=[np.float, np.float])(rB,cB)   

burnFlat = pctBurned[rB,cB]
 
courseFile = 'AK_dLST.tif'
with rasterio.open(iDir +  courseFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    cIn = src.read()
    cMask = src.read_masks(1)
    deltaLST = cIn.squeeze()
    rT,cT = np.where(src.read_masks(1) )
    #rT,cT = np.where(src.read_masks(1) > -9999)
    cTra = src.transform
    cAff = src.affine    

tLon, tLat = np.vectorize(rc2ll_course, otypes=[np.float, np.float])(rT,cT) 
    
lstFlat = deltaLST[rT,cT]

courseFile = 'TC_TrendsAll_v3.tif'
with rasterio.open(iDir3 +  courseFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    cIn = src.read()
    cMask = src.read_masks(1)
    tassCap = cIn.squeeze()
    rTC,cTC = np.where(src.read_masks(1) )
    #rT,cT = np.where(src.read_masks(1) > -9999)
    cTra = src.transform
    cAff = src.affine    

tcLon, tcLat = np.vectorize(rc2ll_course, otypes=[np.float, np.float])(rTC,cTC) 
    
tcBrightFlat = tassCap[0,rTC,cTC]
tcGreenFlat = tassCap[1,rTC,cTC]
tcWetFlat = tassCap[2,rTC,cTC]

fineFile = 'AK_PZI.tif'
with rasterio.open(iDir2 +  fineFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    fIn = src.read()
    fMask = src.read_masks(1)
    pzi = fIn.squeeze()
    rP,cP = np.where(src.read_masks(1) )
    #rT,cT = np.where(src.read_masks(1) > -9999)
    fTra = src.transform
    fAff = src.affine    

fXYCent = fAff * Affine.translation(0.5,0.5)
rc2ll_fine = lambda r, c: (c, r) * fXYCent
ll2rc_fine = lambda y, x: (x,y) * ~fXYCent

#pLon, pLat = np.vectorize(rc2ll_fine, otypes=[np.float, np.float])(rP,cP)
#
b_in_p_Col,b_in_p_Row = np.vectorize(ll2rc_fine, otypes=[np.int, np.int])(bLat, bLon) 
t_in_p_Col,t_in_p_Row = np.vectorize(ll2rc_fine, otypes=[np.int, np.int])(tLat, tLon) 
tc_in_p_Col,tc_in_p_Row = np.vectorize(ll2rc_fine, otypes=[np.int, np.int])(tcLat, tcLon) 

burnPZIFlat = pzi[b_in_p_Row,b_in_p_Col]
lstPZIFlat = pzi[t_in_p_Row,t_in_p_Col]
tcPZIFlat = pzi[tc_in_p_Row,tc_in_p_Col]
    
st = np.arange(0.0,1.0,0.01)
ed = np.arange(0.01,1.01,.01)


#x=np.array([])
#meanPZI = []
x = np.zeros([99,1])
meanBurn = np.empty([99,1])
meanBurn[:] = np.nan



for i in range(len(ed)):
    x[i] = ed[i]
    tInd = np.where((burnPZIFlat > st[i]) & (burnPZIFlat < ed[i]))
    
    if tInd[0].size != 0:
        meanBurn[i] = burnFlat[tInd].mean()
        
        
x1 = np.zeros([99,1])
meanLST= np.empty([99,1])
meanLST[:] = np.nan        
    
for i in range(len(ed)):
   x1[i] = ed[i]
   tInd = np.where((lstPZIFlat > st[i]) & (lstPZIFlat < ed[i]))
    
   if tInd[0].size != 0:
       meanLST[i] = lstFlat[tInd].mean()   


def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N, mode='valid')
    #return np.convolve(x, np.ones((N,))/N,mode='valid')[(N-1):]

flatMeanLST = meanLST.flatten()
runMeanLST = runningMeanFast(flatMeanLST,3) 


x2 = np.zeros([99,1])
meanBright = np.empty([99,1])
meanBright[:] = np.nan
meanGreen = np.empty([99,1])
meanGreen[:] = np.nan
meanWet = np.empty([99,1])
meanWet[:] = np.nan

for i in range(len(ed)):
   x2[i] = ed[i]
   tInd = np.where((lstPZIFlat > st[i]) & (lstPZIFlat < ed[i]))
    
   if tInd[0].size != 0:
       meanBright[i] = tcBrightFlat[tInd].mean()
       meanGreen[i] = tcGreenFlat[tInd].mean()
       meanWet[i] = tcWetFlat[tInd].mean()


   
    """
    ind = np.where((pzi[b_in_p_Col,b_in_p_Row] > st[i]) &\
                   (pzi[b_in_p_Col,b_in_p_Row] < ed[i]))
    if ind.size == 0:
        meanPZI[i] = NaN
    else:
        pzi[ind].mean()
            
    
    
    
"""    