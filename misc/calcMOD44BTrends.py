#!/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:32:42 2017
This file assess trends in the Tasseled Cap index from 2000 - 2017 created
    using the MODIS Land Surface Reflectance Data (MOD09A1) for
    Alaska. Scrit is run after using procTCTimeSeries.py
    

@author: jeth6160
"""


import numpy as np
import numpy.ma as ma
import pylab as pl
import scipy as sp
import matplotlib.pyplot as plt
import datetime # datetime commands
import re # python regular expressions
import os # python os tools
import fnmatch # function matching tools
import rasterio
import glob
import time
from scipy import stats 
from skimage.measure import block_reduce
from tqdm import tqdm 

# set input output directories
iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/MOD44B/'
oDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'

fModPre='MOD44B.006_'
fCanPre = 'Percent_Tree_Cover_doy*'
fVegPre = 'Percent_NonTree_Vegetation_doy*'
fNonVegPre = 'Percent_NonVegetated_doy*'
fPst = '.tif'

#print(fPre,fPst)
r = 13084
c = 17027 

years = range(2000,2018)

#fPre = fBrPre
#fPre = fGrPre
fPre = fCanPre

print('prefix is: ',fPre)
filesIn = glob.glob(iDir+fModPre+fPre+fPst)

print('files in contains: ',filesIn)


dataCube = []
crsOut =[]
print('Building data cubes...')

#for i in range(0,1):
for i in range(0,len(filesIn)):
    with rasterio.open(filesIn[i], nodata=-999,dtype='float64') as src:
        tIn = src.read().squeeze()
        crsOut = src.crs
        traOut = src.transform
    tIn = tIn.astype('float64') 
    tIn_i = np.where((tIn == -999) | (tIn >100))    
    tIn[tIn_i] = np.nan  
    tInAgg = block_reduce(tIn,block_size=(2,2),func=np.nanmean)
    dataCube.append(tInAgg)
    
    
#block_reduce(jtmp,block_size=(2,2),func=np.mean)    
#dataCube = [junk.squeeze() for junk in dataCube]    

dataCube= np.array(dataCube)
dataCube = np.rollaxis(dataCube,0,3)  

#dataMasked = ma.masked_values(dataCube,-54743.4496)
#data_i = dataCube[:,:,0]!=-54743.4496
#data_i = dataCube[:,:,0]!=25873.612799999995
#data_i = dataCube[~np.isnan(dataCube[:,:,0])
data_i = ~np.isnan(dataCube[:,:,0])
#np.savez(str(oDir+'dataCube.npz'),dataCube)
validData = dataCube[data_i,:]

tsSlope = np.zeros([100,1])
tsInt = np.zeros([100,1])
tsLo = np.zeros([100,1])
tsUp = np.zeros([100,1])

[sTime, sClock] = time.time(),time.clock()
# the results will have following format:
#   Theil-Sen slope: Col 1
#   Theil-Sen intercept: Col 2
#   Lower confidence interval: col 3
#   upper confidence interval: col 4
print('Processign stats...')
tsRegResults = np.apply_along_axis(stats.mstats.theilslopes,1,validData)    
#tsSlope, tsInt, tsLo,tsUp = np.apply_along_axis(stats.mstats.theilslopes,1,validData[0:99,:])
#tsRegResults = np.apply_along_axis(stats.mstats.theilslopes,1,validData[0:999999,:])

tsSlopeImg = np.full(tInAgg.shape, -999.)
#tsSlopeImg = np.full(tInAgg.shape, np.nan)
#tsIntImg = np.full((r,c), -9999.)
#tsLoImg = np.full((r,c), -9999.)
#tsUpImg = np.full((r,c), -9999.)

# slope is column 0 of the results; not 1; previous version used 1 :(
tsSlopeImg[data_i] = tsRegResults[:,0]

[eTime, eClock] = time.time(),time.clock()  
print('System time for numpy version is: ',eTime-sTime)
print('Clock time for numpy version is: ',eClock-sClock)

#with rasterio.open(oDir + 'TC_BrightTrends'+'.tif', 'w', driver='GTiff', height=tsSlopeImg.shape[0],
with rasterio.open(oDir + 'Pct_TreeCover_Trends'+'.tif', 'w', driver='GTiff', height=tsSlopeImg.shape[0],
                   width=tsSlopeImg.shape[1], count=1, dtype='float64',
                   crs=crsOut, transform=traOut,nodata=-999) as dst:
    dst.write(tsSlopeImg, 1)


#[sTime, sClock] = time.time(),time.clock()
#[tsSlope, tsInt, tsLo,tsUp] = np.apply_along_axis(stats.mstats.theilslopes,1,validData)    
#tsSlope, tsInt, tsLo,tsUp = stats.mstats.theilslopes(validData[0,:])

#[eTime, eClock] = time.time(),time.clock()  
#print('System time for numpy version is: ',eTime-sTime)
#print('Clock time for numpy version is: ',eClock-sClock)
