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
from scipy import stats 
import matplotlib.pyplot as plt
import datetime # datetime commands
import re # python regular expressions
import os # python os tools
import fnmatch # function matching tools
import rasterio
import glob
import time

# set input output directories
iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
oDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'

fBrPre = '*_dataCube'
fGrPre = '*_TCGreen'
fWtPre = '*_TCWet'
fPst = '.tif'
#print(fPre,fPst)
r = 6542
c = 8514 

years = range(2000,2018)

#fPre = fBrPre
#fPre = fGrPre
fPre = fWtPre

print('prefix is: ',fPre)
filesIn = glob.glob(iDir+fPre+fPst)

print('files in contains: ',filesIn)


dataCube = []
crsOut =[]
print('Building data cubes...')

for i in range(0,len(filesIn)):
    with rasterio.open(filesIn[i]) as src:
        tIn = src.read()
        crsOut = src.crs
        traOut = src.transform
    dataCube.append(tIn)
    
dataCube = [junk.squeeze() for junk in dataCube]    
dataCube= np.array(dataCube)
dataCube = np.rollaxis(dataCube,0,3)  

#dataMasked = ma.masked_values(dataCube,-54743.4496)
#data_i = dataCube[:,:,0]!=-54743.4496
#data_i = dataCube[:,:,0]!=25873.612799999995
data_i = dataCube[:,:,0]!=22759.833599999998
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

tsSlopeImg = np.full((r,c), -9999.)
#tsIntImg = np.full((r,c), -9999.)
#tsLoImg = np.full((r,c), -9999.)
#tsUpImg = np.full((r,c), -9999.)

tsSlopeImg[data_i] = tsRegResults[:,1]

[eTime, eClock] = time.time(),time.clock()  
print('System time for numpy version is: ',eTime-sTime)
print('Clock time for numpy version is: ',eClock-sClock)

#with rasterio.open(oDir + 'TC_BrightTrends'+'.tif', 'w', driver='GTiff', height=tsSlopeImg.shape[0],
with rasterio.open(oDir + 'TC_GreenTrends'+'.tif', 'w', driver='GTiff', height=tsSlopeImg.shape[0],
                   width=tsSlopeImg.shape[1], count=1, dtype='float64',
                   crs=crsOut, transform=traOut,nodata=-9999) as dst:
    dst.write(tsSlopeImg, 1)


#[sTime, sClock] = time.time(),time.clock()
#[tsSlope, tsInt, tsLo,tsUp] = np.apply_along_axis(stats.mstats.theilslopes,1,validData)    
#tsSlope, tsInt, tsLo,tsUp = stats.mstats.theilslopes(validData[0,:])

#[eTime, eClock] = time.time(),time.clock()  
#print('System time for numpy version is: ',eTime-sTime)
#print('Clock time for numpy version is: ',eClock-sClock)