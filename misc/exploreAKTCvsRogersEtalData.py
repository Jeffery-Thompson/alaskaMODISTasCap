#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:10:30 2018

This file is used for exploratory data analysis between the MOID derived
    Tassled Cap data for Alaska and the Rogers et al Fire data. Rogers et al 
    data were originally from MODIS, but are at 1/4 degree


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
#iDir = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'
iDir2 = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
#iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
#iDir2 = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'
#iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
#iDir2 = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'


oDir = '/Users/jeth6160/Desktop/permafrost/Alaska/TasseledCapAnalysis/output/'

fineFile = 'TC_TrendsAll_v3.tif'
#fineFile = 'AK_PZI.tif'
#courseFile = 'AK_meanFRP.tif'
#courseFile = 'AK_PctBurned.tif'
#courseFile = 'AK_PctBorealPixel_v2.tif' 
courseFile = 'AK_VegDestructionIndex.tif'
#courseFile = 'AK_PZI.tif' 
 

# open TC trends rasters
with rasterio.open(iDir +  courseFile, 'r', driver='GTiff', nodatavals = -9999) as src:
    cIn = src.read()
    cMask = src.read_masks(1)
    frp = cIn.squeeze()
    iR,iC = np.where(src.read_masks(1) > 0)
    cTra = src.transform
    cAff = src.affine
    #x_lat,y_lon = rasterio.transform.xy('affine',iX,iY,offset='center')
    #xLat,yLon = src.xy(iX,iY,offset='center')
    #xLat,yLon=src.xy(21,90)


# find pixel locations for course resolution data
#   define lamda function for mapping row/columns to lon/lat   
#   can do reverse transform as well    
cXYCent = cAff * Affine.translation(0.5,0.5)    
rc2ll_course = lambda r, c: (c, r) * cXYCent
ll2rc_course = lambda y, x: (x,y) * ~cXYCent
    
cLon, cLat = np.vectorize(rc2ll_course, otypes=[np.float, np.float])(iR,iC)
#cLon, cLat = np.vectorize(rc2ll_course, otypes=[np.float, np.float])(iC, iR)
# tested reverse to see that it did as expected
cCol, cRow = np.vectorize(ll2rc_course, otypes=[np.int, np.int])(cLat, cLon)
"""
with rasterio.open(iDir2 +  fineFile, 'r', driver='GTiff') as src:
    fIn = src.read()
    bright = fIn[0,:,:]
    green = fIn[1,:,:]
    wet = fIn[2,:,:]
    fineBound=src.bounds
    fineCrs = src.crs
    fAff = src.affine
"""
with rasterio.open(iDir2 +  fineFile, 'r', driver='GTiff') as src:
    fIn = src.read()
    bright = fIn[0,:,:]
    green = fIn[1,:,:]
    wet = fIn[2,:,:]
    fineBound=src.bounds
    fineCrs = src.crs
    fAff = src.affine

fXYCent = fAff * Affine.translation(0.5,0.5)
rc2ll_fine = lambda r, c: (c, r) * fXYCent
ll2rc_fine = lambda y, x: (x,y) * ~fXYCent

fCol,fRow = np.vectorize(ll2rc_fine, otypes=[np.int, np.int])(cLat, cLon)

# masked arrays are the valid values, not values to ignore
#plotData = np.column_stack([frp[iR,iC],bright[fRow,fCol]])
plotData = np.column_stack([frp[iR,iC],wet[fRow,fCol]])
plotMask = (plotData == -9999)
plotData = ma.masked_array(plotData,mask=plotMask)

f, (ax1) = plt.subplots(1,1,sharex=True)
ax1.scatter(plotData[:,0],plotData[:,1])
#ax1.xaxis.set_label('Percent Burned')
ax1.xaxis.set_label('Percent Burned ')
ax1.yaxis.set_label('TC Wet Trend')
ax1.tick_params(which='both',right = 'on',left = 'on', bottom='on', top='on',labelleft = 'on',labelbottom='on') 
ax1.set_title('TC Wet Trend vs Pct Burn')

"""
f, (ax1,ax2,ax3) = plt.subplots(1,3,sharex=True)
f, (ax1) = plt.subplots(1,1,sharex=True)
ax1.scatter(frp[iR,iC],bright[fRow,fCol])
ax1.xaxis.set_label('Veg. Destruction')
ax1.yaxis.set_label('TC Trend')
ax3.tick_params(which='both',right = 'on',left = 'on', bottom='on', top='on',labelleft = 'on',labelbottom='on') 
ax1.set_title('TC Bright')


#plt.subplot(132)
ax2.scatter(frp[iR,iC],green[fRow,fCol])
ax2.xaxis.set_label('Veg. Destruction')
ax2.tick_params(which='both',right = 'on',left = 'on', bottom='on', top='on',labelleft = 'off',labelbottom='on') 
ax2.set_title('TC Green')

#plt.subplot(133)
ax3.scatter(frp[iR,iC],wet[fRow,fCol])
ax3.xaxis.set_label('Veg. Destruction')
ax3.tick_params(which='both',right = 'on',left = 'on', bottom='on', top='on',labelleft = 'off',labelbottom='on') 
ax3.set_title('TC Wet')
"""

# histogram plotting
"""
f, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
ax1.hist(frp[iR,iC],bins=1000)
ax2.hist(bright[fRow,fCol],bins=1000)
ax3.hist(green[fRow,fCol],bins=1000)
ax4.hist(wet[fRow,fCol],bins=1000)

"""

"""
#plt.xlabel('Fire Radiative Power')
#plt.ylabel('TC Brightness Trends')
plt.scatter(frp[cRow,cCol],bright[fRow,fCol])
plt.xlabel('Fire Radiative Power')
plt.ylabel('TC Brightness Trends')

plt.scatter(frp[cRow,cCol],green[fRow,fCol])
plt.xlabel('Fire Radiative Power')
plt.ylabel('TC Green Trends')


plt.scatter(frp[cRow,cCol],wet[fRow,fCol])
plt.xlabel('Fire Radiative Power')
plt.ylabel('TC Wet Trends')

plt.scatter(green[fRow,fCol],wet[fRow,fCol])
plt.xlabel('TC Green Trends')
plt.ylabel('TC Wet Trends')

plt.scatter(bright[fRow,fCol],wet[fRow,fCol])
plt.xlabel('TC Bright Trends')
plt.ylabel('TC Wet Trends')

plt.scatter(bright[fRow,fCol],green[fRow,fCol])
plt.xlabel('TC Bright Trends')
plt.ylabel('TC Green Trends')
"""
#plt.scatter(bright,green)
#plt.xlabel('TC Bright Trends')
#plt.ylabel('TC Green Trends')
