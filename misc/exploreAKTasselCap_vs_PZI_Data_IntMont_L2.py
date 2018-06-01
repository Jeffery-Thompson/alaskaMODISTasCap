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

"""
courseFile = 'AK_PctBurned_v2.tif'
with rasterio.open(iDir +  courseFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    cIn = src.read()
    cMask = src.read_masks(1)
    pctBurned = cIn.squeeze()
    rB,cB = np.where(src.read_masks(1) )
    #rB,cB = np.where(src.read_masks(1) > -9999)
    cTra = src.transform
    cAff = src.affine

#cXYCent = cAff * Affine.translation(0.5,0.5)    
#rc2ll_course = lambda r, c: (c, r) * cXYCent
#ll2rc_course = lambda y, x: (x,y) * ~cXYCent

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
"""

clusFile = 'AK_L2_Cluster7.tif'
with rasterio.open(iDir3 +  courseFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    clIn = src.read()
    clMask = src.read_masks(1)
    clusters = cIn.squeeze()
    rCL,cCL = np.where(src.read_masks(1) )
    #rT,cT = np.where(src.read_masks(1) > -9999)
    clTra = src.transform
    clAff = src.affine    

#courseFile = 'TC_TrendsAll_v3.tif'
courseFile = 'InterMont_L2_TCTrends_v3.tif'
with rasterio.open(iDir3 +  courseFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    cIn = src.read()
    cMask = src.read_masks(1)
    tassCap = cIn.squeeze()
    rTC,cTC = np.where(src.read_masks(1) )
    #rT,cT = np.where(src.read_masks(1) > -9999)
    cTra = src.transform
    cAff = src.affine    

fXYCent = cAff * Affine.translation(0.5,0.5)
rc2ll_course = lambda r, c: (c, r) * fXYCent
ll2rc_course = lambda y, x: (x,y) * ~fXYCent

tcLon, tcLat = np.vectorize(rc2ll_course, otypes=[np.float, np.float])(rTC,cTC) 
    
tcBrightFlat = tassCap[0,rTC,cTC]
tcGreenFlat = tassCap[1,rTC,cTC]
tcWetFlat = tassCap[2,rTC,cTC]

#fineFile = 'AK_PZI.tif'
fineFile = 'AK_IntMont_PZI.tif'
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
pziLon, pziLat = np.vectorize(rc2ll_fine, otypes=[np.float, np.float])(rP,cP)

#
#b_in_p_Col,b_in_p_Row = np.vectorize(ll2rc_fine, otypes=[np.int, np.int])(bLat, bLon) 
#t_in_p_Col,t_in_p_Row = np.vectorize(ll2rc_fine, otypes=[np.int, np.int])(tLat, tLon) 
tc_in_p_Col,tc_in_p_Row = np.vectorize(ll2rc_fine, otypes=[np.int, np.int])(tcLat, tcLon) 

pzi_in_tc_Col,pzi_in_tci_Row = np.vectorize(ll2rc_course, otypes=[np.int, np.int])(pziLat, pziLon) 

#burnPZIFlat = pzi[b_in_p_Row,b_in_p_Col]
#lstPZIFlat = pzi[t_in_p_Row,t_in_p_Col]
tcPZIFlat = pzi[tc_in_p_Row,tc_in_p_Col]
    
pziBrightFlat = tassCap[0,pzi_in_tc_Col,pzi_in_tci_Row]
pziGreenFlat = tassCap[1,pzi_in_tc_Col,pzi_in_tci_Row]
pziWetFlat = tassCap[2,pzi_in_tc_Col,pzi_in_tci_Row]

pziFlat=pzi[rP,cP]

st = np.arange(0.0,1.0,0.01)
ed = np.arange(0.01,1.01,.01)


#x=np.array([])
#meanPZI = []
x = np.zeros([99,1])
meanBright = np.empty([99,1])
meanBright[:] = np.nan
meanGreen = np.empty([99,1])
meanGreen[:] = np.nan
meanWet = np.empty([99,1])
meanWet[:] = np.nan

stdBright = meanBright.copy()
stdGreen = meanGreen.copy()
stdWet = meanWet.copy()

plt.figure()
plt.scatter(pziLon[tInd],pziLat[tInd],marker='.',color='black')
plt.show()

plt.figure()
plt.scatter(pziGreenFlat[tInd],pziWetFlat[tInd],marker='.',color='blue')
plt.show()

plt.figure()
plt.hist(pziGreenFlat[tInd],bins=100)
plt.show()

for i in range(len(ed)):
    x[i] = ed[i]
    tInd = np.where((pziFlat > st[i]) & (pziFlat < ed[i]) & (pziBrightFlat > -9999) & (pziGreenFlat > -9999) & (pziWetFlat > -9999) )
    #tInd_g = np.where((pziFlat > st[i]) & (pziFlat < ed[i]) & (pziGreenFlat > -9999))
    #tInd_w = np.where((pziFlat > st[i]) & (pziFlat < ed[i]) & (pziWetFlat > -9999))
     
    if tInd[0].size != 0:
        meanBright[i] = pziBrightFlat[tInd].mean()
        stdBright[i] = pziBrightFlat[tInd].std()
        
        meanGreen[i] = pziGreenFlat[tInd].mean()
        stdGreen[i] = pziGreenFlat[tInd].std()
        
        meanWet[i] = pziWetFlat[tInd].mean()
        stdWet[i] = pziWetFlat[tInd].std()
        
        
rangeBright=np.array([np.nanmax((meanBright+stdBright).squeeze()),np.nanmin((meanBright-stdBright).squeeze())])
rangeGreen=np.array([np.nanmax((meanGreen+stdGreen).squeeze()),np.nanmin((meanGreen-stdGreen).squeeze())])
rangeWet=np.array([np.nanmax((meanWet+stdWet).squeeze()),np.nanmin((meanWet-stdWet).squeeze())])        
        
grey=[.8,.8,.8]
f, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)
#f, (ax1) = plt.subplots(1,1,sharex=True)
#ax1.scatter(x,meanBright)
ax1.fill_between(x.squeeze(),(meanBright+stdBright).squeeze(),(meanBright-stdBright).squeeze(),color=grey)
ax1.plot(x,meanBright,'b-')
#ax1.xaxis.set_label('PZI')
ax1.tick_params(which='both',right = 'on',left = 'on', bottom='on', top='on',labelleft = 'on',labelbottom='off') 
ax1.set_ylabel('Bright Trend')
#ax1.yaxis.set_label_position('left')
#ax1.y_label('Avg. TC Bright Trend')
ax1.set_title('Averaged Tassled Cap Trends by Permafrost Zone ')
ax1.text(0.01,rangeBright.max(),'no permafrost',style='italic',rotation=90,verticalalignment='top')
ax1.plot([0.05,0.05],[rangeBright.max(),rangeBright.min(),],'k--')
ax1.text(0.06,rangeBright.max(),'isolated',style='italic',rotation=90,verticalalignment='top')
ax1.plot([0.1,0.1],[rangeBright.max(),rangeBright.min()],'k--')
ax1.text(0.11,rangeBright.max(),'sporadic',style='italic',rotation=90,verticalalignment='top')
ax1.plot([0.5,0.5],[rangeBright.max(),rangeBright.min()],'k--')
ax1.text(0.51,rangeBright.max(),'discontinous',style='italic',rotation=90,verticalalignment='top')
ax1.plot([0.9,0.9],[rangeBright.max(),rangeBright.min()],'k--')
ax1.text(0.91,rangeBright.max(),'continous',style='italic',rotation=90,verticalalignment='top')


#plt.subplot(312)
ax2.fill_between(x.squeeze(),(meanGreen+stdGreen).squeeze(),(meanGreen-stdGreen).squeeze(),color=grey)
ax2.plot(x,meanGreen,'b-')
ax2.text(0.01,rangeGreen.max(),'no permafrost',style='italic',rotation=90,verticalalignment='top')
ax2.plot([0.05,0.05],[rangeGreen.max(),rangeGreen.min()],'k--')
ax2.text(0.06,rangeGreen.max(),'isolated',style='italic',rotation=90,verticalalignment='top')
ax2.plot([0.1,0.1],[rangeGreen.max(),rangeGreen.min()],'k--')
ax2.text(0.11,rangeGreen.max(),'sporadic',style='italic',rotation=90,verticalalignment='top')
ax2.plot([0.5,0.5],[rangeGreen.max(),rangeGreen.min()],'k--')
ax2.text(0.51,rangeGreen.max(),'discontinous',style='italic',rotation=90,verticalalignment='top')
ax2.plot([0.9,0.9],[rangeGreen.max(),rangeGreen.min()],'k--')
ax2.text(0.91,rangeGreen.max(),'continous',style='italic',rotation=90,verticalalignment='top')
#ax2.xaxis.set_label('Veg. Destruction')
ax2.set_ylabel('Green Trend')
ax2.tick_params(which='both',right = 'on',left = 'on', bottom='on', top='on',labelleft = 'on',labelbottom='off') 
#ax2.set_title('Average TC Green Trend')

#plt.subplot(313)
ax3.fill_between(x.squeeze(),(meanWet+stdWet).squeeze(),(meanWet-stdWet).squeeze(),color=grey)
ax3.plot(x,meanWet,'b-')
ax3.text(0.01,rangeWet.max(),'no permafrost',style='italic',rotation=90,verticalalignment='top')
ax3.plot([0.05,0.05],[rangeWet.max(),rangeWet.min()],'k--')
ax3.text(0.06,rangeWet.max(),'isolated',style='italic',rotation=90,verticalalignment='top')
ax3.plot([0.1,0.1],[rangeWet.max(),rangeWet.min()],'k--')
ax3.text(0.11,rangeWet.max(),'sporadic',style='italic',rotation=90,verticalalignment='top')
ax3.plot([0.5,0.5],[rangeWet.max(),rangeWet.min()],'k--')
ax3.text(0.51,rangeWet.max(),'discontinous',style='italic',rotation=90,verticalalignment='top')
ax3.plot([0.9,0.9],[rangeWet.max(),rangeWet.min()],'k--')
ax3.text(0.91,rangeWet.max(),'continous',style='italic',rotation=90,verticalalignment='top')
#ax3.xaxis.set_label('Veg. Destruction')
ax3.set_xlabel('PZI')
ax3.set_ylabel('Wet Trend')
ax3.tick_params(which='both',right = 'on',left = 'on', bottom='on', top='on',labelleft = 'on',labelbottom='on') 
#ax3.set_title('Average TC Wet Trend')



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
