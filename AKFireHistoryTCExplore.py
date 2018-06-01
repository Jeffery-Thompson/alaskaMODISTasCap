#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:06:54 2018

@author: jeth6160

Script to look at TC values for different fire dates in Alaska
    Visual inspection of the data suggests a clear pattern for the TC Trends
    Data and here the intent is to use the shapefiles to extract TC trends and 
    average it for each burn polygon

"""

import numpy as np
import rasterio
import fiona
import itertools
from matplotlib import pyplot as plt
from matplotlib import path
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
from functools import partial
from affine import Affine 
from rasterio.tools.mask import mask
from shapely.ops import unary_union, cascaded_union

iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/BLM/FirePerimiters1940/'
fIn = 'FireAreaHistory.shp'
fireIn = gpd.read_file(iDir+fIn)

# need to clean a bit; years not numeric
fireIn.FireYear = pd.to_numeric(fireIn['FireYear'])

iDir2 = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
fIn2 = 'TC_TrendsAll_v3.tif'

oDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'

##########
# import the fire shapefile
#    
#akHistFires = fireIn.set_index(['FireName','FireYear']) 
#fireIn  = fireIn.set_index(['FireYear'])

akHistFires = fireIn[['FireName','FireYear','Complex','AREA','LEN','geometry']]
#akHistFires = akHistFires.set_index(['FireYear'])

# FID for northern most shapefils 6,11,14
#roi = shpIn.iloc[6]
# this is the new level 2 ecoregion
#roi = shpIn.iloc[0]

########
# import tassel cap trends data and set up for being able to use the shapefile
#   to calc that statistics for the TC trends contained within each shape

with rasterio.open(iDir2 + fIn2, 'r', driver='GTiff') as src: 
    tcIn = src.read()
    #rastMasked,rM_tra = mask(src,geoms,crop=True)
    tcMask = src.read_masks(1)
    tassCap = tcIn.squeeze()
    tcR,tcC = np.where(src.read_masks(1) > 0)
    tcTra = src.transform
    tcAff = src.affine
    tcCrs = src.crs

tcXYCent = tcAff * Affine.translation(0.5,0.5)    
rc2ll_tcimg = lambda r, c: (c, r) * tcXYCent
ll2rc_tcimg = lambda y, x: (x,y) * ~tcXYCent

xImg, yImg = np.vectorize(rc2ll_tcimg, otypes=[np.float, np.float])(tcR,tcC) 
imgPts = np.concatenate((xImg,yImg),axis=0).reshape(xImg.shape[0],2,order='F')


#fireByYear = akHistFires.groupby('FireYear')

##########
# setup the partial process and function
#    
#


# TC data go through 2017 ; Fires through 2018
#years = range(akHistFires.index.min(),akHistFires.index.max())
years = range(akHistFires.FireYear.min(),akHistFires.FireYear.max())



# this original funciton fails - multipolygons are painful :(

def avg_TassCap_by_shape(funShp, funImgPts, funYear):
    """ Function that takes the fire shapefile, and iterates by year, calculates the area for each fire by year?"""
    print('Working on: ', funYear)
    yr_fires = funShp.loc[funShp.FireYear==funYear]
    
    # get num pixels, mean & std for each TC band 
    tc_avgs = np.zeros(shape=(yr_fires.shape[0],1+3*2))
    
    nFires = range(0,yr_fires.shape[0])
    
    # for each fire, do the calcs
    for i in nFires:
        
        firePoly = yr_fires.iloc[i]['geometry']
        
        if firePoly.type == 'MultiPolygon':
            polyList = list(firePoly)
            
            pValsBright = []  
            pValsGreen = []
            pValsWet = []
            
            for poly in polyList:
                x,y = poly.exterior.coords.xy
                x = np.array(x)
                y = np.array(y)
                verts = np.concatenate((x,y),axis=0).reshape(x.shape[0],2,order='F')
                fshape_path = path.Path(verts)
                fImgInPoly = fshape_path.contains_points(funImgPts,transform=None,radius=0.0)
                fImgInPoly_i = np.where(fImgInPoly>0)
                pValsBright.append(tassCap[0,tcR[fImgInPoly_i],tcC[fImgInPoly_i]])
                pValsGreen.append(tassCap[1,tcR[fImgInPoly_i],tcC[fImgInPoly_i]])
                pValsWet.append(tassCap[2,tcR[fImgInPoly_i],tcC[fImgInPoly_i]])
                
            if not pValsBright:
                tc_avgs[i,:] = np.nan
                
            else:
                tc_avgs[i,0] = len(pValsBright)
                tc_avgs[i,1] = np.nanmean(pValsBright[0])
                tc_avgs[i,2] = np.nanstd(pValsBright[0])
                tc_avgs[i,3] = np.nanmean(pValsGreen[0])
                tc_avgs[i,4] = np.nanstd(pValsGreen[0])
                tc_avgs[i,5] = np.nanmean(pValsWet[0])
                tc_avgs[i,6] = np.nanstd(pValsWet[0])
            
        else:
             x,y = yr_fires.iloc[i]['geometry'].exterior.coords.xy
             x = np.array(x)
             y = np.array(y)
             verts = np.concatenate((x,y),axis=0).reshape(x.shape[0],2,order='F')
             fshape_path = path.Path(verts)
             fImgInPoly = fshape_path.contains_points(funImgPts,transform=None,radius=0.0)
             fImgInPoly_i = np.where(fImgInPoly>0)
             
             if not fImgInPoly_i:
                 tc_avgs[i,:] = np.nan
                 
             else:
                 tc_avgs[i,0] = fImgInPoly_i[0].shape[0]
                 tc_avgs[i,1] = tassCap[0,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].mean()
                 tc_avgs[i,2] = tassCap[0,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].std()
                 tc_avgs[i,3] = tassCap[1,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].mean()
                 tc_avgs[i,4] = tassCap[1,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].std()
                 tc_avgs[i,5] = tassCap[2,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].mean()
                 tc_avgs[i,6] = tassCap[2,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].std()
    

    tc_avgs = pd.DataFrame(data=tc_avgs,index=yr_fires.index,columns=['numPixels','meanBrigtht','stdBright','meanGreen','stdGreen','meanWet','stdWet']).set_index(yr_fires.index)    
    print('Finished: ', funYear)   
    return (pd.concat([yr_fires.iloc[:,0:2],tc_avgs],axis=1,join='inner'))
           
    """
    if not tc_avgs:
        print("proceesing of tc_avgs failed: ", funYear)
        
    else:
        tc_avgs = pd.DataFrame(data=tc_avgs,index=yr_fires.index,columns=['numPixels','meanBrigtht','stdBright','meanGreen','stdGreen','meanWet','stdWet'],set_index=yr_fires.index)
    
        return (pd.concat([yr_fires.iloc[:,0:2],tc_avgs],axis=1,join='inner'))
    """
          
            
"""   
        # test for mupltipolygon ; causes 
        #t_un.type
        #Out[217]: 'MultiPolygon'
        # if is it, fingure out the number and iterate over
        #t_list = list(yr_fires.loc[287]['geometry'])
        #len(t_list)
        
        x,y = yr_fires.loc[i]['geometry'].exterior.coords.xy
        #x,y = roiGeo.loc['geometry'].exterior.coords.xy

        x = np.array(x)
        y = np.array(y)

        verts = np.concatenate((x,y),axis=0).reshape(x.shape[0],2,order='F')
        fshape_path = path.Path(verts)
        fImgInPoly = fshape_path.contains_points(funImgPts,transform=None,radius=0.0)
        fImgInPoly_i = np.where(fImgInPoly>0)
        
        # store number of pixels contained by each shape
        tc_avgs[i,0] = fImgInPoly_i[0].shape[0]
        
        # calc average and stdev for band within the shape
        tc_avgs[i,1] = tassCap[0,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].mean()
        tc_avgs[i,2] = tassCap[0,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].std()
        
        tc_avgs[i,3] = tassCap[1,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].mean()
        tc_avgs[i,4] = tassCap[1,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].std()
        
        tc_avgs[i,5] = tassCap[2,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].mean()
        tc_avgs[i,6] = tassCap[2,tcR[fImgInPoly_i],tcC[fImgInPoly_i]].std()
"""

    
#########
#
# setup the cpu pool
safety_factor = 2 # or 2 or 3 or 4
num_cpus = mp.cpu_count() - safety_factor
data_pool = mp.Pool(processes = num_cpus)
 
# pass the function and data to the pool
tc_avgs_part = partial(avg_TassCap_by_shape,akHistFires, imgPts)
data_list = data_pool.map(tc_avgs_part, years)

akTCbyFire = pd.concat(data_list)
akTCbyFire = pd.DataFrame(akTCbyFire)

#akTCFire_i = akTCbyFire.isnull().any(axis=1)
akTCFire_i =akTCbyFire.iloc[:,3].notnull()

akTCFireValid=akTCbyFire.dropna(axis=0)
#akTCFireYear =akTCbyFire.groupby('FireYear')

akTCFireValid.plot.scatter('FireYear','meanBrigtht',marker='.')
akTCFireValid.plot.scatter('FireYear','meanGreen',marker='.')
akTCFireValid.plot.scatter('FireYear','meanWet',marker='.')


f, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True)

ax1 = plt.scatter(akTCFireValid.FireYear,akTCFireValid.meanBrigtht,marker='.',s=5)
ax1.set_ylabel('Bright Trend')
ax1.tick_params(which='both',right = 'on',left = 'on', bottom='on', top='on',labelleft = 'on',labelbottom='off') 
ax1.get_yaxis().set_label_coords(-0.1,0.5)


ax2 =plt.scatter(akTCFireValid.FireYear,akTCFireValid.meanGreen,marker='.',s=5)
plt.figure()


 
plt.hist2d(akTCFireValid.FireYear,akTCFireValid.meanBrigtht,bins=[78,25])
plt.hist2d(akTCFireValid.FireYear,akTCFireValid.meanGreen,bins=[78,25])


## you may need these, I haven’t because I was calling this from if __name__ == ‘__main__’, due to silly windows.
#data_pool.join()
#data_pool.close()