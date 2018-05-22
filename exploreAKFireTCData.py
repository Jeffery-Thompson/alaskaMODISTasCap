#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:11:54 2018

@author: jeth6160
"""

import numpy as np
import rasterio
import matplotlib
import fiona
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from affine import Affine 
import numpy.ma as ma

#import rasterio.transform as transform

# set input output directories
#iDir = '/Users/jeth6160/Desktop/permafrost/Arctic/WHRC/output/'
iShpDir = '/Users/jeth6160/Desktop/permafrost/Alaska/BLM/FirePerimiters1940/'
iPziDir = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'
iRoiDir = '/Users/jeth6160/Desktop/permafrost/Alaska/USGS/'
oDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'

imgExt = '.tif'
shpExt ='.shp'

clusFile = 'AK_IntMon_Clu7' + imgExt
tcFile = 'TC_TrendsAll_v3' + imgExt
fireFile = 'FireAreaHistory' + shpExt
roiFile = 'akecoregions_intmontane' + shpExt

with rasterio.open(oDir +  clusFile, 'r', driver='GTiff', nodatavals = -9999.) as src:
    clusIn = src.read()
    clusMask = src.read_masks(1)
    tcClus = clusIn.squeeze()
    rCl,cCl = np.where(src.read_masks(1) )
    #rT,cT = np.where(src.read_masks(1) > -9999)
    clusTra = src.transform
    clusAff = src.affine    


with rasterio.open(oDir + tcFile,'r', driver='GTiff') as src:
    tcIn = src.read()
    tcCRS = src.crs
    tcTra = src.transform
    tcBounds = src.bounds

roi = gpd.read_file(iRoiDir+roiFile)



burnArea = gpd.read_file(iShpDir+fireFile)
burnArea.FireYear = pd.to_numeric(burnArea.FireYear)
firePerms1940s = burnArea.loc[burnArea['FireYear'] <= 1950] & burnArea['FireYear']< 1950]
    
firePerms1940 = burnArea.loc[(burnArea['FireYear'] >= 1940) & (burnArea['FireYear'] < 1950)]    

InnerMontain = burnArea.cx[roi[geometry]]
fireHist = 
