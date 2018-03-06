#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:10:05 2018
This file is used to extract spatial data from netcdfs for alaska
    NetCDF ata are those from :
        Rogers et al. (2015) Influence of tree species on continental 
        differences in boreal fires and climate feedbacks, Nature Geoscience 
        8, 228–234, doi:10.1038/ngeo2352
    
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
import netCDF4 
import fiona
from osgeo import gdal
from netCDF4 import Dataset
from pyproj import Proj, transform
from rasterio.tools.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import crs

ncEPSG = Proj(init='epsg:4326')
 
# set input output directories
iDir = '/Users/jeth6160/Desktop/permafrost/Arctic/WHRC/'
oDir = '/Users/jeth6160/Desktop/permafrost/Arctic/WHRC/output/'
#iDir = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/'
#oDir = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'


iDir2 = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
iDir3 = '/Users/jeth6160/Desktop/permafrost/Alaska/GINA/AKLCC_boundaries/'


#Crs = Proj('+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=0 +y_0=0')
Crs = crs.CRS({'proj':'aea',
       'lat_1':55,
       'lat_2':65,
       'lat_0':50,
       'lon_0':-154,
       'x_0':0,
       'y_0':0})
#fBrPre = '*_dataCube'
#fGrPre = '*_TCGreen'
#fWtPre = '*_TCWet'
fPst = '.nc'
#print(fPre,fPst)
r = 6542
c = 8514 

#with fiona.open(iDir3+'AK_LCC_GCS_WGS84_b20k.shp') as shapefile:
with fiona.open(iDir3+'AK_LCC_GCS_WGS84_edit.shp') as shapefile:
    shpGeom = [feature["geometry"] for feature in shapefile]
    shpCrs = shapefile.crs
    shpBounds=shapefile.bounds
 
shpBounds=rasterio.coords.BoundingBox(left=shpBounds[0],\
                                      bottom=shpBounds[1],\
                                      right=shpBounds[2],\
                                      top=shpBounds[3])
shpCut = [()]
    
#with fiona.open(iDir3+'AK_LCC_GCS_WGS84_b20k.shp') as shapefile:
#    shpExt = shapefile.extent    

#with rasterio.open(iDir + 'FRP_2003_2013_QD.nc','r', driver='NetCDF') as src:
#with rasterio.open(iDir + 'dNBR_2001_2012_QD.nc','r', driver='NetCDF') as src:
#with rasterio.open(iDir + 'PctBurned_2001_2012_QD.nc','r', driver='NetCDF') as src:
#with rasterio.open(iDir + 'VegDestructionIndex_2003_2012_QD.nc','r', driver='NetCDF') as src:
with rasterio.open(iDir + 'dLST_2003_2012_QD.nc','r', driver='NetCDF') as src: 
#with rasterio.open(iDir + 'PctBorealPixel_QD.nc','r', driver='NetCDF') as src:
#with rasterio.open(iDir + 'PZI.flt','r', driver='EHdr') as src:
    profile = src.profile 
    test_prof=src.profile
    test_img, test_trans = mask(src,shpGeom, crop=True)
    #test_img, test_trans = mask(src,shpBounds, crop=True)
    #test_img = src
    test_crs = src.crs
    test_meta = src.meta.copy()
    test_bound=src.bounds
    test_affine=src.affine
    
    test_crs=crs.CRS.from_epsg(4326)
    
    # Calculate the ideal dimensions and transformation in the new crs
    dst_affine, dst_width, dst_height = calculate_default_transform(
            test_crs, Crs, test_img.shape[2], test_img.shape[1], *shpBounds)
    
    
    # update the relevant parts of the profile
    profile.update({
            'crs': Crs,
            'transform': dst_affine,
            'affine': dst_affine,
            'width': dst_width,
            'height': dst_height,
            'driver':'GTiff'
    })
    
    # Reproject and write each band
    
    src_array = test_img
    dst_array = np.empty((dst_height, dst_width), dtype='float32')
    #dst_array = np.empty(( dst_height,dst_width), dtype='float32')
    
    reproject(
            # Source parameters
            source=src_array,
            src_crs=test_crs,
            src_transform=test_trans,
            # Destination paramaters
            destination=dst_array,
            dst_transform=dst_affine,
            dst_crs=Crs,
            # Configuration
            resampling=Resampling.nearest,
            num_threads=2)
    
    dst_array=np.expand_dims(dst_array,0)

    #with rasterio.open(oDir+'AK_PctBorealPixel_v2.tif', 'w', **profile) as dst:
    with rasterio.open(oDir+'AK_dLST.tif', 'w', **profile) as dst:
        dst.write(dst_array)
    
    
"""  
with rasterio.open(oDir+'AK_meanFRP.tif','w',driver='GTiff',\
    height=test_img.shape[1],width=test_img.shape[2],) as dst:
        
    
    
with rasterio.open(oDir + 'TCBright_medMOD09A1_'+str(year) + '.tif', 'w', \
                   driver='GTiff', height=tcBright.shape[1], \
                   width=tcBright.shape[2], count=1, dtype='float32', \
                   crs=Crs) as dst:
    rasterio.reproject(
            source=test_img.squeeze(),
            destination=rasterio.ba)
            
            dst.write(tcBright.astype('int16').squeeze(), 1)    
    
   
# gdal - fail :(    
#gdal.Warp(oDir+'AK_meanFRP.tif',test_img,dstSRS=Crs,srcSRS=test_crs) 
    
    
    
    
fIn = Dataset(iDir+'FRP_2003_2013_QD.nc', "r", format="NETCDF4")
#fIn = netCDF4.MFDataset(iDir+'FRP_2003_2013_QD.nc')
lat = fIn.variables['lat'][:]
lon = fIn.variables['lon'][:]
meanFRP = fIn.variables['mean_FRP'][:]

#with rasterio.open(oDir + 'TC_BrightTrends'+'.tif', 'w', driver='GTiff', height=tsSlopeImg.shape[0],
with rasterio.open(iDir + 'FRP_2003_2013_QD.nc', 'r', driver='netCDF4') as src:
    ncIn=src.read()
    ncCrs = src.crs
    ncTra = src.transform
    ncBound = src.bounds
    
    
with rasterio.open(iDir2 + 'TC_TrendsAll_v3'+'.tif','r', driver='GTiff') as src:
    tIn = src.read()
    crsOut = src.crs
    traOut = src.transform
    boundsOut = src.bounds
    
ul_x_ae, ul_y_ae = boundsOut[0],  boundsOut[3]
ur_x_ae, ur_y_ae = boundsOut[2],  boundsOut[3]
ll_x_ae, ll_y_ae = boundsOut[0],  boundsOut[1]
lr_x_ae, lr_y_ae = boundsOut[2],  boundsOut[1]

ul_x_ae = -175.0

ul_x_ll, ul_y_ll = transform(Crs,ncEPSG,ul_x_ae, ul_y_ae) 
ur_x_ll, ur_y_ll = transform(Crs,ncEPSG,ur_x_ae, ur_y_ae)
ll_x_ll, ll_y_ll = transform(Crs,ncEPSG,ll_x_ae, ll_y_ae)
lr_x_ll, lr_y_ll = transform(Crs,ncEPSG,lr_x_ae, lr_y_ae)

#y_ll_min = np.min(ul_y_ll,ur_y_ll,ll_y_ll,lr_y_ll)

latli = np.argmin(np.abs(lat - np.max([ul_y_ll,ur_y_ll,ll_y_ll,lr_y_ll] )))
latui = np.argmin(np.abs(lat - np.min([ul_y_ll,ur_y_ll,ll_y_ll,lr_y_ll] ))) 

# longitude lower and upper index
lonli = np.argmin( np.abs( lon - np.min([ul_x_ll,ur_x_ll,ll_x_ll,lr_x_ll] )))
lonui = np.argmin( np.abs( lon - np.max([ul_x_ll,ur_x_ll,ll_x_ll,lr_x_ll] )))

alaskaSubset = fIn.variables['mean_FRP'][ latli:latui , lonli:lonui ] 
alaskaSubset = meanFRP[latli:latui , lonli:lonui]
"""