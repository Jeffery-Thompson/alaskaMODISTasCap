#!/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Spyder Editor

This file processes MODIS Land Surface Reflectance Data (MOD09A1) for
    Alaska. Data were obtained from NASA's LPDAAC using their AppEARS tool
    
    Tasseled Cap derived using loadings presented in Lobser & Cohen (2007) that
        were oriiginally developed using Nadir BRDF-Adjusted Reflectance 
        (NBAR, MOD43) data. Assumption is the difference between MOD43 and 
        MOD09 will be minimal.
    
    Lobser & Cohen (2007) MODIS tasselled cap: land cover characteristics 
        expressed through transformed MODIS data, International Journal of 
        Remote Sensing, 28:22, 5079-5101, DOI: 10.1080/01431160701253303
    
    Tassled Cap loadings for bands 1-7
                Bright. Green. Wet.
        b1: Red 0.4395 -0.4064 0.1147
        b2: NIR1 0.5945 0.5129 0.2489
        b3: Blue 0.2460 -0.2744 0.2408
        b4: Green 0.3918 -0.2893 0.3132
        b5: NIR2 0.3506 0.4882 -0.3122
        b6: SWIR1 0.2136 -0.0036 -0.6416
        b7: SWIR2 0.2678 -0.4169 -0.5087
        
"""

# importing the required python libraries 
import numpy as np
#import pylab as pl
#import scipy as sp
import matplotlib.pyplot as plt
import datetime # datetime commands
#import re # python regular expressions
#import os # python os tools
# fnmatch # function matching tools
import rasterio
import glob
import time

# set input output directories
iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/MOD09A1/'
oDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
#print(iDir,oDir)

# set file pre/post-fixes
#   prefix is standard MODIS file prefix
fPre = 'MOD09A1.005_sur_refl_'
fPst = '*.tif'
#print(fPre,fPst)

# set file band prefixes
#   MODIS data are split into individual files, one for each band
b1p = 'b01_doy'
b2p = 'b02_doy'
b3p = 'b03_doy'
b4p = 'b04_doy'
b5p = 'b05_doy'
b6p = 'b06_doy'
b7p = 'b07_doy'

# scale factor - for scaling MOD10A1 data; in MOD10A1, is 0.0001 to scale the
#   int to floats. here, using the inverse to scale the loadings to ints - 
#   more efficient to store as ints for now
scale = 10000

# set mask value for input & output data
mask_in =-28672
mask_out = -28672

# set loading matrices for Tasseled Cap Calculation; values reported in 
#   Lobser & Cohen were for MOD43, which had a different scale factor.
#   loading from L&C are x 10
ldBright = np.array([0.4395, 0.5945, 0.2460, 0.3918, 0.3506, 0.2136, 0.2678]) *scale
ldGreen =np.array([-0.4064, 0.5129, -0.2744, -0.2893, 0.4882, -0.0036, -0.4169]) * scale
ldWet = np.array([0.1147, 0.2489, 0.2408, 0.3132, -0.3122, -0.6416, -0.5087]) * scale

years = range(2000,2010)
print(list(years))

for year in [years[0]]:
    print('processing year: ',year) 
    # list of yearly MOD09A1 files for each and
    b1Files = glob.glob(iDir+fPre+b1p+str(year)+fPst)
    b2Files = glob.glob(iDir+fPre+b2p+str(year)+fPst)
    b3Files = glob.glob(iDir+fPre+b3p+str(year)+fPst)
    b4Files = glob.glob(iDir+fPre+b4p+str(year)+fPst)
    b5Files = glob.glob(iDir+fPre+b5p+str(year)+fPst)
    b6Files = glob.glob(iDir+fPre+b6p+str(year)+fPst)
    b7Files = glob.glob(iDir+fPre+b7p+str(year)+fPst)
    
    #lists for holding yearly band image stacks
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    b5 = []
    b6 = []
    b7 = []
 
    print(year)
#    print(b1Files[0])
#    [sTime, sClock] = time.time(),time.clock()
    if len(b1Files) == len(b2Files) & len(b1Files) == len(b3Files) & \
        len(b1Files) == len(b4Files) & len(b1Files) == len(b5Files) & \
        len(b1Files) == len(b6Files) & len(b1Files) == len(b7Files):
            
            
            #print('in if loop')  
            # dminesion lenght of data cube
            #[sTime, sClock] = time.time(),time.clock()
            for i in range(0,len(b1Files)):
                #print('data cube create loop: ',i) 
                # append band arrays
                with rasterio.open(b1Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                    crsOut = src.crs
                    traOut = src.transform
                b1.append(tIn)
                
                # b2
                with rasterio.open(b2Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b2.append(tIn)
                
                # b3
                with rasterio.open(b3Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b3.append(tIn)
                
                # b4
                with rasterio.open(b4Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b4.append(tIn)
                
                # b5
                with rasterio.open(b5Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b5.append(tIn)
                
                # b6
                with rasterio.open(b6Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b6.append(tIn)
                
                # b7
                with rasterio.open(b7Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b7.append(tIn)
            
            # timing for numpy arrays
            #[eTime, eClock] = time.time(),time.clock()  
            #print('System time for numpy version is: ',eTime-sTime)
            #print('Clock time for numpy version is: ',eClock-sClock) 
            
    else:
        break
         #sys.exit([10])              
    
    #[sTime, sClock] = time.time(),time.clock()              
    b1med = np.median(b1,axis=0)
    b2med = np.median(b2,axis=0)
    b3med = np.median(b3,axis=0)
    b4med = np.median(b4,axis=0)
    b5med = np.median(b5,axis=0)
    b6med = np.median(b6,axis=0)
    b7med = np.median(b7,axis=0)
#[eTime, eClock] = time.time(),time.clock() 
#print('System time for median on numpy array is: ',eTime-sTime)
#print('Clock time for median on numpy array is: ',eClock-sClock)

#plt.imshow(b1med.squeeze())
    
#    mask = (b1med < -100) & (b2med < -100) & (b3med < -100) & \
#        (b4med < -100) & (b5med < -100) & (b6med < -100) & \
#        (b7med < -100)

    mask = (b1med < -100) | (b2med < -100) | (b3med < -100) & \
        (b4med < -100) | (b5med < -100) | (b6med < -100) | \
        (b7med < -100)

    tcBright = b1med*ldBright[0] + b2med*ldBright[1] + b3med*ldBright[2] + \
        b4med*ldBright[3] + b5med*ldBright[4] + b6med*ldBright[5] + \
        b7med*ldBright[6]
        
    tcGreen = b1med*ldGreen[0] + b2med*ldGreen[1] + b3med*ldGreen[2] + \
        b4med*ldGreen[3] + b5med*ldGreen[4] + b6med*ldGreen[5] + \
        b7med*ldGreen[6]
    
    tcWet = b1med*ldWet[0] + b2med*ldWet[1] + b3med*ldWet[2] + \
        b4med*ldWet[3] + b5med*ldWet[4] + b6med*ldWet[5] + \
        b7med*ldWet[6]
    
    tcBright[mask] = -28672
    tcGreen[mask] = -28672
    tcWet[mask] = -28672

#    tcBright[mask] = mask_out
#    tcGreen[mask] = mask_out
#    tcWet[mask] = mask_out

    
#    tcBright = tcBright.astype('int16')
#    tcGreen = tcGreen.astype('int16')
#    tcWet = tcWet.astype('int16')
    
    with rasterio.open(oDir + 'TCBright_medMOD09A1_'+str(year) + '.tif', 'w', \
        driver='GTiff', height=tcBright.shape[1], width=tcBright.shape[2], \
        count=1, dtype='int16', crs=crsOut, transform=traOut, \
        nodata=mask_out) as dst:
            dst.write(tcBright.astype('int16').squeeze(), 1)
    
    with rasterio.open(oDir + 'TCGreen_medMOD09A1_'+str(year) + '.tif', 'w', \
        driver='GTiff', height=tcGreen.shape[1], width=tcGreen.shape[2], \
        count=1, dtype='int16', crs=crsOut, transform=traOut,\
        nodata=mask_out) as dst:
            dst.write(tcGreen.astype('int16').squeeze(), 1)
            
    with rasterio.open(oDir + 'TCWet_medMOD09A1_'+str(year) + '.tif', 'w', \
        driver='GTiff', height=tcWet.shape[1], width=tcWet.shape[2], \
        count=1, dtype='int16', crs=crsOut, transform=traOut, \
        nodata=mask_out) as dst:
            dst.write(tcWet.astype('int16').squeeze(), 1)
    
