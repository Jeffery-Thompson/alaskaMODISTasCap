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

import numpy as np
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

# set input output directories
iDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/MOD09A1/'
oDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'
#print(iDir,oDir)

# set file pre/post-fixes
fPre = 'MOD09A1.005_sur_refl_'
fPst = '*.tif'
print(fPre,fPst)

# set file band prefixes
b1p = 'b01_doy'
b2p = 'b02_doy'
b3p = 'b03_doy'
b4p = 'b04_doy'
b5p = 'b05_doy'
b6p = 'b06_doy'
b7p = 'b07_doy'

# set loading matrices
ldBright = [0.4395, 0.5945, 0.2460, 0.3918, 0.3506, 0.2136, 0.2678]
ldGreen =[-0.4064, 0.5129, -0.2744, -0.2893, 0.4882, -0.0036, -0.4169]
ldWet = [0.1147, 0.2489, 0.2408, 0.3132, -0.3122, -0.6416, -0.5087]

# set dimensions of images
r = 6542
c = 8514 

years = range(2000,2010)
print(list(years))

#for year in years:
#b1= []
#b2= []
#b3= []
#b4= []
#b5= []
#b6= []
#b7= []

for year in [years[0]]:
    b1Files = glob.glob(iDir+fPre+b1p+str(year)+fPst)
    b2Files = glob.glob(iDir+fPre+b2p+str(year)+fPst)
    b3Files = glob.glob(iDir+fPre+b3p+str(year)+fPst)
    b4Files = glob.glob(iDir+fPre+b4p+str(year)+fPst)
    b5Files = glob.glob(iDir+fPre+b5p+str(year)+fPst)
    b6Files = glob.glob(iDir+fPre+b6p+str(year)+fPst)
    b7Files = glob.glob(iDir+fPre+b7p+str(year)+fPst)
    
    # declare empty lists for testing times
    b1l= []
    b2l= []
    b3l= []
    b4l= []
    b5l= []
    b6l= []
    b7l= []
 
     # declare 3D arrays
    b1 = np.zeros([len(b1Files),r,c],dtype='int16')
    b2 = np.zeros([len(b1Files),r,c],dtype='int16')
    b3 = np.zeros([len(b1Files),r,c],dtype='int16')
    b4 = np.zeros([len(b1Files),r,c],dtype='int16')
    b5 = np.zeros([len(b1Files),r,c],dtype='int16')
    b6 = np.zeros([len(b1Files),r,c],dtype='int16')
    b7 = np.zeros([len(b1Files),r,c],dtype='int16')   
    print(year)
#    print(b1Files[0])
#    [sTime, sClock] = time.time(),time.clock()
    if len(b1Files) == len(b2Files) & len(b1Files) == len(b3Files) & \
        len(b1Files) == len(b4Files) & len(b1Files) == len(b5Files) & \
        len(b1Files) == len(b6Files) & len(b1Files) == len(b7Files):
            
            
            print('in if loop')  
            # dminesion lenght of data cube
            #d = len(b1Files)
            #b1 = np.empty([d,r,c], dtype='int16')
            [sTime, sClock] = time.time(),time.clock()
            for i in range(0,len(b1Files)):
                print('data cube create loop: ',i) 
                # append band arrays
                with rasterio.open(b1Files[i]) as src:
                    tIn = src.read()
                    crsOut = src.crs
                    traOut = src.transform
                b1[i,:,:] = tIn.squeeze()
                #b1.append(b1In)
                
                # b2
                with rasterio.open(b2Files[i]) as src:
                    tIn = src.read()
                b3[i,:,:] = tIn.squeeze()
                #b2.append(b2In)
                
                # b3
                with rasterio.open(b3Files[i]) as src:
                    tIn = src.read()
                b3[i,:,:] = tIn.squeeze()
                #b3.append(b3In)
                
                # b4
                with rasterio.open(b4Files[i]) as src:
                    tIn = src.read()
                b4[i,:,:] = tIn.squeeze()
                #b4.append(b4In)
                
                
                # b5
                with rasterio.open(b5Files[i]) as src:
                    tIn = src.read()
                b5[i,:,:] = tIn.squeeze()
                #b5.append(b5In)
                
                
                # b6
                with rasterio.open(b6Files[i]) as src:
                    tIn = src.read()
                b6[i,:,:] = tIn.squeeze()
                #b6.append(b6In)
                
                # b7
                with rasterio.open(b7Files[i]) as src:
                    tIn = src.read()
                b7[i,:,:] = tIn.squeeze()
                #b7.append(b7In)
            
            # timing for numpy arrays
            [eTime, eClock] = time.time(),time.clock()  
            print('System time for numpy version is: ',eTime-sTime)
            print('Clock time for numpy version is: ',eClock-sClock) 
            
            # timing for append list
            [sTime, sClock] = time.time(),time.clock()
            
            for i in range(0,len(b1Files)):
                
                print('looping into append ',i) 
                # append band arrays
                with rasterio.open(b1Files[i]) as src:
                    b1In = src.read()
                    crsOut = src.crs
                    traOut = src.transform
                #b1[i,:,:] = b1In
                b1l.append(b1In)
                
                # b2
                with rasterio.open(b2Files[i]) as src:
                    b2In = src.read()
                #b1[i,:,:] = b1In
                b2l.append(b2In)
                
                # b3
                with rasterio.open(b3Files[i]) as src:
                    b3In = src.read()
                #b1[i,:,:] = b1In
                b3l.append(b3In)
                
                # b4
                with rasterio.open(b4Files[i]) as src:
                    b4In = src.read()
                #b1[i,:,:] = b1In
                b4l.append(b4In)
                
                
                # b5
                with rasterio.open(b5Files[i]) as src:
                    b5In = src.read()
                #b1[i,:,:] = b1In
                b5l.append(b5In)
                
                
                # b6
                with rasterio.open(b6Files[i]) as src:
                    b6In = src.read()
                #b1[i,:,:] = b1In
                b6l.append(b6In)
                
                # b7
                with rasterio.open(b7Files[i]) as src:
                    b7In = src.read()
                #b1[i,:,:] = b1In
                b7l.append(b7In)
            
            [eTime, eClock] = time.time(),time.clock()  
            print('System time for append version is: ',eTime-sTime)
            print('Clock time for append version is: ',eClock-sClock)                             
                
                #if i == 0:
                    
                    #with rasterio.open(b1Files[i]) as src:
                    #    b1In = src.read()
                    #b1[i,:,:] = b1In
                    #plt.imshow(b1.squeeze())
                    #plt.show()
                        
                #else:
                    #with rasterio.open(b1Files[i]) as src:
                    #    b1In = src.read()
                    
                    #b1[i,:,:]
                    #b1 = b1 + b1In 
                    #plt.imshow(b1.squeeze())
                    #plt.show()
    else:
        break
         #sys.exit([10])              
    
#    [eTime, eClock] = time.time(),time.clock()  
#    print('System time for cube create ',i,' is: ',eTime-sTime)
#    print('Clock time for cube create ',i,' is: ',eClock-sClock) 
    
#    [sTime, sClock] = time.time(),time.clock()
    
#    for i in range(0,len(b1Files)):
#        print('looping into append ',i) 
#        with rasterio.open(b1Files[i]) as src:
#            b1In = src.read()
#            b1M[i,:,:] = b1In.squeeze()
                #plt.imshow(b1.squeeze())
                #plt.show()
#    [eTime, eClock] = time.time(),time.clock()  
#    print('System time for matix itteration',i,' is: ',eTime-sTime)
#    print('Clock time for matrix itteration',i,' is: ',eClock-sClock)                   
          
#plt.imshow(b1.squeeze())
#t = np.array(l)


#[sTime, sClock] = time.time(),time.clock() 
#b1med = np.median(b1,axis=0)
#[eTime, eClock] = time.time(),time.clock()
#print('System time for median on list is: ',eTime-sTime)
#print('Clock time for median on list is: ',eClock-sClock)
#[sTime, sClock] = time.time(),time.clock() 
#b1Mmed = np.median(b1M,axis=0)
#[eTime, eClock] = time.time(),time.clock()
#print('System time for median on matrix is: ',eTime-sTime)
#print('Clock time for median on matrix is: ',eClock-sClock)

[sTime, sClock] = time.time(),time.clock()              
b1med = np.median(b1,axis=0)
b2med = np.median(b2,axis=0)
b3med = np.median(b3,axis=0)
b4med = np.median(b4,axis=0)
b5med = np.median(b5,axis=0)
b6med = np.median(b6,axis=0)
b7med = np.median(b7,axis=0)
[eTime, eClock] = time.time(),time.clock() 
print('System time for median on numpy array is: ',eTime-sTime)
print('Clock time for median on numpy array is: ',eClock-sClock)

[sTime, sClock] = time.time(),time.clock()              
b1lmed = np.median(b1l,axis=0)
b2lmed = np.median(b2l,axis=0)
b3lmed = np.median(b3l,axis=0)
b4lmed = np.median(b4l,axis=0)
b5lmed = np.median(b5l,axis=0)
b6lmed = np.median(b6l,axis=0)
b7lmed = np.median(b7l,axis=0)
[eTime, eClock] = time.time(),time.clock() 
print('System time for median on list is: ',eTime-sTime)
print('Clock time for median on list is: ',eClock-sClock)

plt.imshow(b1med.squeeze())

tcBright = b1med*ldBright[0] + b2med*ldBright[1] + b3med*ldBright[2] + \
    b4med*ldBright[3] + b5med*ldBright[4] + b6med*ldBright[5] + \
    b7med*ldBright[6]
tcGreen = b1med*ldGreen[0] + b2med*ldGreen[1] + b3med*ldGreen[2] + \
    b4med*ldGreen[3] + b5med*ldGreen[4] + b6med*ldGreen[5] + \
    b7med*ldGreen[6]
tcWet = b1med*ldWet[0] + b2med*ldWet[1] + b3med*ldWet[2] + \
    b4med*ldWet[3] + b5med*ldWet[4] + b6med*ldWet[5] + \
    b7med*ldWet[6]
    
    
#plt.imshow(tcBright.squeeze())   
#test file listings 
#for file in os.listdir(iDir):
#    if fnmatch.fnmatch(file, fPre+b1p+fPst):
#        print (file)
#        files = file

#   for year in years:
    
files = glob.glob(iDir+fPre+b1p+fPst)
#print(files[0])
#with rasterio.open(files[0]) as src:
#    b1 = src.read()

with rasterio.open(oDir + 'TCBright'+'.tif', 'w', driver='GTiff', height=tcBright.shape[1],
                   width=tcBright.shape[2], count=1, dtype='float64',
                   crs=crsOut, transform=traOut) as dst:
    dst.write(tcBright.squeeze(), 1)

#with rasterio.open(oDir + 'TCBright'+'.tif', 'w', driver='GTiff', height=tcBright.shape[1],
#                   width=tcBright.shape[2], count=1, dtype=tcBright.astype('float64').dtype,
#                   crs=crsOut, transform=traOut) as dst:
#    dst.write(tcBright.squeeze(), 1)


#plt.imshow(b1.squeeze())


#print(files)
