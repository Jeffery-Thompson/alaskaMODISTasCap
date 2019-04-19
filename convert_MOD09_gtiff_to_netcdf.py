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
import numpy.ma as ma
import pylab as pl
import scipy as sp
import matplotlib.pyplot as plt
import datetime # datetime commands
import xarray as xr
import re # python regular expressions
import os # python os tools
import fnmatch # function matching tools
import rasterio
import glob
import time
#import h5py
#import re

# set input output directories
iDir = '/Volumes/spatialData/earth_lab/alaskaTasseledCap/AppEARS/allAK/MOD09A1/'
oDir = '/Volumes/spatialData/earth_lab/alaskaTasseledCap/AppEARS/allAK/output/'

#print(iDir,oDir)

# set file pre/post-fixes
#fPre = 'MOD09A1.005_sur_refl_'
fPre = 'MOD09A1.006_'
dPre='_doy'
fPst = '*.tif'
print(fPre,fPst)

# set mask value for input & output data
mask_in =-28672
mask_out = mask_in
#mask_out = -9999

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


modDict = {'sur_refl_b01' : {'band':'red',
                             'band_long' : '500m Surface Reflectance Band 1 (620-670 nm)',
                             'band_number' : 1,
                             'units' : 'Reflectance',
                             'band_pass' : '620 - 670 nm',
                             'fill_value' : -28672,
                             'scale_factor' : 0.0001,
                             'valid_range' : {'min':-100,
                                              'max':16000}
                             },
           'sur_refl_b02' : {'band' : 'nir',
                             'band_long' : '500m Surface Reflectance Band 2 (841-876 nm)',
                             'band_number' : 2,
                             'units' : 'Reflectance',
                             'band_pass' : '841 - 876 nm',
                             'fill_value' : -28672,
                             'scale_factor' : 0.0001,
                             'valid_range' : {'min':-100,
                                              'max':16000}
                             },
           'sur_refl_b03' : {'band' : 'blue',
                             'band_long' : '500m Surface Reflectance Band 3 (459-479 nm)',
                             'band_number' : 3,
                             'units' : 'Reflectance',
                             'band_pass' : '459 - 479 nm',
                             'fill_value' : -28672,
                             'scale_factor' : 0.0001,
                             'valid_range' : {'min':-100,
                                             'max':16000}
                             },
           'sur_refl_b04' : {'band' : 'green',
                            'band_long' : '500m Surface Reflectance Band 4 (545-565 nm)',
                            'band_number' : 4,
                            'units' : 'Reflectance',
                            'band_pass' : '545 - 565 nm',
                            'fill_value' : -28672,
                            'scale_factor' : 0.0001,
                            'valid_range' : {'min':-100,
                                             'max':16000}
                            },
           'sur_refl_b05' : {'band' : 'nir2',
                            'band_long' : 'Near-Infrared (1230-1250 nm)',
                            'band_number' : 5,
                            'units' : 'Reflectance',
                            'band_pass' : '1230 - 1250 nm',
                            'fill_value' : -28672,
                            'scale_factor' : 0.0001,
                            'valid_range' : {'min':-100,
                                             'max':16000}
                            },                  
           'sur_refl_b06' : {'band' : 'swir1',
                            'band_long' : 'Shortwave-Infrared (1628-1652 nm)',
                            'band_number' : 6,
                            'units' : 'Reflectance',
                            'band_pass' : '1628 - 1652 nm',
                            'fill_value' : -28672,
                            'scale_factor' : 0.0001,
                            'valid_range' : {'min':-100,
                                             'max':16000}
                            },
           'sur_refl_b07' : {'band' : 'swir2',
                            'band_long' : 'Shortwave-Infrared (2105-2155 nm)',
                            'band_number' : 7,
                            'units' : 'Reflectance',
                            'band_pass' : '2105 - 2155 nm',
                            'fill_value' : -28672,
                            'scale_factor' : 0.0001,
                            'valid_range' : {'min':-100,
                                             'max':16000}
                            },
           'sur_refl_qc_500m' : {'band' : 'QA',
                                 'band_long' : 'Qualtiy Assurance',
                                 'band_number' : 7,
                                 'units' : 'Bit Field',
                                 'band_pass' : '2105 - 2155 nm',
                                 'fill_value' : 4294967295,
                                 'scale_factor' : None,
                                 'valid_range' : {'min':None,
                                                  'max':None}
                            },
           'sur_refl_szen' : {'band' : 'szen',
                              'band_long' : 'Solar Zenith Angle',
                              'band_number' : None,
                              'units' : 'Degree',
                              'band_pass' : '2105 - 2155 nm',
                              'fill_value' : 0,
                              'scale_factor' : 0.01,
                              'valid_range' : {'min':0,
                                             'max':18000}
                              },
           'sur_refl_vzen' : {'band' : 'vzen',
                              'band_long' : 'View Zenith Angle',
                              'band_number' : None,
                              'units' : 'Degree',
                              'band_pass' : None,
                              'fill_value' : 0,
                              'scale_factor' : 0.01,
                              'valid_range' : {'min':0,'max':18000}
                              },
           'sur_refl_raz' : {'band' : 'vzen',
                             'band_long' : 'Relative Azimuth Angle',
                             'band_number' : None,
                             'units' : 'Degree',
                             'band_pass' : None,
                             'fill_value' : 0,
                             'scale_factor' : 0.01,
                             'valid_range' : {'min':-18000,
                                             'max':18000}
                             },
           'sur_refl_state_500m' : {'band' : 'state',
                                    'band_long' : '500m State Flags',
                                    'band_number' : None,
                                    'units' : 'Bit Field',
                                    'band_pass' : None,
                                    'fill_value' : 65535,
                                    'scale_factor' : 0.01,
                                    'valid_range' : {'min':0,
                                                     'max':18000}
                                    },
           'sur_refl_day_of_year' : {'band' : 'doy',
                                    'band_long' : 'Julian Day',
                                    'band_number' : None,
                                    'units' : 'Day of Year',
                                    'band_pass' : None,
                                    'fill_value' : 65535,
                                    'scale_factor' : 0.01,
                                    'valid_range' : {'min':1,
                                                     'max':366}
                                    }
        }
           
years = range(2000,2018)
days = [185, 193, 201, 209, 217, 225, 233, 241]

print(list(years))

#for year in years:
#b1= []
#b2= []
#b3= []
#b4= []
#b5= []
#b6= []
#b7= []

#os.chdir(iDir)

for year in [years[0]]:
#for year in years:
    for day in [days[0]]:
        data=[]
        qc=[]
        solar=[]
        state=[]
        band_names=[]
        qc_names=[]
        state_names=[]
        solar_names=[]
        #files = sorted(glob.glob(iDir+fPre+str(year)+str(day)+fPst))
        for key in [*modDict]:
            file = glob.glob(iDir+fPre+str(key)+dPre+str(year)+str(day)+fPst)
            #file = glob.glob(fPre+str(key)+dPre+str(year)+str(day)+fPst)
            #print(file)
            print(key)
            
            #with rasterio.open(file, driver="GTiff") as src:
            #ndv = modDict.get(key).get('fill_value')
            with rasterio.open(file[0], driver='GTiff') as src:
                #tIn = src.read(1, masked=True)
                tIn = src.read()
                #tIn = src.read()
                crsOut = src.crs
                traOut = src.transform
                bBox = src.bounds
                rows = src.height
                cols = src.width
            
            if key in ['sur_refl_b01', 'sur_refl_b02',  'sur_refl_b03',
                       'sur_refl_b04', 'sur_refl_b05','sur_refl_b06', 
                       'sur_refl_b07']:
                data.append(tIn)
                band_names.append(modDict.get(key).get('band'))
                #print('data list appended')
                #print(file)
            elif key in ['sur_refl_qc_500m']:
                qc.append(tIn)
                qc_names.append(modDict.get(key).get('band'))
                #print('qc list appended')
                #print(file)
            elif key in ['sur_refl_szen', 'sur_refl_vzen', 'sur_refl_raz']:
                solar.append(tIn)
                solar_names.append(modDict.get(key).get('band'))
                #print('solar list appended')
                #print(file)
            elif key in ['sur_refl_state_500m', 'sur_refl_day_of_year']:
                state.append(tIn)
                state_names.append(modDict.get(key).get('band'))
                #print('state list appended')
                #print(file)


    # set geospatail coords manually for now; not sure how well this deals with 
    #   Albers
    iX = traOut[2] + traOut[0]/2
    #fX = -66.4583333333286
    #fX = iX * cols
    
    iY = traOut[5] + traOut[4]/2
    #fY = 24.0416666666666
    #fY = iY * rows
    
    xOff = traOut[0]
    yOff = traOut[4]    
    
    # hard code for now. issue here is that transform function is returning
    #   edges for, not actual pixel centers; which causing probles
        #lons = np.arange(iX, fX, xOff, dtype='float32')
        #lats = np.arange(iY, fY, yOff, dtype='float32')
    x = [(iX + xOff * i) for i in range(0,cols)]
    x = np.asarray(x)
    y = [(iY + yOff * i) for i in range(0,rows)]
    y = np.asarray(y)
    
    date = datetime.datetime.strptime(str(year)+str(day), "%Y%j")
    
    #dCube = ma.masked_array(data, fill_value = data[0].get_fill_value())
    dCube = np.asarray(data).squeeze()       
    cubeOut = xr.DataArray(dCube, 
                           coords = {'bands' : band_names,
                                     'y' : y,
                                     'x' : x},
                            dims = ['bands', 'y', 'x'])
    
    
#    qcCube = ma.masked_array(qc, fill_value = qc[0].get_fill_value())  
#    qcCube = np.asarray(qc).squeeze()
    qcCube = np.asarray(qc[0])
    qcOut =  xr.DataArray(qcCube, 
                           coords = {'bands' : qc_names,
                                     'y' : y,
                                     'x' : x},
                            dims = ['bands', 'y', 'x'])
    

#    solCube = ma.masked_array(solar, fill_values = solar[0].get_fill_value())
    solarCube = np.asarray(solar).squeeze()
    solarOut =  xr.DataArray(solarCube, 
                             coords = {'geom' : solar_names,
                                       'y' : y,
                                       'x' : x},
                             dims = ['geom', 'y', 'x'])

    
#    stCube = ma.masked_array(state, fill_value = state[0].get_fill_value())
    stCube = np.asarray(state).squeeze()
    stateOut =  xr.DataArray(stCube, 
                             coords = {'state' : state_names,
                                       'y' : y,
                                       'x' : x},
                             dims = ['state', 'y', 'x'])

    
    mod09ga = xr.Dataset({'surf_ref' : cubeOut,
                          'qc_500m' : qcOut,
                          'view_geom' : solarOut,
                          'state_doy' : stateOut})
  
    
    
    l8_surfref = xr.Dataset({'surf_ref': l8bands, 'qa': l8qa})





    mergeOut = xr.concat(cubeOut, qcOut, solarOut, stateOut, dim=['y','x'])
    
    
    b1Files = sorted(glob.glob(iDir+fPre+b1p+str(year)+fPst))
    b2Files = sorted(glob.glob(iDir+fPre+b2p+str(year)+fPst))
    b3Files = sorted(glob.glob(iDir+fPre+b3p+str(year)+fPst))
    b4Files = sorted(glob.glob(iDir+fPre+b4p+str(year)+fPst))
    b5Files = sorted(glob.glob(iDir+fPre+b5p+str(year)+fPst))
    b6Files = sorted(glob.glob(iDir+fPre+b6p+str(year)+fPst))
    b7Files = sorted(glob.glob(iDir+fPre+b7p+str(year)+fPst))
    
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
            
            
            print('in if loop, year is: ', year)  
            # dminesion lenght of data cube
            #d = len(b1Files)
            #b1 = np.empty([d,r,c], dtype='int16')
            [sTime, sClock] = time.time(),time.clock()
            for i in range(0,len(b1Files)):
                print('data cube create loop: ',i) 
                # append band arrays
                with rasterio.open(b1Files[i],nodatavals = mask_in) as src:
                    tIn = src.read(1, masked = True)
                    crsOut = src.crs
                    traOut = src.transform
                b1[i,:,:] = tIn.squeeze()
                #b1.append(b1In)
                
                # b2
                with rasterio.open(b2Files[i],nodatavals = mask_in) as src:
                    tIn = src.read(1, masked = True)
                b2[i,:,:] = tIn.squeeze()
                #b2.append(b2In)
                
                # b3
                with rasterio.open(b3Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b3[i,:,:] = tIn.squeeze()
                #b3.append(b3In)
                
                # b4
                with rasterio.open(b4Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b4[i,:,:] = tIn.squeeze()
                #b4.append(b4In)
                
                
                # b5
                with rasterio.open(b5Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b5[i,:,:] = tIn.squeeze()
                #b5.append(b5In)
                
                
                # b6
                with rasterio.open(b6Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b6[i,:,:] = tIn.squeeze()
                #b6.append(b6In)
                
                # b7
                with rasterio.open(b7Files[i],nodatavals = mask_in) as src:
                    tIn = src.read()
                b7[i,:,:] = tIn.squeeze()
                #b7.append(b7In)
            
            # timing for numpy arrays
            [eTime, eClock] = time.time(),time.clock()  
#            print('System time for numpy version is: ',eTime-sTime)
#            print('Clock time for numpy version is: ',eClock-sClock) 
            
  
            #print('System time for append version is: ',eTime-sTime)
            #print('Clock time for append version is: ',eClock-sClock)                             
             
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
 
    # generate median data
    b1med = np.median(b1,axis=0)
    b2med = np.median(b2,axis=0)
    b3med = np.median(b3,axis=0)
    b4med = np.median(b4,axis=0)
    b5med = np.median(b5,axis=0)
    b6med = np.median(b6,axis=0)
    b7med = np.median(b7,axis=0)
            
    # generate yearly median TC images
            
    tcBright = b1med*ldBright[0] + b2med*ldBright[1] + b3med*ldBright[2] + \
        b4med*ldBright[3] + b5med*ldBright[4] + b6med*ldBright[5] + \
        b7med*ldBright[6]
            
    tcGreen = b1med*ldGreen[0] + b2med*ldGreen[1] + b3med*ldGreen[2] + \
        b4med*ldGreen[3] + b5med*ldGreen[4] + b6med*ldGreen[5] + \
        b7med*ldGreen[6]
            
    tcWet = b1med*ldWet[0] + b2med*ldWet[1] + b3med*ldWet[2] + \
        b4med*ldWet[3] + b5med*ldWet[4] + b6med*ldWet[5] + \
        b7med*ldWet[6]
    
    br_i = np.where(tcBright == -54743.4496)
    tcBright[br_i] = mask_out
    
    gr_i = np.where(tcGreen == 25873.612799999995)
    tcGreen[gr_i] = mask_out
    
    wt_i = np.where(tcWet == 22759.833599999998)
    tcWet[wt_i] = mask_out
        
            # write files out
            
    with rasterio.open(oDir + str(year) + '_TCBright'+'.tif', 'w', driver='GTiff', height=tcBright.shape[0],
                       width=tcBright.shape[1], count=1, dtype='float64',
                       crs=crsOut, transform=traOut, nodata=mask_out) as dst:
        dst.write(tcBright.squeeze(), 1)
"""                
    with rasterio.open(oDir + str(year) + '_TCGreen'+'.tif', 'w', driver='GTiff', height=tcGreen.shape[0],
                       width=tcGreen.shape[1], count=1, dtype='float64',
                       crs=crsOut, transform=traOut, nodata=mask_out) as dst:
        dst.write(tcGreen.squeeze(), 1)
                
    with rasterio.open(oDir + str(year) + '_TCWet'+'.tif', 'w', driver='GTiff', height=tcWet.shape[0],
                       width=tcWet.shape[1], count=1, dtype='float64',
                       crs=crsOut, transform=traOut, nodata=mask_out) as dst:
        dst.write(tcWet.squeeze(), 1)    
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
"""

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

#[sTime, sClock] = time.time(),time.clock()              
#b1med = np.median(b1,axis=0)
#b2med = np.median(b2,axis=0)
#b3med = np.median(b3,axis=0)
#b4med = np.median(b4,axis=0)
#b5med = np.median(b5,axis=0)
#b6med = np.median(b6,axis=0)
#b7med = np.median(b7,axis=0)
#[eTime, eClock] = time.time(),time.clock() 
#print('System time for median on numpy array is: ',eTime-sTime)
#print('Clock time for median on numpy array is: ',eClock-sClock)

#[sTime, sClock] = time.time(),time.clock()              
#b1lmed = np.median(b1l,axis=0)
#b2lmed = np.median(b2l,axis=0)
#b3lmed = np.median(b3l,axis=0)
#b4lmed = np.median(b4l,axis=0)
#b5lmed = np.median(b5l,axis=0)
#b6lmed = np.median(b6l,axis=0)
#b7lmed = np.median(b7l,axis=0)
#[eTime, eClock] = time.time(),time.clock() 
#print('System time for median on list is: ',eTime-sTime)
#print('Clock time for median on list is: ',eClock-sClock)

#plt.imshow(b1med.squeeze())

#tcBright = b1med*ldBright[0] + b2med*ldBright[1] + b3med*ldBright[2] + \
#    b4med*ldBright[3] + b5med*ldBright[4] + b6med*ldBright[5] + \
#    b7med*ldBright[6]
#tcGreen = b1med*ldGreen[0] + b2med*ldGreen[1] + b3med*ldGreen[2] + \
#    b4med*ldGreen[3] + b5med*ldGreen[4] + b6med*ldGreen[5] + \
#    b7med*ldGreen[6]
#tcWet = b1med*ldWet[0] + b2med*ldWet[1] + b3med*ldWet[2] + \
#    b4med*ldWet[3] + b5med*ldWet[4] + b6med*ldWet[5] + \
#    b7med*ldWet[6]
    
    
#plt.imshow(tcBright.squeeze())   
#test file listings 
#for file in os.listdir(iDir):
#    if fnmatch.fnmatch(file, fPre+b1p+fPst):
#        print (file)
#        files = file

#   for year in years:
    
#files = glob.glob(iDir+fPre+b1p+fPst)
#print(files[0])
#with rasterio.open(files[0]) as src:
#    b1 = src.read()

#with rasterio.open(oDir + 'TCBright'+'.tif', 'w', driver='GTiff', height=tcBright.shape[1],
#                   width=tcBright.shape[2], count=1, dtype='float64',
#                   crs=crsOut, transform=traOut) as dst:
#    dst.write(tcBright.squeeze(), 1)

#with rasterio.open(oDir + 'TCBright'+'.tif', 'w', driver='GTiff', height=tcBright.shape[1],
#                   width=tcBright.shape[2], count=1, dtype=tcBright.astype('float64').dtype,
#                   crs=crsOut, transform=traOut) as dst:
#    dst.write(tcBright.squeeze(), 1)


#plt.imshow(b1.squeeze())


#print(files)