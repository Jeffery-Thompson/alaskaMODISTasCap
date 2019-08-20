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
#b1p = 'b01_doy'
#b2p = 'b02_doy'
#b3p = 'b03_doy'
#b4p = 'b04_doy'
#b5p = 'b05_doy'
#b6p = 'b06_doy'
#b7p = 'b07_doy'

# set loading matrices
#ldBright = [0.4395, 0.5945, 0.2460, 0.3918, 0.3506, 0.2136, 0.2678]
#ldGreen =[-0.4064, 0.5129, -0.2744, -0.2893, 0.4882, -0.0036, -0.4169]
#ldWet = [0.1147, 0.2489, 0.2408, 0.3132, -0.3122, -0.6416, -0.5087]

# set dimensions of images
r = 6542
c = 8514 

#########
#
# this dict holds the metadata as per NASA documentation.
#   is most of the important stuff for using the data
#
#   note: Not sure if xarray does autoscaling of the data, but some
#       but there are suggestions that some NetCDF readers should do that
#       using the values stored in scale_factor
#
#   here, values are basically bands 1-7, with QA, Solar View angles, QA and 
#       State Variables, and one for Day Of Year
#   
#########
m09a1BandDict = {'sur_refl_b01' : {'band':'red',
                             'band_long' : '500m Surface Reflectance Band 1 (620-670 nm)',
                             'band_number' : 1,
                             'units' : 'Reflectance',
                             'band_pass' : '620 - 670 nm',
                             'fill_value' : -28672,
                             'scale_factor' : 0.0001,
                             'valid_range_min':-100,
                             'valid_range_max':16000},
           'sur_refl_b02' : {'band' : 'nir',
                             'band_long' : '500m Surface Reflectance Band 2 (841-876 nm)',
                             'band_number' : 2,
                             'units' : 'Reflectance',
                             'band_pass' : '841 - 876 nm',
                             'fill_value' : -28672,
                             'scale_factor' : 0.0001,
                             'valid_range_min':-100,
                             'valid_range_max':16000},
           'sur_refl_b03' : {'band' : 'blue',
                             'band_long' : '500m Surface Reflectance Band 3 (459-479 nm)',
                             'band_number' : 3,
                             'units' : 'Reflectance',
                             'band_pass' : '459 - 479 nm',
                             'fill_value' : -28672,
                             'scale_factor' : 0.0001,
                             'valid_range_min':-100,
                             'valid_range_max':16000},
           'sur_refl_b04' : {'band' : 'green',
                            'band_long' : '500m Surface Reflectance Band 4 (545-565 nm)',
                            'band_number' : 4,
                            'units' : 'Reflectance',
                            'band_pass' : '545 - 565 nm',
                            'fill_value' : -28672,
                            'scale_factor' : 0.0001,
                            'valid_range_min':-100,
                            'valid_range_max':16000},
           'sur_refl_b05' : {'band' : 'nir2',
                            'band_long' : 'Near-Infrared (1230-1250 nm)',
                            'band_number' : 5,
                            'units' : 'Reflectance',
                            'band_pass' : '1230 - 1250 nm',
                            'fill_value' : -28672,
                            'scale_factor' : 0.0001,
                            'valid_range_min':-100,
                            'valid_range_max':16000},                  
           'sur_refl_b06' : {'band' : 'swir1',
                            'band_long' : 'Shortwave-Infrared (1628-1652 nm)',
                            'band_number' : 6,
                            'units' : 'Reflectance',
                            'band_pass' : '1628 - 1652 nm',
                            'fill_value' : -28672,
                            'scale_factor' : 0.0001,
                            'valid_range_min':-100,
                            'valid_range_max':16000},
           'sur_refl_b07' : {'band' : 'swir2',
                            'band_long' : 'Shortwave-Infrared (2105-2155 nm)',
                            'band_number' : 7,
                            'units' : 'Reflectance',
                            'band_pass' : '2105 - 2155 nm',
                            'fill_value' : -28672,
                            'scale_factor' : 0.0001,
                            'valid_range_min':-100,
                            'valid_range_max':16000},
           'sur_refl_qc_500m' : {'band' : 'QC',
                                 'band_long' : 'Reflectance Band Quality',
                                 'band_number' : 7,
                                 'units' : 'Bit Field',
                                 'band_pass' : '2105 - 2155 nm',
                                 'fill_value' : 4294967295,
                                 'scale_factor' : 'NA',
                                 'valid_range_min':'NA',
                                 'valid_range_max':'NA'},
           'sur_refl_szen' : {'band' : 'szen',
                              'band_long' : 'Solar Zenith Angle',
                              'band_number' : 'NA',
                              'units' : 'Degree',
                              'band_pass' : '2105 - 2155 nm',
                              'fill_value' : 0,
                              'scale_factor' : 0.01,
                              'valid_range_min':0,
                              'valid_range_max':18000},
           'sur_refl_vzen' : {'band' : 'vzen',
                              'band_long' : 'View Zenith Angle',
                              'band_number' : 'NA',
                              'units' : 'Degree',
                              'band_pass' : 'NA',
                              'fill_value' : 0,
                              'scale_factor' : 0.01,
                              'valid_range_min':0,
                              'valid_range_max':18000},
           'sur_refl_raz' : {'band' : 'vzen',
                             'band_long' : 'Relative Azimuth Angle',
                             'band_number' : 'NA',
                             'units' : 'Degree',
                             'band_pass' : 'NA',
                             'fill_value' : 0,
                             'scale_factor' : 0.01,
                             'valid_range_min':-18000,
                             'valid_range_max':18000},
           'sur_refl_state_500m' : {'band' : 'state',
                                    'band_long' : '500m State Flags',
                                    'band_number' : 'NA',
                                    'units' : 'Bit Field',
                                    'band_pass' : 'NA',
                                    'fill_value' : 65535,
                                    'scale_factor' : 0.01,
                                    'valid_range_min':0,
                                    'valid_range_max':18000},
           'sur_refl_day_of_year' : {'band' : 'doy',
                                    'band_long' : 'Julian Day',
                                    'band_number' : 'NA',
                                    'units' : 'Day of Year',
                                    'band_pass' : 'NA;',
                                    'fill_value' : 65535,
                                    'scale_factor' : 'NA',
                                    'valid_range_min':1,
                                    'valid_range_max':366}
        }

# dict for building metadata for these data
# is for the top level of xarray.... probably a better way to do this.
mod09a1DSDict = {'surf_ref' : {'long_name' : '500m Surface Reflectance',
                               'units' : 'Reflectance',
#                               '_FillValue' : np.array([-28672], dtype=np.int16),
                               'fill_value' : np.array([-28672], dtype=np.int16),
                               'scale_factor' : 0.0001,
                               'valid_range_min' : np.array([-100, 16000])},
                'qc_500m' : {'long_name' : 'Reflectance Band Quality',
                              'units' : 'Bit Field',
#                              '_FillValue' : np.array([4294967295], dtype=np.uint32)},
                              'fill_value' : np.array([4294967295], dtype=np.uint32)},
                'view_geom' : {'long_name' : 'Sensor Viewing Geometry',
                               'units' : 'Degree',
#                               '_FillValue' : np.array([0], dtype=np.int16),
                               'fill_value' : np.array([0], dtype=np.int16),
                               'scale_factor' : 0.01,
                               'valid_range' : np.array([-18000, 18000])},
                'state' : {'long_name' : '500m State Flags',
                           'units' : 'Bit Field',
#                           '_FillValue' : np.array([65535], dtype=np.uint16)},
                           'fill_value' : np.array([65535], dtype=np.uint16)},
                'doy' : {'long_name' : 'Julian Day',
                         'units' : 'Day of Year',
#                         '_FillValue' : np.array([65535], dtype=np.uint16),
                         'fill_value' : np.array([65535], dtype=np.uint16),
                         'valid_range' : np.array([1,366])}
                } 

# variables for processing through the data serially
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
        doy = []
        
        band_names=[]
        qc_names=[]
        solar_names=[]
        state_names=[]
        doy_names=[]

        #files = sorted(glob.glob(iDir+fPre+str(year)+str(day)+fPst))
        for key in [*m09a1BandDict]:
            file = glob.glob(iDir+fPre+str(key)+dPre+str(year)+str(day)+fPst)
            #file = glob.glob(fPre+str(key)+dPre+str(year)+str(day)+fPst)
            #print(file)
            print(key)
            
            #with rasterio.open(file, driver="GTiff") as src:
            #ndv = m09a1BandDict.get(key).get('fill_value')
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
                band_names.append(m09a1BandDict.get(key).get('band'))
                #print('data list appended')
                #print(file)
            elif key in ['sur_refl_qc_500m']:
                qc.append(tIn)
                qc_names.append(m09a1BandDict.get(key).get('band'))
                #print('qc list appended')
                #print(file)
            elif key in ['sur_refl_szen', 'sur_refl_vzen', 'sur_refl_raz']:
                solar.append(tIn)
                solar_names.append(m09a1BandDict.get(key).get('band'))
                #print('solar list appended')
                #print(file)
            elif key in ['sur_refl_state_500m']:
                state.append(tIn)
                state_names.append(m09a1BandDict.get(key).get('band'))
                #print('state list appended')
                #print(file)
            elif key in ['sur_refl_day_of_year']:
                doy.append(tIn)
                doy_names.append(m09a1BandDict.get(key).get('band'))


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
    
    # set the date string for the metadata
    date = datetime.datetime.strptime(str(year)+str(day), "%Y%j")
    
    # create the indiviudal DataArrays that make up the final DataSet
    #   essentially, get one dataset per calendear year, each one comprised
    #   of the composited remotely sensed observations for 8, 8-day periods
    #
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
                           coords = {'qc' : qc_names,
                                     'y' : y,
                                     'x' : x},
                            dims = ['qc', 'y', 'x'])
    

#    solCube = ma.masked_array(solar, fill_values = solar[0].get_fill_value())
    solarCube = np.asarray(solar).squeeze()
    solarOut =  xr.DataArray(solarCube, 
                             coords = {'geom' : solar_names,
                                       'y' : y,
                                       'x' : x},
                             dims = ['geom', 'y', 'x'])

    
#    stCube = ma.masked_array(state, fill_value = state[0].get_fill_value())
    stCube = np.asarray(state[0])
    stateOut =  xr.DataArray(stCube, 
                             coords = {'state' : state_names,
                                       'y' : y,
                                       'x' : x},
                             dims = ['state', 'y', 'x'])

    doyCube = np.asarray(doy[0])
    doyOut = xr.DataArray(doyCube, 
                             coords = {'doy' : doy_names,
                                       'y' : y,
                                       'x' : x},
                             dims = ['doy', 'y', 'x'])

    # make the data arrays into an xarray dataset
    mod09a1 = xr.Dataset({'surf_ref' : cubeOut,
                          'qc_500m' : qcOut,
                          'view_geom' : solarOut,
                          'state_500m' : stateOut,
                          'doy_500m' : doyOut,
                          'time':date})

    sr_out = xr.Dataset({'surf_ref': cubeOut})
  
#    plt.figure()
#    ndvi = (mod09ga['surf_ref'].sel(bands='nir') - mod09ga['surf_ref'].sel(bands='red')) / \
#    (mod09ga['surf_ref'].sel(bands='nir') + mod09ga['surf_ref'].sel(bands='red'))
#    plt.figure()
#    plt.imshow(ndvi)
#    qa_ndvi = xr.where(ndvi != 0,
#                   ndvi,
#                   np.nan)
#    plt.imshow(qa_ndvi, vmin=-1, vmax=1, cmap='gist_heat')

#########
#
# metadata for these datasets
#
#########

# frist, top pevel metadata
    mod09a1.attrs['title'] = str('MODIS Terra Surface Reflectance 8-Day L3 Global 500 m, Collecton 6')
    mod09a1.attrs['keywords'] = str('MODIS, Remote Sensing, Arctic, Surface Reflectance' )
    mod09a1.attrs['summary'] = str('''MOD09A1 is Level 3, 8-day composite build from daily MODIS band 1-7 surface reflectance observations.''')

    mod09a1.attrs['publisher_name'] = 'NASA'
    mod09a1.attrs['publisher_url'] = 'http://modis-sr.ltdri.org'
    mod09a1.attrs['publisher_email'] ='mod09@ltdri.org'

    mod09a1.attrs['time'] = str(date)
    
    mod09a1.attrs['time_coverage_start'] = str(date)
    mod09a1.attrs['time_coverage_end'] = str(date + datetime.timedelta(days=7))
    
    mod09a1.attrs['cdm_data_type'] = 'Grid'

    # do metadata for each of the bands/qa
    for key in [*mod09a1DSDict]:
        for k,v in mod09a1DSDict.get(key).items():
            mod09a1[key].attrs[k] = v
    
    
    # this bit is used to make the data searchable if the are ever posted/hosted.
    #   they are NOT the CRS type information needed to enable gdal etc to use the data
    # 
    '''These need worked on: not sure how to do since these are utm/or albers_alaksa'''
    '''These fields, especially geospatial_bounds_crs, requires an EPSG code format to 
        work, that is lame. below are what they would be if the were in lat/lons;
        from a gdalwarp reproject'''
#    mod09a1.attrs['geospatial_bounds'] = str(extentCoords(x, y, xOff, yOff, ptype='center'))
    mod09a1.attrs['geospatial_bounds'] = 'POLYGON ((-179.9999998 71.5869298, 179.9824586 71.5869298, 179.9824586 38.0891442, -179.9999998 38.0891442, -179.9999998 71.5869298))'
    mod09a1.attrs['geospatial_bounds_crs'] = 'epsg:4326'
    mod09a1.attrs['geospatial_lat_units'] = 'degrees_north'
    mod09a1.attrs['geospatial_lon_units'] = 'degrees_east'
    mod09a1.attrs['geospatial_lat_max'] = np.around(71.5869298, decimals=13)
    mod09a1.attrs['geospatial_lat_min'] = np.around(38.0891442, decimals=13)    
    mod09a1.attrs['geospatial_lon_max'] = np.around(179.9824586, decimals=13)  
    mod09a1.attrs['geospatial_lon_min'] = np.around(-179.9999998, decimals=13)  
    mod09a1.attrs['geospatial_lat_resolution'] = np.around(-0.027938103102464, decimals=14)  
    mod09a1.attrs['geospatial_lon_resolution'] = np.around(0.027935935005064, decimals=14)

    
    # getting to xarray/NetCDF to recognized spatial ref infor is not 
    #   immediately obvious. Crs needs setting as a coord, and then requires 
    #   explicit linking to the dataset via the 'grid_mapping' attribute
    #   
    #   if using something other than lon/lats, you often have to specific
    #       a specific, NetCDF enabled/supported CRS. not found an exhaustive list
    #       anywhere
    #
    ''' NEED defined and worked on: 
        https://www.unidata.ucar.edu/software/thredds/v4.5/netcdf-java/reference/StandardCoordinateTransforms.html'''    

    mod09a1.coords['crs'] = np.int32(0)
#    mod09a1.coords['_CoordinateAxes'] = 'bands qc geom state doy y x'
#    mod09a1.coords['_CoordinateTransforms'] = 'AlbersAlaskaWGS84Projection'
#    mod09a1.coords['_CoordinateTransforms'] = 'crs'
#    mod09a1.coords['crs'] = 'srorg:8815'
#    mod09a1.coords['crs'].attrs['grid_mapping'] = 'Projection'
    mod09a1.coords['crs'].attrs['grid_mapping_name'] = 'albers_conical_equal_area'
    mod09a1.coords['crs'].attrs['projected_coordinate_system_name'] = 'albers_conical_equal_area'
#    mod09a1.coords['crs'].attrs['standard_parallel'] = '55.0,  65.0' 
    mod09a1.coords['crs'].attrs['standard_parallel'] = np.array([55., 65.])
#    mod09a1.coords['crs'].attrs['standard_parallel_1'] = 55.
#    mod09a1.coords['crs'].attrs['standard_parallel_2'] = 65.
    mod09a1.coords['crs'].attrs['longitude_of_central_meridian'] = -154.0
    mod09a1.coords['crs'].attrs['latitude_of_projection_origin'] = 50.
    mod09a1.coords['crs'].attrs['false_easting'] = 0.0
    mod09a1.coords['crs'].attrs['false_northing'] = 0.0
    mod09a1.coords['crs'].attrs['_CoordinateTransformType'] = 'Projection'
    mod09a1.coords['crs'].attrs['_CoordinateAxisTypes'] = 'GeoX GeoY'
    mod09a1.coords['crs'].attrs['long_name'] = 'WGS84 / Alaskan Albers'
    mod09a1.coords['crs'].attrs['semi_major_axis'] = 6378137.0
    mod09a1.coords['crs'].attrs['semi_minor_axis'] = 6356752.314245
    mod09a1.coords['crs'].attrs['inverse_flattening'] = 298.257223563
#    mod09a1.coords['crs'].attrs['semi_major_axis'] = 6378137.0
#    mod09a1.coords['crs'].attrs['semi_minor_axis'] = 6356752.314140356
#    mod09a1.coords['crs'].attrs['inverse_flattening'] = 298.257222101
##    mod09a1.coords['crs'].attrs['crs_wkt'] = crsOut.wkt
##    mod09a1.coords['crs'].attrs['spatial_ref'] = crsOut.wkt
    mod09a1.coords['crs'].attrs['crs_wkt'] = crsOut.to_string()
    mod09a1.coords['crs'].attrs['spatial_ref'] = crsOut.to_string()
#
#    mod09a1.coords['crs'].attrs['GeoTransform'] = traOut
##    mod09a1.coords['crs'].attrs['unit'] = 'metre'
##    mod09a1.coords['crs'].attrs['crs_wkt'] = crsOut.wkt

    # going to try specifing things a different way
#    mod09a1.coords['AlbersAlaskaWGS84Projection'] = np.int32(0)
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['standard_parallel'] = np.array([55., 65.])
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['longitude_of_central_meridian'] = -154.0
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['latitude_of_projection_origin'] = 50.
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['false_easting'] = 0.0
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['false_northing'] = 0.0
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['_CoordinateTransformType'] = 'Projection'
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['_CoordinateAxisTypes'] = 'GeoX GeoY'
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['long_name'] = 'WGS84 / Alaskan Albers'
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['crs_wkt'] = crsOut.to_string()
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['spatial_ref'] = crsOut.to_string()
#    mod09a1.coords['AlbersAlaskaWGS84Projection'].attrs['GeoTransform'] = traOut


# more spatial ref needed to get the individual coords dealt with as well
    mod09a1['x'].attrs['standard_name'] = 'projection_x_coordinate'
#    mod09a1['x'].attrs['grid_mapping'] = 'crs'
    mod09a1['x'].attrs['long_name']= 'x coordinate of projection'
    mod09a1['x'].attrs['units'] = 'meters'
    mod09a1['x'].attrs['_CoordinateAxisType'] =  'GeoX'
#    mod09a1['x'].attrs['_CoordinateTransform'] =  'Projection'

    mod09a1['y'].attrs['standard_name'] = 'projection_y_coordinate'
#    mod09a1['y'].attrs['grid_mapping'] = 'crs'
    mod09a1['y'].attrs['long_name']= 'y coordinate of projection'
    mod09a1['y'].attrs['units'] = 'meters'
    mod09a1['y'].attrs['_CoordinateAxisType'] =  'GeoY'    
#    mod09a1['y'].attrs['_CoordinateTransform'] =  'Projection'
    # do the metadata for the individual data arrays, do using dicts

    mod09a1['time'].attrs['_CoordinateAxisType']='Time'
    #mod09a1.encoding['unlimited_dims']='time'
    
    # attach the CRS attribute to the datasets as well
    mod09a1['surf_ref'].attrs['grid_mapping'] = 'crs'
    mod09a1['qc_500m'].attrs['grid_mapping'] = 'crs'
    mod09a1['view_geom'].attrs['grid_mapping'] = 'crs'
    mod09a1['state_500m'].attrs['grid_mapping'] = 'crs'
    mod09a1['state_500m'].attrs['grid_mapping'] = 'crs'
    
    mod09a1.attrs['grid_mapping'] = 'crs'
    mod09a1.attrs['crs_wkt'] = crsOut.to_string()
    mod09a1.attrs['spatial_ref'] = crsOut.to_string()

    mod09a1.coords['time']=date
    mod09a1.coords['y']=('y', y)
    mod09a1.coords['x']=('x', x)
    
    
#    mod09a1['surf_ref'].attrs['_CoordinateSystems'] = 'AlbersAlaskaWGS84Projection'
#    mod09a1['qc_500m'].attrs['_CoordinateSystems'] = 'AlbersAlaskaWGS84Projection'
#    mod09a1['view_geom'].attrs['_CoordinateSystems'] = 'AlbersAlaskaWGS84Projection'
#    mod09a1['state_500m'].attrs['_CoordinateSystems'] = 'AlbersAlaskaWGS84Projection'
#    mod09a1['state_500m'].attrs['_CoordinateSystems'] = 'AlbersAlaskaWGS84Projection'
#    
#    mod09a1.attrs['grid_mapping'] = 'AlbersAlaskaWGS84Projection'

#    sr_out.coords['crs'] = np.int32(0)
#    sr_out.coords['_CoordinateAxes'] = 'bands qc geom state doy y x'
#    sr_out.coords['_CoordinateTransforms'] = 'AlbersAlaskaWGS84Projection'
#    sr_out.coords['AlbersAlaskaWGS84Projection'] = np.int32(0)
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['standard_parallel'] = np.array([55., 65.])
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['longitude_of_central_meridian'] = -154.0
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['latitude_of_projection_origin'] = 50.
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['false_easting'] = 0.0
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['false_northing'] = 0.0
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['_CoordinateTransformType'] = 'Projection'
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['_CoordinateAxisTypes'] = 'GeoX GeoY'
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['long_name'] = 'WGS84 / Alaskan Albers'
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['crs_wkt'] = crsOut.to_string()
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['spatial_ref'] = crsOut.to_string()
#    sr_out.coords['AlbersAlaskaWGS84Projection'].attrs['GeoTransform'] = traOut
#    sr_out['x'].attrs['standard_name'] = 'projection_x_coordinate'
##    mod09a1['x'].attrs['grid_mapping'] = 'Projection'
#    sr_out['x'].attrs['long_name']= 'x coordinate of projection'
#    sr_out['x'].attrs['units'] = 'meters'
#    sr_out['x'].attrs['_CoordinateAxisType'] =  'GeoX'
#
#    sr_out['y'].attrs['standard_name'] = 'projection_y_coordinate'
##    mod09a1['y'].attrs['grid_mapping'] = 'Projection'
#    sr_out['y'].attrs['long_name']= 'y coordinate of projection'
#    sr_out['y'].attrs['units'] = 'meters'
#    sr_out['y'].attrs['_CoordinateAxisType'] =  'GeoY' 
#
#    for k,v in mod09a1DSDict.get('surf_ref').items():
#        sr_out.attrs[k] = v

#    mod09a1.coords['x'] = 'projection_x_coordinate'
#    mod09a1.coords['y'] = 'projection_y_coordinate'

#if (w==True):
    print('writing NetCDF of Prism %s datacube for: %i ' %(year, day))
    #print('path is:', oPath + prism + '/' + oPre + str(year) + '.nc' +'\n' )
    mod09a1.to_netcdf(oDir + 'MOD09a1_' + str(year) + '_' + str(day) + '.nc',
                     mode='w',
                     format='NETCDF4',
                     unlimited_dims='time',
                     encoding={'surf_ref':{'zlib': True, 'complevel': 9},
                               'qc_500m':{'zlib': True, 'complevel': 9},
                               'view_geom':{'zlib': True, 'complevel': 9},
                               'state_500m':{'zlib': True, 'complevel': 9},
                               'doy_500m':{'zlib': True, 'complevel': 9},
                               'time':{'zlib': True, 'complevel': 9}})
    

#    mod09a1.to_netcdf(oDir + 'MOD09a1_bands' + str(year) + '_' + str(day) + '.nc',
#                     mode='w',
#                     format='NETCDF4',
#                     unlimited_dims='time',
#                     encoding={'surf_ref':{'zlib': True, 'complevel': 9}})
#

########
#
# helper funtions 
#
########
def extentCoords(lons, lats, xOff=0., yOff = 0., ptype='edge'):
    '''Function to calculate extent from lon & lat vectors. ptype used to 
    indicate if points are center of pixels, or upper left coners'''
    from shapely.geometry import box
    # minx = lons[0], miny=lats[-1], maxx=;ons[-1], maxy=lats[0] for clock-wise
    #   bounding box
    #mod this for centers/vs edges eventually?
    if ((ptype == 'center') & (xOff == 0.)):
        print('pixel type is center but offsets not provided')
        return()
    elif(ptype == 'edge'):
        bbox = box(lons[0], lats[-1], lons[-1], lats[0], ccw=False)
    elif (ptype == 'center'):
        xOff = np.around(lons[1] - lons[0], decimals=14)
        yOff = np.around(lats[1] - lats[0],decimals=14)        
        bbox = box(lons[0] - xOff/2, lats[-1] + yOff/2, 
                   lons[-1] + xOff/2, lats[0] - yOff/2, ccw=False)
    return(bbox)
