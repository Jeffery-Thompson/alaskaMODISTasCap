#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:11:54 2018

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
#iDir = '/Users/jeth6160/Desktop/permafrost/Arctic/WHRC/output/'
iShpDir = '/Users/jeth6160/Desktop/permafrost/BLM/FirePerimiters1940/'
iPziDir = '/Users/jeth6160/Desktop/permafrost/PermafrostZonationIndex/output/'
oDir = '/Users/jeth6160/Desktop/permafrost/Alaska/AppEARS/allAK/output/'

imgExt = '.tif'
shpExt ='.shp'

clusFile = 'AK_IntMon_Clu7' + imgExt

