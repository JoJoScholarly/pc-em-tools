#!/usr/bin/env python3

"""Helper to read a series of bias frames to fit the EM model in order to deduce
detector EM output characteristics."""

import numpy as np
import astropy.io.fits as fits
import configparser
import os
from sys import argv
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read("pc-tools.cfg")

# File info
filepath = config['files']['path']
outfilename = config['files']['histName']
exten = int(config['files']['exten'])

# Crop info, TODO needed?
xbeg = 0
xend = None
ybeg = 0
yend = None

# Histogram bin info
binMin = config['files']['binValMin']
binMax = config['files']['binValMax'] #2000 #2**14 TODO calculate based on bit depth?
bins = np.linspace(binMin, binMax, int(binMax)-int(binMin) + 1)

# Init
filecount = 0
counts = None

for filename in os.listdir( filepath ):
    if filename.endswith( ".fits" ):
        with fits.open( filepath + filename ) as hdul:
            data = hdul[exten].data
            data1D = data.reshape( data.shape[0]*data.shape[1] )
        if not isinstance(counts, np.ndarray):
            counts, binsOut = np.histogram( data1D, bins=bins )
        else:
            countsToAdd, binsOut = np.histogram( data1D, bins=bins )
            counts += countsToAdd
        filecount+=1

dataOut = list( zip(counts, bins) )

plt.step(bins[1:], counts, where='mid')
plt.yscale('log')
plt.show()

config['files']['filecount'] = filecount
outfile = filepath + outfilename
np.savetxt( outfile, dataOut )
