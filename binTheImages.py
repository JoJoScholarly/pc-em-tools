#!/usr/bin/env python3

import numpy as np
import astropy.io.fits as fits
import os
from sys import argv

import matplotlib.pyplot as plt

# File info
filepath = argv[1]
outfile = argv[2]
exten = 0

# Binning info
binMin = 0
binMax = 2000 #2**14
bins = np.linspace(binMin, binMax, int(binMax)-int(binMin) + 1)

# Crop info
xbeg = 0
xend = None
ybeg = 0
yend = None

# Initialize variables as nulls
counts = None
#bins = None

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

dataOut = list( zip(counts, bins) )

plt.step(bins[1:], counts, where='mid')
plt.yscale('log')
plt.show()

outfile = filepath + outfile
np.savetxt( outfile, dataOut )
