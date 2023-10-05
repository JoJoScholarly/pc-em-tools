#!/usr/bin/env python

import astropy.io.fits as fits
import numpy as np
from sys import argv
from math import erf
from scipy.stats import binom
import configparser
import glob

from maskCosmic import *
from fitBackground import *

configfilename = "pc-tools.cfg"
config = configparser.ConfigParser()
config.read( configfilename )

thresh = float(config['pc']['threshold'])

binMin = float(config['files']['binvalmin'])
binMax = float(config['files']['binvalmax'])

bias = float(config['detector']['biaslevel'])
ron = float(config['detector']['readnoise'])
p_EM = float(config['detector']['p_em'])
p_sCIC = float(config['detector']['p_sCIC'])
p_pCIC = float(config['detector']['p_pCIC'])
stageCount = float(config['detector']['stagecount'])



def threshold( data, thresh ):
    """Function takes an image and threshold level as input and returns an array
    with same dimension with values above threshold valued 1 and values below
    threshold level 0.

    :param iamge: Input image frame
    :type data: Arr[int]
    :param thresh: Threshold level
    :type thresh: int
    :return: Array with the input image dimensions with 0 if input below 
    threshold and 1 if above threshold.
    :rtype: Arr[int]
    """    
    data[ data<=thresh ] = 0
    data[ data>=thresh ] = 1

    return data



def p_lb( lb, thresh, ron, p_EM, stageCount, p_sCIC ):
    """Calculates probability of a given Poisson rate given detector parameters
    and threshold level using Harpsoe et al. (2012) eq. 24.

    :param lb: _description_
    :type lb: _type_
    :param thresh: _description_
    :type thresh: _type_
    :param ron: _description_
    :type ron: _type_
    :param p_EM: _description_
    :type p_EM: _type_
    :param p_sCIC: _description_
    :type p_sCIC: _type_
    :return: _description_
    :rtype: _type_
    """    
    m = stageCount
    EMgain = calcEMgain( p_EM, m )

    S = 0
    for k in np.arange(1, m+1):
        EMgain_k = calcEMgain( p_EM, k )
        S += (1 - np.e**(-thresh/EMgain_k))*np.e**(-lb)

    p_lb = ( np.e**(-thresh/EMgain)*( 1 - np.e**(-lb))
             + (np.e**(-lb) -m*p_sCIC)/2 * (1 - erf(thresh/ron/2**0.5) )
             + p_sCIC * S
            )
    return p_lb



def lbEstVar( N, Q, thresh, ron, p_EM, p_sCIC ):
    # Harpsoe+12 Eq. 25 and 26
    E = 0
    V = 0

    # Correction term for overestimation by RON raising above threshold
    # Affects only low gains
    corr = N*(1 - erf(thresh/ron/2**0.5) ) / 2 / (N-Q) 

    # Estimate lambda and probability for lambda
    lb = -np.log( (N-Q) / N ) - corr
    p = p_lb(lb, thresh, ron, p_EM, stageCount, p_sCIC)

    # Calculate best estimator and variance
    for q in np.arange(0, N):
        lb_q = -np.log((N-q) / N )
        a = binom.pmf( q, N, p )

        E += lb_q * a
        V += (lb_q - lb)**2 * a
    return E, V



def lbEstVar2d( N, Q_2d, thresh, ron, p_EM, p_sCIC ):
    # 1. Calculate all the possible values for E and V
    #    (N possibilities, less than calculating s)
    # 2. Look up for pixel value from the LUTs

    # Initialize look-up tabels
    lutE = np.zeros(N)
    lutV = np.zeros(N)

    print("Calculating Look-Up Tables for possible E(lb) and Var(lb) values")

    # Calculate all possible values for E(lb) and Var(lb) given the frame count
    for Q in np.arange(0, N-1):
        E, V = lbEstVar( N, Q, thresh, ron, p_EM, p_sCIC )
        lutE[Q] = E
        lutV[Q] = V

    print("LUTs ready.")

    # Initialize output arrays
    lb_E = np.zeros(Q_2d.shape)
    lb_V = np.zeros(Q_2d.shape)

    # Fill in the 2D array with by drawing LUT values with corresponding index
    for j in np.arange(Q_2d.shape[0]):
        for k in np.arange(Q_2d.shape[1]):
            Q = int(Q_2d[j,k])
            lb_E[j,k] = lutE[Q]
            lb_V[j,k] = lutV[Q]

    return lb_E, lb_V



def determinePoissonRate( filepath, thresh, bias, ron, p_EM, p_sCIC, crLimit ):
    # Do thresholding for images
    # Detect cosmic rays as removal
    cosmics = None
    detections = None
    exten = 0
    N = 0

    #for filename in os.listdir( filepath ):
    for filename in glob.iglob( filepath + '**/*.fits', recursive=True):
        if filename.endswith(".fits"):
            with fits.open( filename ) as hdul:
                data = hdul[exten].data
            if N == 0:
                cosmics = np.zeros(data.shape)
                detections = np.zeros(data.shape)

            # Each cosmic ray gives negative count, i.e. missed frame
            cm = maskCosmicEM( data, crLimit )
            cosmics += cm

            # Each (photo) electron counts as positive, cosmics masked
            mask = np.array(cm, dtype=bool)

            # Take a region from top of frame with little or no light on it
            #overscan = data[:,530:590:]
            overscan = data[:,1050:]
            overscan = overscan.reshape(overscan.shape[0]*overscan.shape[1])

            binMin = 0; binMax = 9000
            bins = np.linspace(binMin, binMax, int(binMax)-int(binMin) + 1)
            counts, binsOut = np.histogram(overscan, bins=bins)

            overscanHist = np.vstack([counts, binsOut[1:]]).T

            Norm, bias, ron = fitBias( overscanHist, bias=bias, readnoise=ron, plotFig=False )

            thresholdLevel = round(bias + thresh)

            aboveThreshold = threshold( data, thresh=thresholdLevel )
            aboveThreshold[mask] = 0
            detections += aboveThreshold

            # Number of frames up
            N += 1

    print("Frames read in: ", N)

    # Take the detection rates and calculate lambda estimate
    Q_2d = detections
    lb_E, lb_V = lbEstVar2d( N, Q_2d, thresh, ron, p_EM, p_sCIC )
    eff = (N-cosmics)/N

    return lb_E, lb_V, eff



if __name__ == "__main__":
    filepath = argv[1]

    configfilename = "pc-tools.cfg"
    config = configparser.ConfigParser()
    config.read( configfilename )

    thresh = float(config['pc']['threshold'])

    bias = float(config['detector']['biaslevel'])
    ron = float(config['detector']['readnoise'])
    p_EM = float(config['detector']['p_em'])
    p_sCIC = float(config['detector']['p_sCIC'])
    p_pCIC = float(config['detector']['p_pCIC'])
    stageCount = int(config['detector']['stagecount'])
    crLimit = float(config['pc']['crlimit'])

    # Get the rate parameter estimate, variance and CR efficiency
    img_E, img_V, eff = determinePoissonRate( filepath, thresh, bias, ron, p_EM, p_sCIC, crLimit )

    # Write to file
    outfile1 = filepath + "lambda_E.fits"
    outfile2 = filepath + "lambda_Var.fits"
    outfile3 = filepath + "CR_eff.fits"

    hdu1 = fits.PrimaryHDU( img_E )
    hdu1.writeto( outfile1, overwrite=True )

    hdu2 = fits.PrimaryHDU( img_V )
    hdu2.writeto( outfile2, overwrite=True )

    hdu3 = fits.PrimaryHDU( eff )
    hdu3.writeto( outfile3, overwrite=True )
