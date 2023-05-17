#!/usr/bin/env python

import astropy.io.fits as fits
import numpy as np
from sys import argv, exit
from math import erf
from scipy.stats import binom
import os

from maskCosmic import *
from fitBackground import *



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


def p_lb( lb, thresh, ron, EMprob, p_sCIC ):
    """Calculates probability of a given Poisson rate given detector parameters
    and threshold level using Harpsoe et al. (2012) eq. 24.

    :param lb: _description_
    :type lb: _type_
    :param thresh: _description_
    :type thresh: _type_
    :param ron: _description_
    :type ron: _type_
    :param EMprob: _description_
    :type EMprob: _type_
    :param p_sCIC: _description_
    :type p_sCIC: _type_
    :return: _description_
    :rtype: _type_
    """    
    # Stage count specific to E2V 201-20, TODO pull from config
    m = 604
    EMgain = calcEMgain( EMprob, m )

    S = 0
    for k in range(1, m):
        EMgain_k = calcEMgain( EMprob, k )
        S += (1 - np.e**(-thresh/EMgain_k))*np.e**(-lb)

    p_lb = ( np.e**(-thresh/EMgain)*( 1 - np.e**(-lb))
             + (np.e**(-lb) -m*p_sCIC)/2 * (1 - erf(thresh/ron/2**0.5) )
             + p_sCIC * S
            )
    return p_lb


def poissonRateParameter1( filepath, thresh, bias, ron, EMprob, p_sCIC, p_pCIC ):
    # Do thresholding for images
    # Detect cosmic rays as removal
    cosmics = None
    detections = None
    exten = 0
    N = 0

    for filename in os.listdir( filepath ):
        if filename.endswith(".fits"):
            with fits.open( filepath + filename ) as hdul:
                data = hdul[exten].data
            if N == 0:
                cosmics = np.zeros(data.shape)
                detections = np.zeros(data.shape)

            # Each cosmic ray gives negative count, i.e. missed frame
            cm = maskCosmicEM( data )
            cosmics += cm

            # Each (photo) electron counts as positive, cosmics masked
            mask = np.array(cm, dtype=bool)

            # Take a region from top of frame with little or no light on it
            overscan = data[:,1050:]
            overscan = overscan.reshape(overscan.shape[0]*overscan.shape[1])

            binMin = 0; binMax = 2000
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

    Q = detections #+ cosmics

    """
    # False positive rate because of ron raising above threshold
    # Harpsoe et al. (2012)
    lb_fp = N*(1 - erf(thresh/ron/2**0.5))/2/(N-Q)
    hdu = fits.PrimaryHDU(lb_fp)

    # Write out to file
    outfile = filepath + "falsePosRate.fits"
    hdu.writeto(outfile, overwrite=True)
    """

    return -np.log((N-Q) / N ), cosmics/N



def poissonRateParameter2( filepath, thresh, bias, ron, EMprob, p_sCIC, p_pCIC ):
    # Do thresholding for images
    # Detect cosmic rays as removal
    cosmics = None
    detections = None
    exten = 0
    N = 0

    for filename in os.listdir( filepath ):
        if filename.endswith(".fits"):
            with fits.open( filepath + filename ) as hdul:
                data = hdul[exten].data
            if N == 0:
                cosmics = np.zeros(data.shape)
                detections = np.zeros(data.shape)

            # Each cosmic ray gives negative count, i.e. missed frame
            cm = maskCosmicEM( data, saturationLimit=65536 )
            cosmics += cm

            # Each (photo) electron counts as positive, cosmics masked
            mask = np.array(cm, dtype=bool)

            # Take a region from top of frame with little or no light on it
            #overscan = data[:,530:590:]
            overscan = data[:,1050:]
            overscan = overscan.reshape(overscan.shape[0]*overscan.shape[1])

            binMin = 0; binMax = 2000
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
    Q_2d = detections #+ cosmics
    lb_E, lb_V = lbEstVar2d( N, Q_2d, thresh, ron, EMprob, p_sCIC )
    eff = (N-cosmics)/N

    return lb_E, lb_V, eff



def lbEstVar2d( N, Q_2d, thresh, ron, EMprob, p_sCIC ):
    # 1. Calculate all the possible values for E and V
    #    (N possibilities, less than calculating s)
    # 2. Look up for pixel values from 

    # Initialize look-up tabels
    lutE = []; lutV = []

    print("Calculating possible values for E(lb) and Var(lb)")

    # Calculate possible values for E(lb) and Var(lb)
    for Q in np.arange(N+1):
        E, V = lbEstVar( N, Q, thresh, ron, EMprob, p_sCIC )
        lutE.append(E)
        lutV.append(V)

    print("LUTs ready.")
    print(len(lutE), len(lutV))

    # Initialize output arrays
    lb_E = np.zeros(Q_2d.shape)
    lb_V = np.zeros(Q_2d.shape)

    # Fill in pre-calculated values drawing value the corresponding index of
    # the look-up tables
    for j in range(Q_2d.shape[0]):
        for k in range(Q_2d.shape[1]):
            Q = int(Q_2d[j,k])
            lb_E[j,k] = lutE[Q]
            lb_V[j,k] = lutV[Q]

    return lb_E, lb_V



def lbEstVar( N, Q, thresh, ron, EMprob, p_sCIC ):
    E = 0
    V = 0

    lb = -np.log( (N-Q) / N )

    for q in range(0, Q):
        lb_q = -np.log((N-q) / N )

        p = p_lb(lb_q, thresh, ron, EMprob, p_sCIC)
        a = binom.pmf( q, N, p )

        E += lb_q * a
        V += (lb_q - lb)**2 * a
    return E, V



if __name__ == "__main__":
    filepath = argv[1]
    #valMin = float(argv[2])
    #valMax = float(argv[3])

    configfilename = "pc-tools.cfg"
    config = configparser.ConfigParser()
    config.read( configfilename )

    thresh = float(config['pc']['threshold'])

    bias = float(config['detector']['biaslevel'])
    ron = float(config['detector']['readnoise'])
    EMprob = float(config['detector']['p_pCIC'])
    p_sCIC = float(config['detector']['p_sCIC'])
    p_pCIC = float(config['detector']['p_pCIC'])

    # Get the rate parameter estimate and variance
    img_E, img_V, eff = poissonRateParameter2( filepath, thresh, bias, ron, EMprob, p_sCIC, p_pCIC )

    # Sanity check Poisson direct Poisson rate without corrections
    img_lb, img_CR = poissonRateParameter1( filepath, thresh, bias, ron, EMprob, p_sCIC, p_pCIC )

    """
    # For a sanity check
    img_lb = poissonRateParameter1( filepath )
    outfile0 =  filepath + "lambda.fits"
    hdu1 = fits.PrimaryHDU( img_lb )
    hdu1.writeto( outfile0, overwrite=True )
    """

    # Write to file
    outfile1 = filepath + "lambda_E.fits"
    outfile2 = filepath + "lambda_Var.fits"
    outfile3 = filepath + "CR_eff.fits"
    outfile4 = filepath + "lambda_lb.fits"

    hdu1 = fits.PrimaryHDU( img_E )
    hdu1.writeto( outfile1, overwrite=True )

    hdu2 = fits.PrimaryHDU( img_V )
    hdu2.writeto( outfile2, overwrite=True )

    hdu3 = fits.PrimaryHDU( eff )
    hdu3.writeto( outfile3, overwrite=True )

    hdu4 = fits.PrimaryHDU( img_lb )
    hdu4.writeto( outfile4, overwrite=True )
