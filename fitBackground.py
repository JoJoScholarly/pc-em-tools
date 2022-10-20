#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from iminuit import Minuit
from scipy.stats import norm, poisson, chi2
from scipy.special import gamma
import configparser
import os
import sys

sys.path.append(
    '/home/jojo/Code/misc/')

from ExternalFunctions import Chi2Regression


configFile = "config.ini"
config = configparser.ConfigParser()
config.read( configFile )

"""
- Take in a stack of bias frames and merge
- Fit to get the profile for read noise
- Take the observed bias and 

"""

def gaussian( x, N, mu, sigma ):
    return N * norm.pdf(x, mu, sigma)


def EMsingleStageProbability( EMgain, stageCount ):
    return EMgain**(1/stageCount) - 1


def calcEMgain( singleStageProbability, stageCount ):
    return ( 1 + singleStageProbability )**stageCount


def pCICpdf( x, EMgain ):
    return np.exp(-x/EMgain) / EMgain * np.heaviside(x, 0)


def sCICpdf( x, stageCount, singleStageProbability ):
    sum = 0
    for i in np.arange(1, stageCount+1):
        gain = calcEMgain( singleStageProbability, i)
        sum += np.exp( -x/gain ) / gain * np.heaviside(x, 0)
    return sum



def stackBias( filepath, exten=0 ):
    biaslist = []
    for filename in os.listdir():
        if filename.endswith(".fits"):
            with fits.open( filename ) as hdul:
                data = hdul[exten].data
                hdr  = hdul[exten].header
                biaslist.append( data )

    mbias = np.median( biaslist, axis=0 )
    hdu = fits.PrimaryHDU( mbias )
    hdu.writeto( "mbias.fits", overwrite=True )



def EMbiasModel( data, N, mu, sigma, pp, ps, EMprob ):
    # Uses Harpsoe et al. 2012 paper eq. 17 to model the EM output. Get the
    # probability of parallel and serial CIC event as an output. 

    stageCount = 604
    ignoreSerialCIC = False

    EMgain = calcEMgain( EMprob, stageCount )

    # Substract bias level since the model will not be correct otherwise
    #data = data - mu

    # RON model with zero mean
    R = norm.pdf(data, mu, sigma )

    if ignoreSerialCIC:
        return N * ( (1 - pp) * R
                     + pp*pCICpdf( data, EMgain ) * np.heaviside(data, mu)
                    )
    else:
        #singleStageProbability = EMsingleStageProbability( EMgain, stageCount )
        return N * ( (1 - pp - stageCount*ps) * R
                     + ( pp*pCICpdf( data, EMgain )
                     + ps*sCICpdf( data, stageCount, EMprob)
                        ) * np.heaviside(data, mu)
                     )




def fitBias( data, bias=100, readnoise=5, debug=False ):
    # Fit the bias level with normal distribution.

    counts = data[:,0]
    centers = data[:,1] # bins[:-1] + np.diff(bins)/2
    centers = centers
    # Remove empty bins for Chi2 fitting
    x = centers[ counts>0 ]
    y = counts[ counts>0 ]
    sy = np.sqrt( y )

    # Do Chi2 fitting with the Applied Stats regression + iminuit
    chi2fit = Chi2Regression(gaussian, x, y, sy)
    chi2fit.errordef = 1

    N = np.trapz(counts,centers);

    # Let iminuit do the fit
    minuit = Minuit(chi2fit, N=N, mu=bias, sigma=readnoise)
    minuit.migrad()

    N, bias, ron = minuit.values[:]

    print("Bias fitted with mean BIAS={:.2f}".format(bias)+
          " and RON={:.2f}".format(ron) )

    # Produce plots if the debug mode was enabled
    if debug:
        plt.step(centers, counts, where='mid')
        plt.plot(centers, gaussian(centers, *minuit.values[:]))
        plt.yscale('log')
        plt.ylim((10, 1e3))
        plt.xlim((0,200))
        plt.show()

    # Return the fit result N, mu, sigma
    return minuit.values[:]



def fitEMBias( data, N, bias, readnoise, pp, ps, EMprob, debug=False ):

    counts = data[:,0]
    centers = data[:,1]

    # Remove empty bins for Chi2 fitting
    x = centers[ counts>0 ]
    y = counts[ counts>0 ]
    sy = np.sqrt(y)

    # Do Chi2 fitting with the Applied Stats regression + iminuit
    chi2fit = Chi2Regression(EMbiasModel, x, y, sy)
    chi2fit.errordef = 1

    # Let iminuit do the fit
    minuit = Minuit(chi2fit, N=N, mu=bias, sigma=readnoise,
                    pp=pp, ps=ps, EMprob=EMprob )
    minuit.migrad()

    # Calculate for printing.
    Ndof = len(centers) - len(minuit.values[:])
    chi2score = minuit.fval
    p = chi2.sf( chi2score, Ndof )

    s1 = "Chi2={:.2f}, ".format(chi2score)
    s2 = "Ndof=" + str(int(Ndof))
    s3 = ", p={:.2f}".format(p)

    print("EM bias fitted: "+s1+s2+s3)

    N, bias, ron, pp, ps, EMprob = minuit.values[:]

    print("Bias fitted with mean BIAS={:.2f}".format(bias)+
          " and RON={:.2f}".format(ron) )

    print("p_pCIC={:.5f}, ".format(pp) + "p_sCIC={:.5f} ".format(ps)
          + "and EMgain={:.2f}".format(calcEMgain(EMprob, 604)) )
    print("Corresponding single stage EM gain probability p_m=", EMprob)

    # Produce informative plots if the debug mode was enabled
    if debug:
        plt.step(centers, counts, where='mid')
        plt.plot(centers, EMbiasModel(centers, *minuit.values[:]))
        plt.plot(centers, EMbiasModel(centers, N, bias, ron, pp, 0, EMprob))
        plt.yscale('log')
        plt.ylim((100))
        plt.show()

    # Return the fit result N, mu, sigma
    return minuit.values[:]




if __name__ == "__main__":

    filename = sys.argv[1]

    data = np.loadtxt(filename)
    data = data[70:2000]

    data[:,1] = data[:,1] # Guess a gain

    # Stack of 250 2MB images divided into bins
    filecount = 29
    N = filecount * 1048*1024 / len(data)
    bias = 100
    ron = 5

    # Fit mean bias level to be removed from the data
    N, bias, ron = fitBias( data, bias=bias, readnoise=ron, debug=False )

    # Remove mean bias level
    data[:,1] = ( data[:,1] - bias )

    # Take some initial parameters
    pp = 0.0334
    ps = 1e-3
    EMprob = 0.01

    pp = 0.0334
    ps = 1e-3
    EMprob = 0.01

    fitEMBias(data, N, 0, ron, pp, ps, EMprob, debug=True)
