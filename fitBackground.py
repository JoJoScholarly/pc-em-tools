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

class Chi2Regression:
    # override the class with a better one
    # Author: Christian Michelsen, NBI, 2018    
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])


configFile = "config.ini"
config = configparser.ConfigParser()
config.read( configFile )

"""
- Take in a stack of bias frames and merge
- Fit to get the profile for read noise
- Take the observed bias and 

"""

def gaussian( x, N, mu, sigma ):
    """
    Returns a normal distribution for given x-axis.
    """
    return N * norm.pdf(x, mu, sigma)


def EMsingleStageProbability( EMgain, stageCount ):
    """
    Given EM gain and EM amplifier stage count, returns single stage
    probability for EM amplification.
    """
    return EMgain**(1/stageCount) - 1


def calcEMgain( singleStageProbability, stageCount ):
    """
    Calculates EM gain given single stage electron avalanche probability
    and total stage count.
    """
    return ( 1 + singleStageProbability )**stageCount


def pCICpdf( x, EMgain ):
    """
    Probability density function of EM output parallel CIC events.
    """
    return np.exp(-x/EMgain) / EMgain * np.heaviside(x, 0)


def sCICpdf( x, stageCount, singleStageProbability ):
    """
    Probability density function of EM output serial CIC events.
    """
    sum = 0
    for i in np.arange(1, stageCount+1):
        gain = calcEMgain( singleStageProbability, i)
        sum += np.exp( -x/gain ) / gain * np.heaviside(x, 0)
    return sum



def stackBias( filepath, exten=0 ):
    """
    Read in fits files in a directory and median stack. Default fits extension 0.
    """
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
    """
    Uses Harpsoe et al. 2012 paper eq. 17 to model the EM output. Get the
    probability of parallel and serial CIC event as an output.
    """

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
    """
    Fit the bias level with normal distribution.
    """

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
    """
    Fit full model.
    """
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

    # Iterate and correct mistakes in initial bias fit
    N, bias, ron, pp, ps, EMprob = minuit.values[:]
    centers = centers+bias
    minuit = Minuit(chi2fit, N=N, mu=0, sigma=readnoise,
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

    print("p_pCIC={:.9f}, ".format(pp) + "p_sCIC={:.9f} ".format(ps)
          + "and EMgain={:.2f}".format(calcEMgain(EMprob, 604)) )
    print("Corresponding single stage EM gain probability p_m=", EMprob)

    # Produce informative plots if the debug mode was enabled
    if debug:
        # Main figure
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.subplots(nrows=1, ncols=1, sharex=False, sharey=False,
                           gridspec_kw={'height_ratios':[1]})

        ax1.errorbar(centers, counts, np.sqrt(counts),
                     drawstyle='steps-mid', capsize=2, label='Data', alpha=0.5)
        ax1.plot(centers, EMbiasModel(centers, *minuit.values[:]),
                 label='Model')
        ax1.plot(centers, EMbiasModel(centers, N, bias, ron, pp, 0, EMprob),
                 label='Model, no sCIC')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.set_ylim((1))
        ax1.set_yticks([100, 1e4, 1e6, 1e8])
        ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                        labelbottom=False)

        # Residual plot
        ax2 = fig.add_axes([0.125, 0.1, 0.775, 0.1])
        ax2.errorbar(centers,
                     (counts-EMbiasModel(centers, *minuit.values[:])), np.sqrt(counts),
                     drawstyle='steps-mid', alpha=0.5)
        ax2.set_xlabel('Counts (ADU)')
        #ax2.set_yscale('log')
        ax2.set_ylim((-200,200))

        # Residual distribution
        #ax3 = fig.add_axes([0.615, 0.28, 0.25, 0.25])

        plt.show()

    # Return the fit result N, mu, sigma
    return minuit.values[:]




if __name__ == "__main__":

    filename = sys.argv[1]

    data = np.loadtxt(filename)
    data = data[80:3000]

    data[:,1] = data[:,1] # Guess a gain

    # Stack of 250 2MB images divided into bins
    filecount = 175
    N = filecount * 1048*1024 / len(data)
    bias = 100
    ron = 5.35

    # Fit mean bias level to be removed from the data
    N, bias, ron = fitBias( data, bias=bias, readnoise=ron, debug=False )

    # Remove mean bias level
    data[:,1] = ( data[:,1] - bias )

    # Take some initial parameters
    pp = 0.012
    ps = 2e-5
    EMprob = 0.01

    fitEMBias(data, N, 0, ron, pp, ps, EMprob, debug=True)
