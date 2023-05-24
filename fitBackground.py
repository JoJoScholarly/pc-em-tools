#!/usr/bin/env python3

"""Module to fit EMCCD bias frames and deduce detector characteristics.
Overall organization is to take in a histogram of a series of bias frames, fit
the histogram with PDF for EM output varying EM gain, CIC and readnoise. PDFs
are according to Harpsoe et al. (2012) paper Bayesian Photon Counting with EMCCDs.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from iminuit import Minuit
from scipy.stats import norm, poisson, chi2
from scipy.special import gamma
import configparser
import os
import sys


from iminuit.util import make_func_code
from iminuit import describe #, Minuit,

# TODO best way?
global stageCount = int(config['detector']['stagecount'])

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    

def compute_f(f, x, *par):
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


class Chi2Regression:  # override the class with a better one
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

    def __call__(self, *par):  # par are a variable number of model parameters
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2



def gaussian( x, N, mu, sigma ):
    """Returns a normal distribution PDF for the given value range.

    :param x: _description_
    :type x: _type_
    :param N: Normalization factor.
    :type N: float
    :param mu: Mean of normal the distribution.
    :type mu: float
    :param sigma: Standard deviation of the normal distribution.
    :type sigma: float
    :return: _description_
    :rtype: _type_
    """
    return N * norm.pdf(x, mu, sigma)



def EMsingleStageProbability( EMgain, stageCount ):
    """Given EM gain and EM amplifier stage count, returns single stage
    probability for EM amplification.

    :param EMgain: _description_
    :type EMgain: float
    :param stageCount: _description_
    :type stageCount: int
    :return: _description_
    :rtype: float
    """
    return EMgain**(1/stageCount) - 1



def calcEMgain( singleStageProbability, stageCount ):
    """Calculates EM gain given single stage electron avalanche probability
    and total stage count.

    :param singleStageProbability: Probability of cascade amplification event in
        a single register stage.
    :type singleStageProbability: float
    :param stageCount: Number of stages in EM register.
    :type stageCount: int
    :return: Returns the total system EM gain.
    :rtype: float
    """    
    return ( 1 + singleStageProbability )**stageCount



def pCICpdf( x, EMgain ):
    """Probability density function of EM output parallel CIC events.

    :param x: _description_
    :type x: _type_
    :param EMgain: _description_
    :type EMgain: _type_
    :return: _description_
    :rtype: _type_
    """    
    return np.exp(-x/EMgain) / EMgain * np.heaviside(x, 0)



def sCICpdf( x, stageCount, singleStageProbability ):
    """Probability density function of EM output serial CIC events.

    :param x: _description_
    :type x: _type_
    :param stageCount: _description_
    :type stageCount: _type_
    :param singleStageProbability: _description_
    :type singleStageProbability: _type_
    :return: _description_
    :rtype: _type_
    """    
    sum = 0
    for i in np.arange(1, stageCount+1):
        gain = calcEMgain( singleStageProbability, i)
        sum += np.exp( -x/gain ) / gain * np.heaviside(x, 0)
    return sum



def EMbiasModel( data, N, mu, sigma, pp, ps, p_EM ):
    """Uses Harpsoe et al. 2012 paper eq. 17 to model the EM output. Get the
    probability of parallel and serial CIC event as an output.

    :param data: Input histogram values, steps of full ADUs or electronics expected
    :type data: List[int]
    :param N: Normal distribution PDF normalization factor.
    :type N: float
    :param mu: Normal distribution mean, i.e. bias level.
    :type mu: float
    :param sigma: Normal distribution standard deviation, i.e. readnoise.
    :type sigma: float
    :param pp: Probability for parallel CIC.
    :type pp: float
    :param ps: Probabilty for serial CIC.
    :type ps: float
    :param p_EM: EM register single stage probability for impact ionisation.
    :type p_EM: float
    :param stageCount: Number of EM register stages.
    :type stageCount: int
    :param ignoreSerialCIC: Flag to omit serial CIC.
    :type ignoreSerialCIC: boolean, optional
    :return: Calculated EM output.
    :rtype: List[float]
    """    
    ignoreSerialCIC=False    
    EMgain = calcEMgain( p_EM, stageCount )

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
                     + ps*sCICpdf( data, stageCount, p_EM)
                        ) * np.heaviside(data, mu)
                     )



def fitBias( data, bias, readnoise, plotFig=False ):
    """Fit bias level with a normal distribution

    :param data: Input data in histogram format, bins starting from 1 to len(data)
    :type data: List[float]
    :param bias: Bias level
    :type bias: int, optional
    :param readnoise: Readout noise
    :type readnoise: float, optional
    :param plotFig: Flag for additional plotting, defaults to False
    :type plotFig: bool, optional
    :return: Returns fitting parameters for a normal distribution representing the bias level.
    :rtype: List[float]
    """    
    counts = data[:,0]
    centers = data[:,1] # bins[:-1] + np.diff(bins)/2
  
    # Remove empty bins for Chi2 fitting
    x = centers[ counts>0 ]
    y = counts[ counts>0 ]
    sy = np.sqrt( y )

    # Do Chi2 fitting with the Applied Stats regression + iminuit
    chi2fit = Chi2Regression(gaussian, x, y, sy)
    chi2fit.errordef = 1

    N = np.trapz(counts,centers)

    # Let iminuit do the fit
    minuit = Minuit(chi2fit, N=N, mu=bias, sigma=readnoise)
    minuit.migrad()

    N, bias, ron = minuit.values[:]

    print("Bias fitted with mean BIAS={:.2f}".format(bias)+
          " and RON={:.2f}".format(ron) )

    # Produce plots if the plotFig mode was enabled
    if plotFig:
        plt.step(centers, counts, where='mid')
        plt.plot(centers, gaussian(centers, *minuit.values[:]))
        plt.yscale('log')
        plt.ylim((10, 1e3))
        plt.xlim((0,200))
        plt.show()

    # Return the fit result N, mu, sigma
    return minuit.values[:]



def fitEMBias( data, N, bias, readnoise, pp, ps, p_EM, stageCount, plotFig=False ):
    """Fit full EM output model including readnoise, parallel+serial CIC.

    :param data: Input histogram data
    :type data: List[int]
    :param N: Normal distribution normalization.
    :type N: float
    :param bias: Bias level
    :type bias: float
    :param readnoise: Readout noise./
    :type readnoise: float
    :param pp: Probability for a parallel CIC event.
    :type pp: float
    :param ps: Probability for a serial CIC event.
    :type ps: float
    :param p_EM: Single register stage EM gain probability.
    :type p_EM: float
    :param stageCount: Number of EM register stages.
    :type stageCount: int
    :param plotFig: Flag to produce additional graphical output, defaults to False
    :type plotFig: bool, optional
    :return: Returns fitted parameters.
    :rtype: List[float]
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
                    pp=pp, ps=ps, p_EM=p_EM )
    minuit.migrad()

    # Iterate and correct mistakes in initial bias fit
    N, bias, ron, pp, ps, p_EM = minuit.values[:]
    centers = centers+bias
    minuit = Minuit(chi2fit, N=N, mu=0, sigma=readnoise,
                    pp=pp, ps=ps, p_EM=p_EM )
    minuit.migrad()


    # Calculate for printing.
    Ndof = len(centers) - len(minuit.values[:])
    chi2score = minuit.fval
    p = chi2.sf( chi2score, Ndof )

    s1 = "Chi2={:.2f}, ".format(chi2score)
    s2 = "Ndof=" + str(int(Ndof))
    s3 = ", p={:.2f}".format(p)

    print("EM bias fitted: "+s1+s2+s3)

    N, bias, ron, pp, ps, p_EM = minuit.values[:]

    print("Bias fitted with mean BIAS={:.2f}".format(bias)+
          " and RON={:.2f}".format(ron) )

    print("p_pCIC={:.9f}, ".format(pp) + "p_sCIC={:.9f} ".format(ps)
          + "and EMgain={:.2f}".format(calcEMgain(p_EM, stageCount)) )
    print("Corresponding single stage EM gain probability p_m=", p_EM)

    # Produce informative plots if the mode was enabled
    if plotFig:
        # Main figure
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.subplots(nrows=1, ncols=1, sharex=False, sharey=False,
                           gridspec_kw={'height_ratios':[1]})

        ax1.errorbar(centers, counts, np.sqrt(counts),
                     drawstyle='steps-mid', capsize=2, label='Data', alpha=0.5)
        ax1.plot(centers, EMbiasModel(centers, *minuit.values[:]),
                 label='Model')
        ax1.plot(centers, EMbiasModel(centers, N, bias, ron, pp, 0, p_EM),
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

    # Read config, TODO move to main? TODO if moved stageCount needs passed as argument
    configfilename = 'pc-tools.cfg'
    config = configparser.ConfigParser()
    config.read( configfilename )

    filepath = sys.argv[1]
    filename = config['files']['histName']
    data = np.loadtxt( filepath + filename )
    
    # Clip values making the fits worse
    # TODO necessary or better way to do? Crop overscans out?
    data = data[80:3000] 
    
    # TODO Pre-amp gain here? Electron conversion needed or not?
    #data[:,1] = data[:,1]

    # Stack of 250 2MB images divided into bins
    filecount = int(config['files']['filecount'])
    xsize = int(config['detector']['xsize'])
    ysize = int(config['detector']['ysize'])
    bias = float(config['detector']['biasLevel'])
    ron = float(config['detector']['readnoise'])

    # Calculate reasonable starting point for normalization
    N = filecount * xsize*ysize / data.shape[0]

    # Find the mean bias level to be removed from the data
    N, bias, ron = fitBias( data, bias=bias, readnoise=ron, plotFig=False )
    
    # Remove mean bias level
    data[:,1] = ( data[:,1] - bias )

    # Get initial values from config
    pp = float(config['detector']['p_pCIC'])
    ps = float(config['detector']['p_sCIC'])
    p_EM = float(config['detector']['p_EM'])
    stageCount = int(config['detector']['stagecount'])
    
    # Update bias in config file befire the EM fit, going to be set to 0
    config['detector']['biaslevel'] = str(bias)

    N, bias, ron, pp, ps, p_EM = fitEMBias(data, N, 0, ron, pp, ps, p_EM, stageCount,
                                             plotFig=True)

    # Update config with new values
    config['detector']['readnoise'] = str(ron)
    config['detector']['p_pcic'] = str(pp)
    config['detector']['p_scic'] = str(ps)
    config['detector']['p_em'] = str(p_EM)
    config['detector']['systemgain'] = str(calcEMgain(p_EM, stageCount))

    with open( configfilename, 'w') as configfile:
        config.write( configfile )
