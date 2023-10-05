# Photon Counting Tools

## Description
This is a collection of scripts that can be used as building blocks for statistical photon counting with Electron Multiplication CCDs (EMCCDs).

Release 0.1b allows changing detector parameters with a configuration file and statistical photon counting by thresholding. Expected value and variance for thresholding are calculated as described in Harpsoe et al. (2012) ["Bayesian Photon Counting with Electron-Multiplying Charge-Coupled Devices"](https://doi.org/10.1051/0004-6361/201117089).

In case of long EMCCD exposures and doing dead time corrections (counting no-detections), cosmic ray hits will affect the final thresholded flux estimate. In some cases, such as faint object spectroscopy, EMCCDs has to be operated at low temperature to control dark current which makes its EMCCD Charge Transfer Efficiency (CTE) worse. This may lead to cosmic ray (CR) hits leaving trails and bringing entire columns above the threshold level. The exposure times might also be longer, from tens of seconds to minutes, making CR hits more likely.

Dealing with CRs is very simplistic at the moment. A CR is consider to be any pixel value higher than 'crlimit' in the configuration file. A number of pixels around the value above limit and entire row after the CR hit is disgarded by the code. Effectively, this translates into degrading detective quantum efficiency in the pixels further away from readout amplifier. Code outputs a CR effieciency image reflecting on how large fraction of frames has been used for thresholding.

In order to make use of the code, one needs a series of bias frames read out with the same settings as will be used for counting the photons, and the actual photon counting data. Bias and science frames are supposed to be placed in separated directories and the code will search (by default recursively) the directory for FITS files which it will read in. Firstly, the Detector Clock-Induced Charge, readout noise, and gain are deduced from the set of bias frames. Respective values are updated in the configuration and are used for thresholding.

Final output of 'threshold' is three FITS images, expected value, variance, and fraction of accepted frames.


## Pre-requisites
Numpy, astropy, scipy, iMinuit, matplotlib, configparser, sys, glob


## Usage
The main routine is called `threshold` which goes through a directory of 2D fits images, and based on the found pixel values above a given threshold level, it produces a 2D photon rate image. `threshold` makes use of `maskCosmics` which rejects saturated pixels and an area around them. Additionally, it will mask the remaining line due to possible EM register afterglow. Some times this effect can be observed more than the rest of the line and it will be better to reject the following line in readout direction as well. The EM register afterglow after a cosmic ray hit could be analyzed and characterized better to make this routine more sophisticated.

The remaining programs are helpers to get the thresholding setup optimized. The intended use it to record a large set of bias frames with the same EM setup as is used for the scientific data. More than 100 bias frames is needed to sample the high value tail in the bias histogram to fit for CIC. `binTheImages` is run on the directory containg the bias file sequence and it store the number counts in bins in a txt file. Then `fitBackground` can be run on the txt file and it will output probabilities for serial and parallel CIC, bias level, readout noise and single register element EM gain. `threshold` is dependent on these values.

Finally, `findThreshold` can be used for finding the threshold level giving mininum false negative and false positive rate. Only readout noise and gain are considered making it useful for low gains only. At the moment, sCIC is not considered even though it is highly relevant with higher gains. 

## Authors and acknowledgment
(c) Joonas Viuho 2023

## License
CC-BY-SA-4.0 license.

## Project status
The code is developed for a personal project. It may still contain significant bugs and it is not optimized for performance.
