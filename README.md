# Photon Counting Tools

## Description
This is a collection of scripts that can be used as building blocks for the NTE UV arm spectrograph and Slit Viewer photon counting routines. The initial commit is a dump of a personal project and needs to be refined and generalized in order to be usable in NTE case. The code is based on Harpsoe et al. (2012) "Bayesian Photon Counting with Electron-Multiplying Charge-Coupled Devices". The code is guaranteed to contain bugs in its current state. 

## Pre-requisites
Numpy, multi-threading

## Installation
Install pre-requisited, clone the repo and run.

## Usage
The main routine is called `threshold` which goes through a directory of 2D fits images, and based on the found pixel values above a given threshold level, it produces a 2D photon rate image. `threshold` makes use of `maskCosmics` which rejects saturated pixels and an area around them. Additionally, it will mask the remaining line due to possible EM register afterglow. Some times this effect can be observed more than the rest of the line and it will be better to reject the following line in readout direction as well. The EM register afterglow after a cosmic ray hit could be analyzed and characterized better to make this routine more sophisticated.

The remaining programs are helpers to get the thresholding setup optimized. The intended use it to record a large set of bias frames with the same EM setup as is used for the scientific data. More than 100 bias frames is needed to sample the high value tail in the bias histogram to fit for CIC. `binTheImages` is run on the directory containg the bias file sequence and it store the number counts in bins in a txt file. Then `fitBackground` can be run on the txt file and it will output probabilities for serial and parallel CIC, bias level, readout noise and single register element EM gain. `threshold` is dependent on these values.

Finally, `findThreshold` can be used for finding the threshold level giving mininum false negative and false positive rate. However, it might be better to use 5-sigma limit for the readnoise as a threshold criteria. The penalty of having too high threshold level is minimal.

## Authors and acknowledgment
(c) Joonas Viuho 2022

## License
For open source projects, say how it is licensed.

## Project status
An initial commit has been made based on code developed for a personal project. Needs to be made more general, more user-friendly and documented better.
