#!/usr/bin/env python3

import numpy as np
from astropy.io import fits
from sys import argv


def maskCosmicEM ( image, saturationLimit=16000, r=2 ):
    """Mask the pixels affected by cosmic ray hits. With high EM gain CR hits 
    saturate the severely and the CR hit will leave a trail visible in the 
    readout direction. For each saturated pixel every pixels within radius of 5px
    will be disgarded and additionally the remaining row after the pixel in
    readout direction.

    Disgarded pixels are returned in a truth table where a boolean "True" means a
    disgarded pixel. Upstream in the reduction software the disgarded pixels will
    be handled as missed observation in the given frame.

    :param image: Input image to be masked
    :type image: Arr[int]
    :param saturationLimit: Detector saturation limit, defaults to 16000
    :type saturationLimit: int, optional
    :param r: Rejection radius around a cosmic ray hit, defaults to 2
    :type r: int, optional
    :return: Returns a mask with the same dimensions as the input image
    :rtype: Arr[bool]
    """    
    xmax, ymax = image.shape
    mask = np.zeros(image.shape)

    saturated = locateSaturated( image, saturationLimit )

    for coord in saturated:
        for apertureCoord in apertureSaturated( coord[0], coord[1], r ):
            x = apertureCoord[0]
            y = apertureCoord[1]
            if ( x< xmax ) and ( y<ymax ):
                mask[ x, y ] = 1

        for trailCoord in trailSaturated( coord[0], coord[1], ymax ):
            x = trailCoord[0]
            y = trailCoord[1]
            if ( x<xmax ) and ( y<ymax ):
                mask[x, y] = 1
    return mask



def locateSaturated (image, saturationLimit ):
    """Detect saturated pixels above the saturation limit, to be used as a 
    starting point for masking.

    :param image: Image to be masked
    :type image: Arr[int]
    :param saturationLimit: Detector saturation limit, used as a threshold.
    :type saturationLimit: int
    :return: List of tuples indicating coordinates of saturated pixels.
    :rtype: List[float]
    """    
    cr = np.where( image >= saturationLimit )
    coordinateList = list(zip(*cr))

    # NB! Returns list of tuples unlike all the other functions which return
    # list of lists.
    return coordinateList



def apertureSaturated ( xcen, ycen, r ):
    """Returns list of pixels included in a circular aperture around the input
    pixel coordinate. NB! This may include coordinates outside the chip area!

    :param xcen: Aperture center x-axis
    :type xcen: float
    :param ycen: Aperture center y-axis
    :type ycen: float
    :param r: Radius to be included
    :type r: float
    :return: List of pixel coordinates included in the aperture
    :rtype: List[float]
    """    
    r = round(r)
    coordinateList = []
    # Max size grid to be considered
    xmin = xcen-r
    xmax = xcen+r
    ymin = ycen-r
    ymax = ycen+r

    xx = np.arange(xmin, xmax+1)
    yy = np.arange(ymin, ymax+1)

    for x in xx:
        for y in yy:
            if (x-xcen)**2+(y-ycen)**2 <= r:
                coordinateList.append([x, y])
    return coordinateList



def trailSaturated ( x0, y0, ymax ):
    """Returns a list of coordinates included in a CR (CTI?) trail.

    :param x0: Starting x-coordinate
    :type x0: int
    :param y0: Starting y-coordinate
    :type y0: int
    :param ymax: Detector width? TODO
    :type ymax: int
    :return: List of coordinates affected by a cosmic ray hit.
    :rtype: List[float]
    """    
    coordinateList = []

    yy = np.arange(0, ymax)

    for y in yy:
        # Add the entire next row after CR hit, bit excessive, imporove by model
        coordinateList.append([x0+1, y])
        # Add all on the same row after CR hit
        if y >= y0:
            coordinateList.append([x0,y])

    return coordinateList



if __name__ == "__main__":
    filename = argv[1]
    filenamebase = filename.split(".")[0]
    exten = 0

    with fits.open(filename) as hdul:
        data = hdul[0].data

    mask = maskCosmicEM(data)
    outfilename = filenamebase + "_cr_mask.fits"
    hdu = fits.PrimaryHDU( mask )
    hdu.writeto( outfilename, overwrite=True)

