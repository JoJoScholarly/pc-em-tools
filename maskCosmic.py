import numpy as np
from astropy.io import fits
from sys import argv



def maskCosmicEM ( image, saturationLimit=16000, r=5 ):
    """
    Mask the pixels affected by cosmic ray hits. With high EM gain CR hits saturate the severely and the CR hit will leave a trail visible in the readout direction. For each saturated pixel every pixels within radius of 5px will be disgarded and additionally the remaining row after the pixel in readout direction.

    Disgarded pixels are returned in a truth table where a boolean "True" means a disgarded pixel. Upstream in the reduction software the disgarded pixels will be handled as missed observation at a given time.
    """
    xmax, ymax = image.shape
    mask = np.zeros(image.shape)

    saturated = locateSaturated( image, saturationLimit )

    for coord in saturated:
        for apertureCoord in apertureSaturated( coord[0], coord[1], r ):
            x = coord[0]
            y = coord[1]
            if ( x<= xmax ) and ( y<=ymax ):
                mask[ x, y ] = 1

        for trailCoord in trailSaturated( coord[0], coord[1], xmax-1 ):
            x = coord[0]
            y = coord[1]
            if ( x<= xmax ) and ( y<=ymax ):
                mask[x, y] = 1
    return mask



def locateSaturated (image, saturationLimit ):
    """
    Detect saturated pixels to use as a starting point for masking.
    """
    cr = np.where( image >= saturationLimit )
    coordinateList = list(zip(*cr))

    # NB! Returns list of tuples unlike all the other functions which return
    # list of lists.
    return coordinateList



def apertureSaturated ( xcen, ycen, r ):
    """
    Returns list of pixels included in a circular aperture around the input
    pixel coordinate. NB! This may include coordinates outside the chip area!
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



def trailSaturated ( x, y, ymax ):
    """
    Return list of coordinates that are assumed to be a trail of a CR.
    """
    coordinateList = []

    yy = np.arange(y, ymax)
    for y in yy:
        coordinateList.append([x, y])
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

