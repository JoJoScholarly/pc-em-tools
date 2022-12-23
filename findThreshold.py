#!/usr/bin/env python3

"""
Utility function to help finding optimum threshold level.

Parameters
----------
thresh : float
    The threshold level to use.
ron : float
    Detector readout noise.
EMprob : float
    Single EM stage avalanche multiplication probablility.
"""

import numpy as np
from sys import argv
from math import erf
from fitBackground import calcEMgain

thresh = int(argv[1])

ron = 4.7
EMprob = 0.009576050704909498
EMgain = calcEMgain(EMprob, 604)


# Probability that RON raises above threshold (false positive)
p_fp = 0.5 - 0.5*erf(thresh/ron/2**0.5)

# Probabilty of not to get EM amplified (false negative)
p_fn = 1 - np.e**(-thresh/EMgain)

print("False positive rate: {:.1f}".format(p_fp*100)+"%")
print("False negative rate: {:.1f}".format(p_fn*100)+"%")
