#!/usr/bin/env python3
import numpy as np
from math import erf
import configparser
from fitBackground import calcEMgain


def falsePositive( thresh, ron ):
    """Probability that EON raises above threshold.

    Harpsoe et al. (2012) Eq. 20

    :param thresh: treshold level
    :type thresh: int
    :param EMgain: EM gain
    :type EMgain: float
    :return: Probability that RON raises above threshold.
    :rtype: float
    """    
    return 0.5 - 0.5*erf(thresh/ron/2**0.5)


def falseNegative( thresh, EMgain ):
    """Probability that signal does not get amplified over threshold level.

    Harpsoe et al. (2012) Eq. 23

    :param thresh: treshold level
    :type thresh: int
    :param EMgain: EM gain
    :type EMgain: float
    :return: Probability that signal does not raises above threshold.
    :rtype: float
    """
    return 1 - np.e**(-thresh/EMgain)


if __name__ == "__main__":
    import configparser

    configfilename = "pc-tools.cfg"
    config = configparser.ConfigParser()
    config.read( configfilename )

    thresh = float(config['pc']['threshold'])
    ron = float(config['detector']['readnoise'])
    p_EM = float(config['detector']['p_em'])
    stageCount = float(config['detector']['stagecount'])

    EMgain = calcEMgain(p_EM, stageCount)

    p_fp = falsePositive( thresh, ron )
    p_fn = falseNegative( thresh, EMgain )

    print("Probability false positive: {:.2f}".format(p_fp*100)+"%")
    print("Probabiilty false negative: {:.2f}".format(p_fn*100)+"%")
