#!/usr/bin/env python3

import beamGas
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", help="input file for analysis", dest = "input", action = "store", type = "string")
    parser.add_option("-n", "--nbins", help="number of bins for histograms", dest = "nbins", action = "store", type = "int", default=50)
    (options, args) = parser.parse_args()

    beamGas.analysis(options.__dict__['input'], options.__dict__['nbins'])
