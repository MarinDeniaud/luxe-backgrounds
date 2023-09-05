#!/usr/bin/env python3

import beamNeedle
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input",      help="input file for bdsim",            dest="input",       action="store", type="string")
    parser.add_option("-p", "--npart",      help="number of initial particles",     dest="npart",       action="store", type="int",     default=100)
    parser.add_option("-s", "--seed",       help="seed for bdsim simulation",       dest="seed",        action="store", type="int",     default=None)
    parser.add_option("-o", "--output",     help="bdsim output for analysis",       dest="output",      action="store", type="string")
    parser.add_option("-n", "--nbins",      help="number of bins for histograms",   dest="nbins",       action="store", type="int",     default=50)
    (options, args) = parser.parse_args()

    print("Start BDSIM")
    beamNeedle.runOneOffset(options.__dict__['input'], outputfilename=options.__dict__['output'],
                            npart=options.__dict__['npart'], seed=options.__dict__['seed'],)
    print("End BDSIM / Start Analysis")
    beamNeedle.analysis(options.__dict__['output']+".root", nbins=options.__dict__['nbins'])
    print("End Analysis")
