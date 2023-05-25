#!/usr/bin/env python3

import beamWire
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input",      help="input file for bdsim",            dest="input",       action="store", type="string")
    parser.add_option("-p", "--npart",      help="number of initial particles",     dest="npart",       action="store", type="int",     default=100)
    parser.add_option("-d", "--diameter",   help="wire diameter",                   dest="diameter",    action="store", type="int",     default=0.5)
    parser.add_option("-x", "--offsetX",    help="wire offset",                     dest="offsetX",     action="store", type="int",     default=0)
    parser.add_option("-s", "--seed",       help="seed for bdsim simulation",       dest="seed",        action="store", type="int",     default=None)
    parser.add_option("-o", "--output",     help="bdsim output for analysis",       dest="output",      action="store", type="string")
    parser.add_option("-n", "--nbins",      help="number of bins for histograms",   dest="nbins",       action="store", type="int",     default=50)
    (options, args) = parser.parse_args()

    beamWire.runOneOffset(options.__dict__['input'], outputfilename=options.__dict__['output'], npart=options.__dict__['npart'],
                          diameter=options.__dict__['diameter'], offsetX=options.__dict__['offsetX'], seed=options.__dict__['seed'])
    beamWire.analysis(options.__dict__['output']+".root", nbins=options.__dict__['nbins'])
