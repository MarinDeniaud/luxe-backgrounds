#!/usr/bin/env python3

import pybdsim as _bd
import ROOT as _rt
import pickle as _pk
import glob as _gl
from collections import Counter

Decoder_dict = {0:    'Total',
                11:   '$e^-$', 
                -11:  '$e^+$', 
                12:   r'${\nu}_e$', 
                13:   '${\mu}^-$',
                -13:  '${\mu}^+$',
                14:   r'${\nu}_{\mu}$',
                -14:  r'$\overline{{\nu}_{\mu}}$',
                22:   '$\gamma$',
                211:  '${\pi}^+$',
                -211: '${\pi}^-$',
                321:  '$K^+$',
                2112: 'n', 
                2212: 'p'
               }

tag = 'T20_dens_1e-12'

# _bd.Run.RebdsimHistoMerge("{}_merged_hist.root".format(tag), "*{}_hist.root".format(tag), rebdsimHistoExecutable='rebdsimCombine')

picklefiles = _gl.glob('*{}*.pk'.format(tag))

print("Tag : ", tag, "/ {} files found".format(len(picklefiles)))

all_part_dict = {}

for pickle in picklefiles:
    file = open(pickle, 'rb')
    part_dict = _pk.load(file)
    file.close()
    all_part_dict = dict(Counter(all_part_dict)+Counter(part_dict))

for key in all_part_dict:
    all_part_dict[key] = all_part_dict[key]/len(picklefiles)

nb_bins = len(all_part_dict)
PART_HIST = _rt.TH1D("EndSampler_partID", "{} Particle distribution at end sampler".format(tag), nb_bins, 0, nb_bins-1)
for key in all_part_dict:
    if key in Decoder_dict:
        PART_HIST.Fill(Decoder_dict[key], all_part_dict[key])
    else:
        raise KeyError("Unknown particle of ID {}".format(key))

data = _bd.Data.Load("{}_merged_hist.root".format(tag))
merged_file = _bd.Data.CreateEmptyRebdsimFile('{}_merged_hist.root'.format(tag), data.header.nOriginalEvents)
_bd.Data.WriteROOTHistogramsToDirectory(merged_file, "Event/MergedHistograms", list(data.histograms.copy().values()))
_bd.Data.WriteROOTHistogramsToDirectory(merged_file, "Event/MergedHistograms", [PART_HIST])
merged_file.Close()
