import pybdsim as _bd
import ROOT as _rt
import pickle as _pk
import glob as _gl
from collections import Counter

Decoder_dict = {0: 'Total', 11: 'e-', -11: 'e+', 22: 'photons', 12: 'neutrinos', 13: 'muons', 2112: 'neutrons', 211: 'pions'}

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

print(all_part_dict)

nb_bins = len(all_part_dict)
PART_HIST = _rt.TH1D("EndSampler_partID", "{} Particle distribution at end sampler".format(tag), nb_bins, 0, nb_bins-1)
for key in all_part_dict:
    if key in Decoder_dict:
        PART_HIST.Fill(Decoder_dict[key], all_part_dict[key])
    else:
        raise KeyError("Unknown particle of ID {}".format(key))

data = _bd.Data.Load("TEST_{}_merged_hist.root".format(tag))
merged_file = _bd.Data.CreateEmptyRebdsimFile('TEST_{}_merged_hist.root'.format(tag), data.header.nOriginalEvents)
_bd.Data.WriteROOTHistogramsToDirectory(merged_file, "Event/MergedHistograms", list(data.histograms.copy().values()))
_bd.Data.WriteROOTHistogramsToDirectory(merged_file, "Event/MergedHistograms", [PART_HIST])
merged_file.Close()
