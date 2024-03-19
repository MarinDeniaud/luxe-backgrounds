import pybdsim as _bd
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
import re


Decoder_dict = {0:    {'Name': 'All',                    'Symbol': 'All'},
                11:   {'Name': 'electron',               'Symbol': '$e^-$'},
                -11:  {'Name': 'positron',               'Symbol': '$e^+$'},
                12:   {'Name': 'electron_neutrino',      'Symbol': r'${\nu}_e$'},
                -12:  {'Name': 'electron_anti_neutrino', 'Symbol': r'$\overline{{\nu}_e}$'},
                13:   {'Name': 'muon',                   'Symbol': r'${\mu}^-$'},
                -13:  {'Name': 'anti_muon',              'Symbol': r'${\mu}^+$'},
                14:   {'Name': 'muon_neutrino',          'Symbol': r'${\nu}_{\mu}$'},
                -14:  {'Name': 'muon_anti_neutrino',     'Symbol': r'$\overline{{\nu}_{\mu}}$'},
                22:   {'Name': 'photon',                 'Symbol': r'$\gamma$'},
                211:  {'Name': 'pion',                   'Symbol': r'${\pi}^+$'},
                -211: {'Name': 'anti_pion',              'Symbol': r'${\pi}^-$'},
                321:  {'Name': 'kaon',                   'Symbol': '$K^+$'},
                2112: {'Name': 'neutron',                'Symbol': 'n'},
                2212: {'Name': 'proton',                 'Symbol': 'p'},
                }


Coord_dict = {'x':  {'Name': "$X$",  'Unit': '[m]'},
              'xp': {'Name': "$X'$", 'Unit': '[rad]'},
              'y':  {'Name': "$Y$",  'Unit': '[m]'},
              'yp': {'Name': "$Y'$", 'Unit': '[rad]'},
              'E':  {'Name': "$E$",  'Unit': '[GeV]'},
              'KE': {'Name': "$kE$", 'Unit': '[GeV]'}
              }


SamplerList = ['D08500', 'DUM1', 'EXT1', 'DUM2', 'EXT2', 'DUM3', 'WALL']


def analysis(inputfilename):
    root_data = _bd.Data.Load(inputfilename)
    e = root_data.GetEvent()
    et = root_data.GetEventTree()

    HIST_DICT = {}

    def storedata(data_dict, key, data, weight=1.0):
        try:
            if type(data) == float:
                data_dict[key].append(data * weight)
            elif len(data) > 1:
                for d, w in zip(data, weight):
                    data_dict[key].append(d * w)
            else:
                data_dict[key].append(data[0]*weight[0])
        except KeyError:
            data_dict[key] = []
            storedata(data_dict, key, data, weight)

    for t in et:
        for samplername in SamplerList:
            samplerdata = e.GetSampler('{}.'.format(samplername))
            if len(samplerdata.weight) != 0:
                storedata(HIST_DICT, '{}_x_All'.format(samplername), samplerdata.x, samplerdata.weight)
                storedata(HIST_DICT, '{}_xp_All'.format(samplername), samplerdata.xp, samplerdata.weight)
                storedata(HIST_DICT, '{}_y_All'.format(samplername), samplerdata.y, samplerdata.weight)
                storedata(HIST_DICT, '{}_yp_All'.format(samplername), samplerdata.yp, samplerdata.weight)
                storedata(HIST_DICT, '{}_KE_All'.format(samplername), samplerdata.kineticEnergy, samplerdata.weight)
                storedata(HIST_DICT, '{}_E_All'.format(samplername), samplerdata.energy, samplerdata.weight)
                for i, partID in enumerate(samplerdata.partID):
                    storedata(HIST_DICT, '{}_x_{}'.format(samplername, Decoder_dict[partID]['Name']), samplerdata.x[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_xp_{}'.format(samplername, Decoder_dict[partID]['Name']), samplerdata.xp[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_y_{}'.format(samplername, Decoder_dict[partID]['Name']), samplerdata.y[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_yp_{}'.format(samplername, Decoder_dict[partID]['Name']), samplerdata.yp[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_KE_{}'.format(samplername, Decoder_dict[partID]['Name']), samplerdata.kineticEnergy[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_E_{}'.format(samplername, Decoder_dict[partID]['Name']), samplerdata.energy[i], samplerdata.weight[i])

    return HIST_DICT


def getSymbolByName(name):
    for key in Decoder_dict.keys():
        if Decoder_dict[key]['Name'] == name:
            return Decoder_dict[key]['Symbol']


def plothist(data_dict, sampler, coord, nbins=50, logScale=False, figsize=[11, 7]):
    keys = data_dict.keys()
    reduced_keys = list(filter(re.compile("^{}_{}_".format(sampler, coord)).match, keys))
    bins = _np.histogram(data_dict[reduced_keys[0]], bins=nbins)[1]

    plotOptions(figsize=figsize)
    for key in reduced_keys:
        name = '_'.join(key.split('_')[2:])
        symbol = getSymbolByName(name)
        _plt.hist(data_dict[key], bins=bins, histtype='step', label=symbol)
    _plt.xlabel(Coord_dict[coord]['Name']+' '+Coord_dict[coord]['Unit'])
    _plt.ylabel('Entries')
    _plt.legend()
    if logScale:
        _plt.yscale('log')


def plotAlongS(data_dict, coord):
    keys = data_dict.keys()
    reduced_keys = list(filter(re.compile("^_{}_".format(coord)).match, keys))


def plotOptions(figsize=[9, 6], rows_colums=[1, 1], height_ratios=None, sharex=False, sharey=False, font_size=17):
    _plt.rcParams['font.size'] = font_size
    if height_ratios is not None:
        fig, ax = _plt.subplots(rows_colums[0], rows_colums[1], figsize=(figsize[0], figsize[1]),
                                gridspec_kw={'height_ratios': height_ratios}, sharex=sharex, sharey=sharey)
    else:
        fig, ax = _plt.subplots(rows_colums[0], rows_colums[1], figsize=(figsize[0], figsize[1]),
                                sharex=sharex, sharey=sharey)
    fig.tight_layout()
    return fig, ax