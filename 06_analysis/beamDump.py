import pybdsim as _bd
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
import re
import G4Dict

Particle_type_dict = G4Dict.Particle_type_dict
Process_type_dict = G4Dict.Process_type_subtype_dict

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
                    storedata(HIST_DICT, '{}_x_{}'.format(samplername, Particle_type_dict[partID]['Name']), samplerdata.x[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_xp_{}'.format(samplername, Particle_type_dict[partID]['Name']), samplerdata.xp[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_y_{}'.format(samplername, Particle_type_dict[partID]['Name']), samplerdata.y[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_yp_{}'.format(samplername, Particle_type_dict[partID]['Name']), samplerdata.yp[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_KE_{}'.format(samplername, Particle_type_dict[partID]['Name']), samplerdata.kineticEnergy[i], samplerdata.weight[i])
                    storedata(HIST_DICT, '{}_E_{}'.format(samplername, Particle_type_dict[partID]['Name']), samplerdata.energy[i], samplerdata.weight[i])

    return HIST_DICT


def analysisHIST(inputfilename, nbins=50):
    root_data = _bd.Data.Load(inputfilename)
    e = root_data.GetEvent()
    et = root_data.GetEventTree()

    HIST_DICT = {}
    HIST_DICT['ELECTRONS_E_det'] = _rt.TH1D('ELECTRONS_E_det', "{} Electrons wrt energy at detector".format(tag), nbins, 0, 14)
    HIST_DICT['ELECTRONS_X_Y_det'] = _rt.TH2D('ELECTRONS_X_Y_det', "{} Electrons X-Y at detector".format(tag), nbins, 0.1, 0.6, nbins, -0.25, 0.25)
    HIST_DICT['PHOTONS_E_Theta_log'] = _rt.TH2D('PHOTONS_E_Theta_log', r"{} Photons E-$\theta$ at sampler".format(tag),
                                                nbins, _np.logspace(-4, 2, nbins + 1),
                                                nbins, _np.logspace(-6, -2, nbins + 1))


def getSymbolByName(name):
    for key in Particle_type_dict.keys():
        if Particle_type_dict[key]['Name'] == name:
            return Particle_type_dict[key]['Symbol']


def getParticleTypes(inputfilename):
    root_data = _bd.Data.Load(inputfilename)
    et = root_data.GetEventTree()
    partIDset = set()
    for i, evt in enumerate(et):
        traj_data = _bd.Data.TrajectoryData(root_data, i)
        for traj in traj_data.trajectories:
            partIDset.add(traj['partID'])
    return partIDset


class TrajData:
    def __init__(self, inputfilename):
        self.inputfilename = inputfilename
        self.bdsim_data = _bd.Data.Load(inputfilename)

    def printTrajLog(self, evtnb):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        print(f'{"partID":10s}{"min Z":10s}{"max Z":10s}{"trackID":10s}{"parentID":10s}{"parentIDX":10s}{"parentStepIDX":15s}{"primaryStepIDX":15s}')
        for traj in traj_data.trajectories:
            line = [traj['partID'], round(traj['Z'].min(), 3), round(traj['Z'].max(), 3), traj['trackID'], traj['parentID'],
                    traj['parentIDX'], traj['parentStepIDX'], traj['primaryStepIDX']]
            print(f'{str(line[0]):10s}{str(line[1]):10s}{str(line[2]):10s}{str(line[3]):10s}{str(line[4]):10s}'
                  f'{str(line[5]):10s}{str(line[6]):15s}{str(line[7]):15s}')

    def getEndOfChainTrackID(self, evtnb):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        trackIDlist = []
        parentIDlist = []
        for traj in traj_data.trajectories:
            trackIDlist.append(traj['trackID'])
            parentIDlist.append(traj['parentID'])
        return _np.setdiff1d(trackIDlist, parentIDlist)

    def getEndOfChainIndex(self, evtnb):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        trackIDs = self.getEndOfChainTrackID(evtnb)
        indexList = []
        for trackID in trackIDs:
            for index, traj in enumerate(traj_data):
                if trackID == traj['trackID']:
                    indexList.append(index)
        return indexList

    def getTrackNumberWithID(self, evtnb, trackID):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        for tracknb, traj in enumerate(traj_data.trajectories):
            if traj['trackID'] == trackID:
                return tracknb

    def getPocessTypeSubtype(self, evtnb, index):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        traj = traj_data.trajectories[index]
        preSet = []
        postSet = []
        for prePT, prePST, postPT, postPST in zip(traj['prePT'], traj['prePST'], traj['postPT'], traj['postPST']):
            preSet.append((prePT, prePST))
            postSet.append((postPT, postPST))
        return preSet, postSet

    def getProcessChain(self, evtnb, index):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        traj = traj_data.trajectories[index]
        processList = []
        processNameList = []
        while traj['parentID'] != 0:
            processList.append((traj['prePST'][0], traj['prePT'][0]))
            processNameList.append(())
            traj = traj_data.trajectories[traj['parentIDX']]

        return _np.flip(processList)

    def printTrackDataByTrackID(self, evtnb, trackID):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        tracknb = self.getTrackNumberWithID(evtnb, trackID)
        traj = traj_data[tracknb]
        print('Particle : {}'.format(Particle_type_dict[traj['partID']]['Name']))
        preSet, postSet = self.getPocessTypeSubtype(evtnb, tracknb)
        print('preSet : ', preSet)
        print('postSet : ', postSet)

    def printTrackDataByIndex(self, evtnb, index):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        traj = traj_data[index]
        print('Particle : {}'.format(Particle_type_dict[traj['partID']]['Name']))
        preSet, postSet = self.getPocessTypeSubtype(evtnb, index)
        print('preSet : ', preSet)
        print('postSet : ', postSet)


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