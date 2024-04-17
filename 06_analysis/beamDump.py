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
              'KE': {'Name': "$kE$", 'Unit': '[GeV]'}}


def analysis(inputfilename, nbins=50, ELECTRONS_PER_BUNCH=2e9):
    root_data = _bd.Data.Load(inputfilename)
    e = root_data.GetEvent()
    et = root_data.GetEventTree()
    npart = et.GetEntries()

    HIST_DICT = {}
    HIST_SIZES_1D = {'x': {'Xmin': -2.5, 'Xmax': 2.5},
                     'y': {'Xmin': -2.5, 'Xmax': 2.5},
                     'xp': {'Xmin': -1, 'Xmax': 1},
                     'yp': {'Xmin': -1, 'Xmax': 1}}
    HIST_SIZES_1D_LOG = {'energy': {'XminExp': -4, 'XmaxExp': 1},
                         'kineticEnergy': {'XminExp': -4, 'XmaxExp': 1}}
    HIST_SIZES_2D = {'x_y': {'Xcoord': 'x', 'Ycoord': 'y', 'Xmin': -2.5, 'Xmax': 2.5, 'Ymin': -2.5, 'Ymax': 2.5},
                     'xp_yp': {'Xcoord': 'xp', 'Ycoord': 'yp', 'Xmin': -1, 'Xmax': 1, 'Ymin': -1, 'Ymax': 1}}

    def fill1D(data_dict, key, data, weight):
        if type(data) == float:
            data_dict[key].Fill(data, weight)
        elif len(data) > 1:
            for d, w in zip(data, weight):
                data_dict[key].Fill(d, w)
        else:
            data_dict[key].Fill(data[0], weight[0])

    def fill2D(data_dict, key, Xdata, Ydata, weight):
        if type(Xdata) == float and type(Ydata) == float:
            data_dict[key].Fill(Xdata, Ydata, weight)
        elif len(Xdata) > 1 and len(Ydata) > 1:
            for dx, dy, w in zip(Xdata, Ydata, weight):
                data_dict[key].Fill(dx, dy, w)
        else:
            data_dict[key].Fill(Xdata[0], Ydata[0], weight[0])

    def storedata1D(data_dict, key, data, weight, Xnbins, Xmin, Xmax):
        try:
            fill1D(data_dict, key, data, weight)
        except KeyError:
            data_dict[key] = _rt.TH1D(key, key, Xnbins, Xmin, Xmax)
            fill1D(data_dict, key, data, weight)

    def storedata1DLog(data_dict, key, data, weight, Xnbins, XminExp, XmaxExp):
        try:
            fill1D(data_dict, key, data, weight)
        except KeyError:
            data_dict[key] = _rt.TH1D(key, key, Xnbins, _np.logspace(XminExp, XmaxExp, Xnbins + 1))
            fill1D(data_dict, key, data, weight)

    def storedata2D(data_dict, key, Xdata, Ydata, weight, Xnbins, Ynbins, Xmin, Xmax, Ymin, Ymax):
        try:
            fill2D(data_dict, key, Xdata, Ydata, weight)
        except KeyError:
            data_dict[key] = _rt.TH2D(key, key, Xnbins, Xmin, Xmax, Ynbins, Ymin, Ymax)
            fill2D(data_dict, key, Xdata, Ydata, weight)

    def storedata2DLog(data_dict, key, Xdata, Ydata, weight, Xnbins, Ynbins, XminExp, XmaxExp, YminExp, YmaxExp):
        try:
            fill2D(data_dict, key, Xdata, Ydata, weight)
        except KeyError:
            data_dict[key] = _rt.TH2D(key, key, Xnbins, _np.logspace(XminExp, XmaxExp, Xnbins + 1), Ynbins, _np.logspace(YminExp, YmaxExp, Ynbins + 1))
            fill2D(data_dict, key, Xdata, Ydata, weight)

    def getParticleName(partID):
        try:
            return Particle_type_dict[partID]['Name']
        except KeyError:
            return partID

    for j, t in enumerate(et):
        _printProgressBar(j, npart, prefix='Building hist from {}. Event {}/{}:'.format(inputfilename.split('/')[-1], j, npart), suffix='Complete', length=50)
        for sampler in e.Samplers:
            weight = sampler.weight
            name = sampler.samplerName
            if len(weight) != 0:
                for coord in HIST_SIZES_1D.keys():
                    storedata1D(HIST_DICT, "{}_{}_All".format(name, coord), getattr(sampler, coord), weight,
                                nbins, HIST_SIZES_1D[coord]['Xmin'], HIST_SIZES_1D[coord]['Xmax'])
                    for i, partID in enumerate(sampler.partID):
                        particle = getParticleName(partID)
                        storedata1D(HIST_DICT, "{}_{}_{}".format(name, coord, particle), getattr(sampler, coord)[i], weight[i],
                                    nbins, HIST_SIZES_1D[coord]['Xmin'], HIST_SIZES_1D[coord]['Xmax'])
                for coord in HIST_SIZES_1D_LOG.keys():
                    storedata1DLog(HIST_DICT, "{}_{}_All".format(name, coord), getattr(sampler, coord), weight,
                                   nbins, HIST_SIZES_1D_LOG[coord]['XminExp'], HIST_SIZES_1D_LOG[coord]['XmaxExp'])
                    for i, partID in enumerate(sampler.partID):
                        particle = getParticleName(partID)
                        storedata1DLog(HIST_DICT, "{}_{}_{}".format(name, coord, particle), getattr(sampler, coord)[i], weight[i],
                                       nbins, HIST_SIZES_1D_LOG[coord]['XminExp'], HIST_SIZES_1D_LOG[coord]['XmaxExp'])
                for coords in HIST_SIZES_2D.keys():
                    storedata2D(HIST_DICT, "{}_{}_All".format(name, coords), getattr(sampler, HIST_SIZES_2D[coords]['Xcoord']),
                                getattr(sampler, HIST_SIZES_2D[coords]['Ycoord']), weight, nbins, nbins,
                                HIST_SIZES_2D[coords]['Xmin'], HIST_SIZES_2D[coords]['Xmax'],
                                HIST_SIZES_2D[coords]['Ymin'], HIST_SIZES_2D[coords]['Ymax'])
                    for i, partID in enumerate(sampler.partID):
                        particle = getParticleName(partID)
                        storedata2D(HIST_DICT, "{}_{}_{}".format(name, coords, particle), getattr(sampler, HIST_SIZES_2D[coords]['Xcoord'])[i],
                                    getattr(sampler, HIST_SIZES_2D[coords]['Ycoord'])[i], weight[i], nbins, nbins,
                                    HIST_SIZES_2D[coords]['Xmin'], HIST_SIZES_2D[coords]['Xmax'],
                                    HIST_SIZES_2D[coords]['Ymin'], HIST_SIZES_2D[coords]['Ymax'])
    _printProgressBar(npart, npart, prefix='Building hist from {}. Event {}/{}:'.format(inputfilename.split('/')[-1], npart, npart), suffix='Complete', length=50)

    for hist in HIST_DICT:
        HIST_DICT[hist].Scale(ELECTRONS_PER_BUNCH/npart)

    outputfilename = inputfilename.replace('04_dataLocal', '06_analysis/root_files/beamDump').replace('05_dataFarm', '06_analysis/root_files/beamDump').replace('.root', '')
    outfile = _bd.Data.CreateEmptyRebdsimFile('{}_hist.root'.format(outputfilename), root_data.header.nOriginalEvents)
    _bd.Data.WriteROOTHistogramsToDirectory(outfile, "Event/MergedHistograms", list(HIST_DICT.values()))
    outfile.Close()


def combineHistFiles(globstring):
    filelist = _gl.glob(globstring)
    if not filelist:
        raise FileNotFoundError("Glob did not find any files")
    folder = filelist[0].split('/')[0] + '/' + filelist[0].split('/')[1] + '/'
    prefix = filelist[0].split('/')[-1].split('_part')[0][3:23]
    suffix = filelist[0].split('_part')[-1].replace('hist', 'merged_hist')
    npart = 0
    for filename in filelist:
        npart += int(filename.split('/')[-1].split('_part')[0].split('_')[-1])
    outputfile = folder + prefix + "{}_part".format(npart) + suffix
    _sub.call('rebdsimCombine ' + outputfile + ' ' + globstring, shell=True)


def getHistList(rebdsimfile, sampler=None, coord=None, particle=None):
    histlist = rebdsimfile.histograms.keys()

    if sampler is not None:
        histlist = [k for k in histlist if sampler in k]
    if coord is not None:
        histlist = [k for k in histlist if '_'+coord+'_' in k]
    if particle is not None:
        histlist = [k for k in histlist if particle in k]
    return histlist


def plot1DHist(inputfilename, sampler=None, coord=None, particle=None, xlabel=None, ylabel=None,
               xLogScale=False, yLogScale=False, elinewidth=0, printLegend=True):
    rebdsimfile = _bd.Data.Load(inputfilename)
    npart = rebdsimfile.header.nOriginalEvents
    histlist = getHistList(rebdsimfile, sampler=sampler, coord=coord, particle=particle)

    fig, ax = plotOptions()
    for histname in histlist:
        try:
            python_hist = rebdsimfile.histograms1dpy[histname]
            _bd.Plot.Histogram1D(python_hist, xlabel=xlabel, ylabel=ylabel, title=None, log=yLogScale, ax=ax, elinewidth=elinewidth,
                                 label=histname.split('/')[-1])
        except KeyError:
            pass

    ax.relim()
    ax.autoscale()

    if xLogScale:
        _plt.xscale("log")
    if printLegend:
        _plt.legend()


def plot2DHist(inputfilename, sampler=None, coord=None, particle=None, xlabel=None, ylabel=None, zlabel=None,
               xLogScale=False, yLogScale=False, zLogScale=False):
    rebdsimfile = _bd.Data.Load(inputfilename)
    npart = rebdsimfile.header.nOriginalEvents
    histname = getHistList(rebdsimfile, sampler=sampler, coord=coord, particle=particle)[0]

    fig, ax = plotOptions()
    python_hist = rebdsimfile.histograms2dpy[histname]
    _bd.Plot.Histogram2D(python_hist, xLogScale=xLogScale, yLogScale=yLogScale, logNorm=zLogScale, ax=ax,
                         xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=None)


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