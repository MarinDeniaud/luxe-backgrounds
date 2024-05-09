import pybdsim as _bd
import ROOT as _rt
import numpy as _np
import matplotlib.pyplot as _plt
import glob as _gl
import subprocess as _sub
import G4Dict


Particle_type_dict = G4Dict.Particle_type_dict
Process_type_dict = G4Dict.Process_type_subtype_dict


def analysis(inputfilename, nbins=50, ELECTRONS_PER_BUNCH=2e9):
    root_data = _bd.Data.Load(inputfilename)
    e = root_data.GetEvent()
    et = root_data.GetEventTree()
    npart = et.GetEntries()

    HIST_DICT = {}
    HIST_SIZES_1D = {'x':  {'Name': "$X$",  'Unit': '[m]',   'Xmin': -2.5, 'Xmax': 2.5},
                     'y':  {'Name': "$Y$",  'Unit': '[m]',   'Xmin': -2.5, 'Xmax': 2.5},
                     'xp': {'Name': "$X'$", 'Unit': '[rad]', 'Xmin': -1,   'Xmax': 1},
                     'yp': {'Name': "$Y'$", 'Unit': '[rad]', 'Xmin': -1,   'Xmax': 1}}
    HIST_SIZES_1D_LOG = {'energy':        {'Name': "$E$",  'Unit': '[GeV]', 'XminExp': -4, 'XmaxExp': 1},
                         'kineticEnergy': {'Name': "$kE$", 'Unit': '[GeV]', 'XminExp': -4, 'XmaxExp': 1}}
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


class Trajectories:
    def __init__(self, bdsim_data, evtnb):
        self.track_data = _bd.Data.TrajectoryData(bdsim_data, evtnb)
        self.track_table = self.track_data.trajectories

    def printTrajLog(self):
        print(f'{"index":7s}{"partID":11s}{"min Z":10s}{"max Z":10s}{"trackID":10s}{"parentID":10s}{"parentIDX":10s}'
              f'{"parentStepIDX":15s}{"primaryStepIDX":15s}')
        for i, track in enumerate(self.track_table):
            line = [track['partID'], round(track['Z'].min(), 3), round(track['Z'].max(), 3), track['trackID'], track['parentID'],
                    track['parentIDX'], track['parentStepIDX'], track['primaryStepIDX']]
            print(f'{str(i):7s}{str(line[0]):11s}{str(line[1]):10s}{str(line[2]):10s}{str(line[3]):10s}{str(line[4]):10s}'
                  f'{str(line[5]):10s}{str(line[6]):15s}{str(line[7]):15s}')

    def getEndOfChainTrackID(self):
        trackIDlist = []
        parentIDlist = []
        for track in self.track_table:
            trackIDlist.append(track['trackID'])
            parentIDlist.append(track['parentID'])
        IDlist = _np.setdiff1d(trackIDlist, parentIDlist)
        return _np.delete(IDlist, _np.argwhere(IDlist == 1))

    def getEndOfChainIndex(self):
        trackIDs = self.getEndOfChainTrackID()
        indexList = []
        for trackID in trackIDs:
            indexList.append(self.getIndexWithTrackID(trackID))
        return indexList

    def getEndOfChainIndexWithCut(self, scut=18.5):
        indexList = []
        for index, track in enumerate(self.track_table):
            if track['Z'].max() >= scut and track['trackID'] != 0 and track['trackID'] != 1:
                indexList.append(index)
        return indexList

    def getAttributesDict(self, index):
        track = self.track_table[index]
        track_dict = {'index': index}
        for attribute in ['partID', 'trackID', 'parentID', 'parentIDX', 'parentStepIDX', 'primaryStepIDX']:
            track_dict[attribute] = track[attribute]
        track_dict['Zmin'] = track['Z'].min()
        track_dict['Zmax'] = track['Z'].max()
        track_dict['partName'] = self.getParticleName(index)
        return track_dict

    def getAttribute(self, index, attribute):
        return self.track_table[index][attribute]

    def findOtherIndicesWithAttribute(self, index, attribute, value):
        Indices = []
        for i, track in enumerate(self.track_table):
            if track[attribute] == value and i != index:
                Indices.append(i)
        return Indices

    def getIndexWithTrackID(self, trackID):
        for i, track in enumerate(self.track_table):
            if track['trackID'] == trackID:
                return i
            else:
                raise ValueError("Unable to find track {}".format(trackID))

    def getParticleName(self, index):
        partID = self.getAttribute(index, 'partID')
        return self.getParticleNameFromID(partID)

    def getParticleNameFromID(self, partID):
        try:
            partName = Particle_type_dict[partID]['Name']
        except KeyError:
            partName = partID
        return partName

    def getParentIndex(self, index):
        return self.getAttribute(index, 'parentIDX')

    def getSisterIndices(self, index):
        parentID = self.getAttribute(index, 'parentID')
        return self.findOtherIndicesWithAttribute(index, 'parentID', parentID)

    def getDaugtherIndices(self, index):
        trackID = self.getAttribute(index, 'trackID')
        return self.findOtherIndicesWithAttribute(index, 'parentID', trackID)

    def getPreProcessIDs(self, index):
        prePT = self.getAttribute(index, 'prePT')
        prePST = self.getAttribute(index, 'prePST')
        preProcessIDs = []
        for PT, PST in zip(prePT, prePST):
            preProcessIDs.append((PT, PST))
        return preProcessIDs

    def getPostProcessIDs(self, index):
        postPT = self.getAttribute(index, 'postPT')
        postPST = self.getAttribute(index, 'postPST')
        postProcessIDs = []
        for PT, PST in zip(postPT, postPST):
            postProcessIDs.append((PT, PST))
        return postProcessIDs

    def getPreProcessNames(self, index):
        preProcessIDs = self.getPreProcessIDs(index)
        preProcessNames = []
        for PT, PST in preProcessIDs:
            try:
                preProcessNames.append((Process_type_dict[PT]['Name'], Process_type_dict[PT]['Subtype'][PST]))
            except KeyError:
                preProcessNames.append((PT, PST))
        return preProcessNames

    def getPostProcessNames(self, index):
        postProcessIDs = self.getPostProcessIDs(index)
        postProcessNames = []
        for PT, PST in postProcessIDs:
            try:
                postProcessNames.append((Process_type_dict[PT]['Name'], Process_type_dict[PT]['Subtype'][PST]))
            except KeyError:
                postProcessNames.append((PT, PST))
        return postProcessNames

    def getCreationProcessAndParentParticleName(self, index):
        partID = self.getAttribute(index, 'partID')
        while self.getAttribute(index, 'partID') == partID:
            parentStepIndex = self.getAttribute(index, 'parentStepIDX')
            index = self.getAttribute(index, 'parentIDX')
        creationProcess = self.getPostProcessNames(index)[parentStepIndex]
        return self.getParticleName(index), creationProcess

    def getProcessChain(self, index, transpSteps=False):
        partNameList = [self.getParticleName(index)]
        processList = [self.getPostProcessNames(index)]
        while self.getAttribute(index, 'parentID') != 0:
            parentStepIndex = self.getAttribute(index, 'parentStepIDX')
            index = self.getAttribute(index, 'parentIDX')
            partNameList.insert(0, self.getParticleName(index))
            processList.insert(0, self.getPostProcessNames(index)[0:parentStepIndex + 1])
        chain_dict = {}
        for i in range(len(partNameList)):
            partName = str(i) + '_' + partNameList[i]
            chain_dict[partName] = []
            if not transpSteps:
                for process in processList[i]:
                    if process[0] != 'fTransportation' and process[0] != 'fParallel':
                        chain_dict[partName].append(process)
            else:
                chain_dict[partName] = processList[i]
        return chain_dict


class TrajData:
    def __init__(self, inputfilename):
        self.inputfilename = inputfilename
        self.bdsim_data = _bd.Data.Load(inputfilename)
        self.Event = self.bdsim_data.GetEvent()
        self.EventTree = self.bdsim_data.GetEventTree()
        self.Samplers = self.Event.Samplers
        self.npart = self.EventTree.GetEntries()

    def Trajectory(self, evtnb):
        return Trajectories(self.bdsim_data, evtnb)

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

    def getEndOfChainIndex(self, evtnb, scut=18.5):
        traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
        trackIDs = self.getEndOfChainTrackID(evtnb)
        indexList = []
        for trackID in trackIDs:
            for index, traj in enumerate(traj_data):
                if trackID == traj['trackID'] and traj['Z'].max() >= scut and traj['trackID'] != 0:
                    indexList.append(index)
        return indexList

    def findUnknownPartID(self, includeNucleus=False):
        s = set()
        for evtnb, evt in enumerate(self.EventTree):
            _printProgressBar(evtnb, self.npart, prefix='Looking for unknown particle. Event {}/{}:'.format(evtnb, self.npart), suffix='Complete', length=50)
            traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
            for traj in traj_data:
                partID = traj['partID']
                try:
                    name = Particle_type_dict[partID]['Name']
                except KeyError:
                    if partID > 1000000000 and includeNucleus:
                        s.add(partID)
                    elif partID < 1000000000 and not includeNucleus:
                        s.add(partID)
        _printProgressBar(self.npart, self.npart, prefix='Looking for unknown particle. Event {}/{}:'.format(self.npart, self.npart), suffix='Complete', length=50)
        return s

    def findEventAndIndexByPartID(self, partIDList, scut=0):
        print(f'{"partID":10s}{"Event nb":10s}{"Index":10s}')
        for evtnb, evt in enumerate(self.EventTree):
            traj_data = _bd.Data.TrajectoryData(self.bdsim_data, evtnb)
            for index, traj in enumerate(traj_data.trajectories):
                if traj['partID'] in partIDList and traj['Z'].max() >= scut:
                    print(f"{str(traj['partID']):10s}{str(evtnb):10s}{str(index):10s}")

    def storeAllProcesses(self):
        # IT TAKES AGES OMG MARIN OPTIMIZE
        data = {}
        for evtnb, evt in enumerate(self.EventTree):
            _printProgressBar(evtnb, self.npart, prefix='Storing processes. Event {}/{}:'.format(evtnb, self.npart), suffix='Complete', length=50)
            indexList = self.getEndOfChainIndex(evtnb)
            for index in indexList:
                ParentName, PostProcessName = self.Trajectory(evtnb).getCreationProcessAndParentParticleName(index)
                try:
                    data[ParentName][PostProcessName] += 1
                except KeyError:
                    data[ParentName] = {PostProcessName: 1}
        _printProgressBar(self.npart, self.npart, prefix='Storing processes. Event {}/{}:'.format(self.npart, self.npart), suffix='Complete', length=50)
        return data

    def storeAllProcessesWithCut(self, scut=18.5):
        # IT TAKES AGES OMG MARIN OPTIMIZE
        data = {}
        for evtnb, evt in enumerate(self.EventTree):
            _printProgressBar(evtnb, self.npart, prefix='Storing processes. Event {}/{}:'.format(evtnb, self.npart), suffix='Complete', length=50)
            indexList = self.Trajectory(evtnb).getEndOfChainIndexWithCut(scut=scut)
            for index in indexList:
                ParentName, PostProcessName = self.Trajectory(evtnb).getCreationProcessAndParentParticleName(index)
                try:
                    data[ParentName][PostProcessName] += 1
                except KeyError:
                    data[ParentName] = {PostProcessName: 1}
        _printProgressBar(self.npart, self.npart, prefix='Storing processes. Event {}/{}:'.format(self.npart, self.npart), suffix='Complete', length=50)
        return data

    def storeCreationProcessesForOneEvent(self, evtnb, data, scut=18.5):
        Traj = self.Trajectory(evtnb)
        for index, track in enumerate(Traj.track_table):
            if track['Z'].max() >= scut and track['trackID'] != 0 and track['trackID'] != 1:
                partName = Traj.getParticleName(index)
                parentPartName, creationProcess = Traj.getCreationProcessAndParentParticleName(index)
                try:
                    data[partName]["count"] += 1
                except KeyError:
                    data[partName] = {"Parent": parentPartName, "Process": creationProcess, "count": 1}

    def storeCreationProcessesAllEvents(self, scut=18.5):
        data = {}
        for evtnb, evt in enumerate(self.EventTree):
            _printProgressBar(evtnb, self.npart, prefix='Storing processes. Event {}/{}:'.format(evtnb, self.npart), suffix='Complete', length=50)
            self.storeCreationProcessesForOneEvent(evtnb, data, scut=scut)
        _printProgressBar(self.npart, self.npart, prefix='Storing processes. Event {}/{}:'.format(self.npart, self.npart), suffix='Complete', length=50)
        return data


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


def _printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    import sys
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix) + printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()