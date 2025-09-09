import h5py as _h5
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import time as t
import pymad8 as _m8
import matplotlib.ticker as mtick
import pickle as _pk


class EnergyPerBunchData:
    def __init__(self, inputfilename):
        self.inputfilename = inputfilename
        self.file = open(inputfilename, 'rb')
        self.energy_dict = _pk.load(self.file)

    def RemoveNan(self):
        mask = (_np.nan_to_num(self.energy_dict['data']) != 0)
        return self.energy_dict['data'][mask[:, 1]]

    def CutData(self, minBunch=0, maxBunch=None):
        E = self.RemoveNan()
        return E[minBunch:maxBunch]

    def PlotEnergyPerBunch(self, minBunch=0, maxBunch=None, color=None , label=True):
        E = self.CutData(minBunch=minBunch, maxBunch=maxBunch)
        if label:
            _plt.plot(E[:, 0], E[:, 1], color=color, label="$\overline{{E}}$ = {:.2e}".format(_np.mean(E[:400][:, 1])))
        else:
            _plt.plot(E[:, 0], E[:, 1], color=color)

    def PlotEnergyHist(self, minBunch=0, maxBunch=None, color=None):
        E = self.CutData(minBunch=minBunch, maxBunch=maxBunch)
        _plt.hist(E[:, 1], bins=20, color=color, label="$\sigma_E$ = {:.2e}".format(_np.std(E[:400][:, 1])))

    def PlotAll(self, minCut=None, maxCut=None, figsize=[12, 5]):
        if minCut is None or maxCut is None:
            raise ValueError("Must provide cuts for plotting")
        fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 2], height_ratios=None, sharex=False, sharey=False, font_size=15)
        ax[0].yaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].yaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[0].xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))

        _plt.subplot(1, 2, 1)
        self.PlotEnergyPerBunch(0, minCut, color='C0')
        self.PlotEnergyPerBunch(maxCut, None, color='C1')
        self.PlotEnergyPerBunch(minCut, maxCut, color='grey', label=False)
        _plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        #_plt.xticks(rotation=45)
        _plt.ylabel("Energy [MeV]")
        _plt.xlabel("Bunch time [ns]")
        _plt.legend()

        _plt.subplot(1, 2, 2)
        self.PlotEnergyHist(0, minCut, color='C0')
        self.PlotEnergyHist(maxCut, None, color='C1')
        _plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        #_plt.xticks(rotation=45)
        _plt.xlabel("Energy [MeV]")
        _plt.legend()


class EnergyPerTrainData:
    def __init__(self, inputfilename):
        self.inputfilename = inputfilename
        self.df = getH5dataInDF(inputfilename, getEnergy=True, getPosition=False)

    def GetValidDF(self):
        return self.df[self.df['Valid'] == 1]

    def GetFirstBunchDF(self):
        df_valid = self.GetValidDF()
        return df_valid[df_valid.index.get_level_values('BunchID') == 0]

    def PlotHist(self):
        df_e = self.GetFirstBunchDF()
        _plt.hist(df_e['E'], bins=20, label="$\sigma_E$ = {:.2e}".format(_np.std(df_e['E'])))

    def PlotHistPerTrain(self):
        df_e = self.GetFirstBunchDF()
        df_train = df_e[df_e.index.get_level_values('BPM') == df_e.index.get_level_values('BPM')[0]]
        _plt.hist(df_train['E'], bins=20, label="$\sigma_E$ = {:.2e}".format(_np.std(df_train['E'])))

    def PlotHistPerBPM(self):
        df_e = self.GetFirstBunchDF()
        df_bpm = df_e[df_e.index.get_level_values('TrainID') == df_e.index.get_level_values('TrainID')[0]]
        _plt.hist(df_bpm['E'], bins=20, label="$\sigma_E$ = {:.2e}".format(_np.std(df_bpm['E'])))

    def PlotEnergyPerTrain(self):
        df_e = self.GetFirstBunchDF()
        df_train = df_e[df_e.index.get_level_values('BPM') == df_e.index.get_level_values('BPM')[0]]
        train_ids = df_train.index.get_level_values('TrainID') - df_train.index.get_level_values('TrainID')[0]
        _plt.plot(train_ids, df_train['E'], label="$\overline{{E}}$ = {:.2e}".format(_np.mean(df_train['E'])))

    def PlotEnergyPerBPM(self):
        df_e = self.GetFirstBunchDF()
        df_bpm = df_e[df_e.index.get_level_values('TrainID') == df_e.index.get_level_values('TrainID')[0]]
        _plt.plot(df_bpm.index.get_level_values('BPM'), df_bpm['E'], label="$\overline{{E}}$ = {:.2e}".format(_np.mean(df_bpm['E'])))

    def PlotAll(self, figsize=[9, 6]):
        fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 2], height_ratios=None, sharex=False, sharey=False, font_size=15)
        ax[0].yaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].yaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[0].xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))

        _plt.subplot(1, 2, 1)
        self.PlotEnergyPerTrain()
        _plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        #_plt.xticks(rotation=45)
        _plt.ylabel("Energy [GeV]")
        _plt.xlabel("Train ID")
        _plt.legend()

        _plt.subplot(1, 2, 2)
        self.PlotHistPerTrain()
        _plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        #_plt.xticks(rotation=45)
        _plt.xlabel("Energy [GeV]")
        _plt.legend()

        # _plt.subplot(2, 2, 3)
        # self.PlotEnergyPerBPM()
        # _plt.ylabel("Energy [GeV]")
        # _plt.xlabel("BPM")
        # _plt.xticks(rotation=90)
        # _plt.legend()

        # _plt.subplot(2, 2, 4)
        # self.PlotHistPerBPM()
        # _plt.xlabel("Energy [GeV]")
        # _plt.legend()


class ArrivalTimeData:
    def __init__(self, inputfilename):
        self.inputfilename = inputfilename
        self.df = getH5dataInDF(inputfilename, getTime=True, getPosition=False)

    def GetValidDF(self):
        df_valid = self.df[self.df['Valid'] == 1]
        return df_valid

    def GetTimePer(self, Bunch=None, Train=None, BPM=None):
        df = self.GetValidDF()
        if Bunch is not None:
            df = df[df.index.get_level_values('BunchID') == Bunch]
        if Train is not None:
            df = df[df.index.get_level_values('TrainID') == Train]
        if BPM is not None:
            df = df[df.index.get_level_values('BPM') == BPM]
        return df

    def PlotTimePer(self, index='BunchID', Bunch=None, Train=None, BPM=None, minCut=0, maxCut=None):
        df_time = self.GetTimePer(Bunch, Train, BPM)
        ids = df_time.index.get_level_values(index)[minCut:maxCut] - df_time.index.get_level_values(index)[0]
        ids_cut_min = df_time.index.get_level_values(index)[0:minCut] - df_time.index.get_level_values(index)[0]
        ids_cut_max = df_time.index.get_level_values(index)[maxCut:None] - df_time.index.get_level_values(index)[0]
        _plt.plot(ids, df_time['Time'][minCut:maxCut], label="$\overline{{t}}$ = {:.2e}".format(_np.mean(df_time['Time'][minCut:maxCut])))
        if minCut != 0:
            _plt.plot(ids_cut_min, df_time['Time'][0:minCut], color='grey')
        if maxCut is not None:
            _plt.plot(ids_cut_max, df_time['Time'][maxCut:None], color='grey')

    def PlotTimeHistPer(self, Bunch=None, Train=None, BPM=None, minCut=0, maxCut=None, bins=30):
        df_time = self.GetTimePer(Bunch, Train, BPM)
        _plt.hist(df_time['Time'][minCut:maxCut], bins=bins, label="$\sigma_t$ = {:.2e}".format(_np.std(df_time['Time'][minCut:maxCut])))

    def PlotTimeAndHistPer(self, index='BunchID', Bunch=None, Train=None, BPM=None, minCut=0, maxCut=None, bins=30, figsize=[12, 5]):
        fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 2], height_ratios=None, sharex=False, sharey=False, font_size=15)
        ax[0].yaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].yaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[0].xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))

        _plt.subplot(1, 2, 1)
        self.PlotTimePer(index=index, Bunch=Bunch, Train=Train, BPM=BPM, minCut=minCut, maxCut=maxCut)
        _plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        #_plt.xticks(rotation=45)
        _plt.ylabel("Time [ps]")
        _plt.xlabel(index)
        _plt.legend()

        _plt.subplot(1, 2, 2)
        self.PlotTimeHistPer(Bunch=Bunch, Train=Train, BPM=BPM, minCut=minCut, maxCut=maxCut, bins=bins)
        _plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        #_plt.xticks(rotation=45)
        _plt.xlabel("Time [ps]")
        _plt.legend()

    def PlotAllTimeHist(self, bins=50, figsize=[9, 6]):
        df_time = self.GetValidDF()
        fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 1], height_ratios=None, sharex=False, sharey=False, font_size=15)
        _plt.hist(df_time['Time'], bins=bins, label="std = " + str(_np.std(df_time['Time'])))
        _plt.ticklabel_format(axis="x", style="sci", scilimits=(-6, -6))
        _plt.xlabel("Time [s]")
        _plt.legend()



class CrispData:
    def __init__(self, inputfilename):
        self.inputfilename = inputfilename
        self.df = _pd.read_pickle(inputfilename)

    def calcLengthOneTrain(self, trainID):
        time = self.df.Time
        train = self.df[self.df.keys()[1+trainID]]
        return 2 * _np.std(time[train > 0])

    def calcLengthOneTrainCUMSUM(self, trainID, percentile=0.95, plotCumsum=False):
        time = self.df.Time
        current = self.df[self.df.keys()[1 + trainID]]
        if list(current.unique()) == [0]:
            return 0
        cumsum = _np.cumsum(current/current.sum())

        def calcBunchEdge(time, cumsum, value):
            time1 = time.to_numpy()[cumsum.to_numpy() < value][-1]
            time2 = time.to_numpy()[cumsum.to_numpy() > value][0]
            cumsum1 = cumsum.to_numpy()[cumsum.to_numpy() < value][-1]
            cumsum2 = cumsum.to_numpy()[cumsum.to_numpy() > value][0]

            slope = (cumsum2 - cumsum1) / (time2 - time1)
            intercept = cumsum1 - slope * time1

            return (value - intercept)/slope

        timemax = calcBunchEdge(time, cumsum, percentile)
        timemin = calcBunchEdge(time, cumsum, 1-percentile)

        if plotCumsum:
            _plt.plot(time, cumsum, '+', color='C0')
            _plt.hlines([1-percentile, percentile], xmin=min(time), xmax=max(time), colors=['C1', 'C2'], linestyles='--')
            _plt.vlines([timemin, timemax], ymin=min(cumsum), ymax=max(cumsum), colors=['C1', 'C2'], linestyles='-')

        return timemax - timemin

    def calcLengthAllTrains(self, percentile=0.95, excludefirsts=True):
        Lengths = _np.array([])
        trainIDs = range(len(self.df.keys())-1)
        for trainID in trainIDs:
            Lengths = _np.append(Lengths, self.calcLengthOneTrainCUMSUM(trainID, percentile=percentile))
        if excludefirsts:
            return trainIDs[100:], Lengths[100:]
        return trainIDs, Lengths

    def plotFirstTrainProfile(self, figsize=[9, 7]):
        fig, ax = plotOptions(figsize=figsize)
        _plt.plot(self.df.Time, self.df.Train1, label=self.df.Train1.name)
        _plt.ylabel('$Current$')
        _plt.xlabel('$Time$ [fs]')
        _plt.legend()

    def plotAllTrainProfile(self, figsize=[10, 8]):
        fig, ax = plotOptions(figsize=figsize)
        time = self.df.Time
        for train in self.df.keys()[1:]:
            _plt.plot(time, self.df[train])
        _plt.ylabel('$Current$')
        _plt.xlabel('$Time$ [fs]')

    def plotSelectTrainProfile(self, select=_np.array([0]), figsize=[10, 8]):
        fig, ax = plotOptions(figsize=figsize)
        time = self.df.Time
        for train in self.df.keys()[1:][select]:
            _plt.plot(time, self.df[train], label=self.df[train].name)
        _plt.ylabel('$Current$')
        _plt.xlabel('$Time$ [fs]')
        _plt.legend()

    def plotLengthPerTrain(self, percentile=0.95, figsize=[14, 4]):
        fig, ax = plotOptions(figsize=figsize)
        trainIDs, Lengths = self.calcLengthAllTrains(percentile=percentile)
        _plt.plot(trainIDs, Lengths, label='$\overline{{\Delta_t}}$ = {:.2e}'.format(_np.mean(Lengths)))
        _plt.ylabel('$\Delta_t$ [fs]')
        _plt.xlabel('$Train ID$')
        _plt.legend()

    def plotLenghtHist(self, percentile=0.95, bins=50, figsize=[9, 7]):
        fig, ax = plotOptions(figsize=figsize)
        trainIDs, Lengths = self.calcLengthAllTrains(percentile=percentile)
        Lengths = Lengths[Lengths.nonzero()]
        _plt.hist(Lengths, bins=bins, label="$\sigma_{{\Delta_t}}$ = {:.2e}".format(_np.std(Lengths)))
        _plt.xlabel('$\Delta_t$ [fs]')
        _plt.legend()

    def PlotLengthAndHistPer(self, percentile=0.95, bins=50, figsize=[12, 5]):
        fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 2], height_ratios=None, sharex=False, sharey=False, font_size=15)
        ax[0].yaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].yaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[0].xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))
        ax[1].xaxis.set_major_formatter(mtick.ScalarFormatter(useOffset=False))

        trainIDs, Lengths = self.calcLengthAllTrains(percentile=percentile)

        _plt.subplot(1, 2, 1)
        _plt.plot(trainIDs, Lengths, label='$\overline{{\Delta_t}}$ = {:.2e}'.format(_np.mean(Lengths)))
        _plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        #_plt.xticks(rotation=45)
        _plt.ylabel('$\Delta_t$ [fs]')
        _plt.xlabel('$Train ID$')
        _plt.legend()

        _plt.subplot(1, 2, 2)
        _plt.hist(Lengths, bins=bins, label="$\sigma_{{\Delta_t}}$ = {:.2e}".format(_np.std(Lengths)))
        _plt.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        #_plt.xticks(rotation=45)
        _plt.xlabel('$\Delta_t$ [fs]')
        _plt.legend()


bunch_pattern_adress = "XFEL.DIAG/TIMER/DI1914TL/BUNCH_PATTERN"

bl_time_adress = "XFEL.SDIAG/THZ_SPECTROMETER.RECONSTRUCTION/CRD.1934.TL.NTH/OUTPUT_TIMES"
bl_current_adress = "XFEL.SDIAG/THZ_SPECTROMETER.RECONSTRUCTION/CRD.1934.TL.NTH/CURRENT_PROFILE"
bl_number_adress = "XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/NTH_BUNCH"


class BPMData:
    def __init__(self, inputfilename, excelfilename="~/Users/marindeniaud/Desktop/component_list_2023.07.02.xls",
                 EmitX=3.58e-11, EmitY=3.58e-11, Esprd=1e-6, getPosition=True, getCharge=False, getEnergy=False, getTime=False):
        print("Loaded file '{}'".format(inputfilename.split('/')[-1]))
        self.inputfilename = inputfilename
        self.bpm_adress = "XFEL.DIAG/BPM"
        self.energy_adress = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY.ALL"
        self.time_S_adress = "XFEL.SDIAG/BAM.DAQ/1932S.TL.ARRIVAL_TIME.RELATIVE"
        self.time_M_adress = "XFEL.SDIAG/BAM.DAQ/1932M.TL.ARRIVAL_TIME.RELATIVE"

        self.rawdata = _h5.File(inputfilename, 'r')
        self.bpmdata = self.rawdata[self.bpm_adress]
        self.energydata = self.rawdata[self.energy_adress]
        self.timeSdata = self.rawdata[self.time_S_adress]
        self.timeMdata = self.rawdata[self.time_M_adress]

        self.bpmIDs = _np.array(list(self.bpmdata.keys()))
        self.nbbpm = len(self.bpmIDs)

        self.trainIDs_raw = self.bpmdata[self.bpmIDs[0]]['TrainId'][:]
        self.trainIDs_matched = _np.setdiff1d(self.trainIDs_raw, self.getUnmatchedTrainID())
        self.nbtrain = len(self.trainIDs_matched)

        self.nbbunch = self.bpmdata[self.bpmIDs[0]]['X.TD'].shape[1]
        self.bunchIDs = _np.array(range(self.nbbunch))

        df_excel_T1 = _pd.read_excel("/Users/marindeniaud/Desktop/component_list_2023.07.02.xls", sheet_name='I1toT5D')
        df_excel_T1_bpm = df_excel_T1[df_excel_T1['NAME1'].isin(self.bpmIDs)]
        df_excel_T2 = _pd.read_excel("/Users/marindeniaud/Desktop/component_list_2023.07.02.xls", sheet_name='I1toT4D')
        df_excel_T2_bpm = df_excel_T2[df_excel_T2['NAME1'].isin(self.bpmIDs)]

        self.df_excel = _pd.concat((df_excel_T1_bpm, df_excel_T2_bpm[df_excel_T2['NAME1'].isin(df_excel_T1['NAME1']) == False]))
        self.calcBeamSize(EmitX, EmitY, Esprd)

        self.s_by_section = _np.array(self.df_excel.S)
        self.bpmIDs_by_section = _np.array(self.df_excel.NAME1)

        self.s_by_s = self.s_by_section[self.s_by_section.argsort()]
        self.bpmIDs_by_s = self.bpmIDs_by_section[self.s_by_section.argsort()]

        self.df_bpm = self.getH5dataInDF(getPosition=getPosition, getCharge=getCharge, getEnergy=getEnergy, getTime=getTime)

    def calcBeamSize(self, EmitX, EmitY, Esprd):
        SigmaX = []
        SigmaY = []
        SigmaXP = []
        SigmaYP = []
        E0 = self.df_excel['ENERGY'].to_numpy()[0]
        for i in range(self.nbbpm):
            BetaX = self.df_excel['BETX'].to_numpy()[i]
            BetaY = self.df_excel['BETY'].to_numpy()[i]
            GammaX = (1 + self.df_excel['ALFX'].to_numpy()[i] ** 2) / BetaX
            GammaY = (1 + self.df_excel['ALFY'].to_numpy()[i] ** 2) / BetaY
            DispX = self.df_excel['DX'].to_numpy()[i]
            DispY = self.df_excel['DY'].to_numpy()[i]
            DispXP = self.df_excel['DPX'].to_numpy()[i]
            DispYP = self.df_excel['DPY'].to_numpy()[i]

            # Beam size calculation
            SigmaX.append(_np.sqrt(BetaX * EmitX + (DispX * Esprd / E0) ** 2))
            SigmaY.append(_np.sqrt(BetaY * EmitY + (DispY * Esprd / E0) ** 2))
            # Beam divergence calculation
            SigmaXP.append(_np.sqrt(GammaX * EmitX + (DispXP * Esprd / E0) ** 2))
            SigmaYP.append(_np.sqrt(GammaY * EmitY + (DispYP * Esprd / E0) ** 2))

        self.df_excel = self.df_excel.assign(SIGX=SigmaX, SIGY=SigmaY, SIGXP=SigmaXP, SIGYP=SigmaYP)

    def isUnmatchedTrainIDinBPMs(self):
        for bpm in self.bpmIDs:
            check = _np.unique(self.bpmdata[self.bpmIDs[0]]['TrainId'][:] == self.bpmdata[bpm]['TrainId'][:])
            if False in check:
                return True
            return False

    def getUnmatchedTrainIDinBPMs(self):
        firstbpmtrains = self.bpmdata[self.bpmIDs[0]]['TrainId'][:]
        unmatched_trains = _np.array([])
        for bpm in self.bpmIDs:
            bpmtrains = self.bpmdata[bpm]['TrainId'][:]
            unmatched_trains = _np.append(unmatched_trains, _np.setdiff1d(firstbpmtrains, bpmtrains))
            unmatched_trains = _np.append(unmatched_trains, _np.setdiff1d(bpmtrains, firstbpmtrains))
        return unmatched_trains

    def getUnmatchedTrainID(self):
        bpmtrains = self.bpmdata['BPMI.1860.TL']['TrainId'][:]
        energytrains = self.energydata['TrainId'][:]
        timeStrains = self.timeSdata['TrainId'][:]
        timeMtrains = self.timeMdata['TrainId'][:]

        unmatched_trains = _np.array([])

        if self.isUnmatchedTrainIDinBPMs():
            unmatched_trains = _np.append(unmatched_trains, self.getUnmatchedTrainIDinBPMs())

        def diff2array(array1, array2):
            diff1 = _np.setdiff1d(array1, array2)
            diff2 = _np.setdiff1d(array2, array1)
            return _np.concatenate((diff1, diff2))

        unmatched_trains = _np.append(unmatched_trains, diff2array(bpmtrains, energytrains))
        unmatched_trains = _np.append(unmatched_trains, diff2array(bpmtrains, timeMtrains))
        unmatched_trains = _np.append(unmatched_trains, diff2array(bpmtrains, timeStrains))

        return _np.unique(unmatched_trains).astype(int)

    def getH5dataInDFviaDict(self):
        dict_bpm = {}
        keys = ['X', 'Y', 'S', 'Valid']
        for i, bpm in enumerate(self.bpmIDs_by_s):
            S = self.df_excel[self.df_excel['NAME1'] == bpm]['S'].to_numpy()[0]
            _printProgressBar(i, self.nbbpm, prefix='Load in dict', suffix='Complete', length=50)
            for train in self.trainIDs_matched[:100]:
                for bunch in self.trainIDs_matched[:100]:
                    dict_bpm[(bpm, train, bunch)] = {'X': 0.0, 'Y': 0.0, 'S': S, 'Valid': 1.0}
        _printProgressBar(self.nbbpm, self.nbbpm, prefix='Load in dict:', suffix='Complete', length=50)
        return _pd.DataFrame.from_dict(dict_bpm, orient="index").rename_axis(["BPM", "TrainID", "BunchID"])

    def getH5dataInDF(self, getPosition=True, getCharge=False, getEnergy=False, getTime=False):

        def storedata(data_dict, key, data, factor=1.0):
            try:
                data_dict[key].append(data[:] * factor)
            except:
                data_dict[key] = [data[:] * factor]

        data_dict = {}
        mask = _np.isin(self.trainIDs_raw, self.trainIDs_matched)
        for i, bpm in enumerate(self.bpmIDs_by_s):
            _printProgressBar(i, self.nbbpm, prefix='Loading {} bpms, {} trains and {} bunches in df:'.format(self.nbbpm, self.nbtrain, self.nbbunch),
                              suffix='Complete', length=50)
            storedata(data_dict, 'Valid', self.bpmdata[bpm]['BUNCH_VALID.TD'][mask])
            if getPosition:
                storedata(data_dict, 'X', self.bpmdata[bpm]['X.TD'][mask], factor=1e-3)  # mm converted in m
                storedata(data_dict, 'Y', self.bpmdata[bpm]['Y.TD'][mask], factor=1e-3)  # mm converted in m
                storedata(data_dict, 'S', _np.full((self.nbtrain, self.nbbunch), self.s_by_s[i]))  # m
            if getCharge:
                storedata(data_dict, 'Charge', self.bpmdata[bpm]['CHARGE.TD'][mask], factor=1e-9)  # nC converted to C
            if getTime:
                storedata(data_dict, 'TimeS', self.timeSdata['Value'][mask], factor=1e-12)  # ps converted to s
                storedata(data_dict, 'TimeM', self.timeMdata['Value'][mask], factor=1e-12)  # ps converted to s
            if getEnergy:
                storedata(data_dict, 'E', _np.tile(self.energydata['Value'][mask], (self.nbbunch, 1)).transpose(), factor=1e-3)  # MeV converted to GeV
        for key in data_dict:
            data_dict[key] = _np.asarray(data_dict[key]).flatten()
        df = _pd.DataFrame(data_dict, index=_pd.MultiIndex.from_product([range(s) for s in (self.nbbpm, self.nbtrain, self.nbbunch)],
                                                                        names=['BPM', 'TrainID', 'BunchID']))
        df.index.set_levels([self.bpmIDs_by_s, self.trainIDs_matched], level=[0, 1], inplace=True)
        _printProgressBar(self.nbbpm, self.nbbpm, prefix='Loading {} bpms, {} trains and {} bunches in df:'.format(self.nbbpm, self.nbtrain, self.nbbunch),
                          suffix='Complete', length=50)
        return df

    def reduceDFbyIndex(self, index, value):
        if type(index) == int:
            indexid = index
            indexname = self.df_bpm.index.names[0]
        elif type(index) == str:
            indexid = self.df_bpm.index.names.index(index)
            indexname = index
        else:
            raise TypeError('Unknown type {} for index value. Must be either int or str'.format(type(index)))

        try:
            value = self.df_bpm.index.levels[indexid][value]
        except IndexError:
            pass

        try:
            mask = self.df_bpm.index.get_level_values(indexname) == value
        except:
            mask = self.df_bpm.index.get_level_values(indexname).isin(value)
        return self.df_bpm.loc[mask]

    def reduceDFbyBPMTrainBunchByIndex(self, bpms=None, trains=None, bunches=None, valid=True):
        if valid:
            df = self.df_bpm[self.df_bpm['Valid'] == 1]
        if bpms is not None:
            df = self.reduceDFbyIndex('BPM', bpms)
        if trains is not None:
            df = self.reduceDFbyIndex('TrainID', trains)
        if bunches is not None:
            df = self.reduceDFbyIndex('BunchID', bunches)
        df.index = df.index.remove_unused_levels()
        return df

    def checkTrainBunchConsistancy(self):
        df_valid = self.reduceDFbyBPMTrainBunchByIndex()
        bpmids_count = df_valid.index.get_level_values(0).value_counts(sort=False).values
        trainids_count = df_valid.index.get_level_values(1).value_counts(sort=False).values
        bunchids_count = df_valid.index.get_level_values(2).value_counts(sort=False).values

        A = bpmids_count / self.nbtrain
        B = trainids_count
        C = bunchids_count / self.nbtrain

        fig, ax = plotOptions(figsize=[14, 8], rows_colums=[3, 1])
        _plt.subplot(3, 1, 1)
        _plt.plot(_np.abs(A - A.round()))
        ax[0].set_xticks(range(len(self.bpmIDs_by_s)))
        ax[0].set_xticklabels(self.bpmIDs_by_s, fontsize=8, rotation=45, ha='right')
        _plt.subplot(3, 1, 2)
        _plt.plot(B)
        ax[1].ticklabel_format(useOffset=False)
        _plt.subplot(3, 1, 3)
        _plt.plot(_np.abs(C - C.round()))

    def getBunchPattern(self, refT1='BPMA.2097.T1', refT2='BPMA.2161.T2', sample=1):
        df_reduced = reduceDFbyBPMTrainBunchByIndex(self.df_bpm, trains=1)
        df_T1 = df_reduced[df_reduced.index.get_level_values(0) == refT1]
        df_T2 = df_reduced[df_reduced.index.get_level_values(0) == refT2]
        bunchIDs_T1 = df_T1.index.get_level_values(2).unique().values
        bunchIDs_T2 = df_T2.index.get_level_values(2).unique().values
        bunchIDs_TL = df_reduced.index.get_level_values(2).unique().values
        bunchIDs_TL = _np.setdiff1d(bunchIDs_TL, bunchIDs_T1)
        bunchIDs_TL = _np.setdiff1d(bunchIDs_TL, bunchIDs_T2)

        return bunchIDs_TL[0::sample], bunchIDs_T1[0::sample], bunchIDs_T2[0::sample]


BPM_DICT = {'BPMI.1860.TL': {'MAD8_name': 'BPMI.Y1.TL', 'Line': 'TL', 'S': 1838.149255, 'X': 0.000000, 'Y': -2.389866},
            'BPMI.1863.TL': {'MAD8_name': 'BPMI.X1.TL', 'Line': 'TL', 'S': 1840.737255, 'X': 0.000000, 'Y': -2.390811},
            'BPMA.1868.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 1845.310755, 'X': 0.000000, 'Y': -2.392481},
            'BPMA.1873.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 1850.160755, 'X': 0.000000, 'Y': -2.394251},
            'BPMI.1878.TL': {'MAD8_name': 'BPMI.Y2.TL', 'Line': 'TL', 'S': 1856.104255, 'X': 0.000000, 'Y': -2.396422},
            'BPMI.1889.TL': {'MAD8_name': 'BPMI.X2.TL', 'Line': 'TL', 'S': 1866.304255, 'X': 0.000000, 'Y': -2.400146},
            'BPMI.1910.TL': {'MAD8_name': 'BPMI.Y3.TL', 'Line': 'TL', 'S': 1887.259255, 'X': 0.000000, 'Y': -2.407797},
            'BPMI.1925.TL': {'MAD8_name': 'BPMI.X3.TL', 'Line': 'TL', 'S': 1902.259255, 'X': 0.000000, 'Y': -2.413274},
            'BPMI.1930.TL': {'MAD8_name': 'BPMI.Y4.TL', 'Line': 'TL', 'S': 1907.559255, 'X': 0.000000, 'Y': -2.415209},
            'BPMI.1939.TL': {'MAD8_name': 'BPMI.X4.TL', 'Line': 'TL', 'S': 1916.304255, 'X': 0.000000, 'Y': -2.418402},
            'BPMA.1966.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 1943.710757, 'X': 0.000000, 'Y': -2.428409},  # 12
            'BPMD.1977.TL': {'MAD8_name': 'BPMD.TL',    'Line': 'TL', 'S': 1955.124959, 'X': 0.000000, 'Y': -2.432576},
            # TLD
            'BPMA.1995.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 1972.760701, 'X': 0.000000, 'Y': -2.439016},
            'BPMA.2011.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 1988.710701, 'X': 0.000000, 'Y': -2.444839},
            'BPMD.2022.TL': {'MAD8_name': 'BPMD.TL',    'Line': 'TL', 'S': 2000.125103, 'X': 0.000000, 'Y': -2.449007},
            'BPMA.2041.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 2018.710845, 'X': 0.000000, 'Y': -2.455793},  # 4

            'BPMA.2054.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 2031.915847, 'X': 0.000000, 'Y': -2.460614},  # Difference with mad8 lattice ...

            'BPMA.2040.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2017.710851, 'X': 0.240162, 'Y': -2.431523},
            'BPMA.2044.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2021.851529, 'X': 0.333960, 'Y': -2.433035},
            'BPMA.2055.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2032.375950, 'X': 0.558250, 'Y': -2.436877},
            'BPMA.2062.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2040.110684, 'X': 0.707628, 'Y': -2.439700},
            'BPMA.2068.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2046.031866, 'X': 0.805806, 'Y': -2.441862},
            'BPMA.2082.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2060.020296, 'X': 1.108241, 'Y': -2.446968},
            'BPMA.2088.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2065.865296, 'X': 1.341356, 'Y': -2.456983},
            'BPMA.2092.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2069.655296, 'X': 1.492512, 'Y': -2.465618},
            'BPMA.2097.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2075.020296, 'X': 1.706483, 'Y': -2.476357},
            'BPMA.2109.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2087.020296, 'X': 2.185077, 'Y': -2.480735},
            'BPMA.2124.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2101.520296, 'X': 2.763378, 'Y': -2.486025},
            'BPMA.2138.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2116.020296, 'X': 3.341679, 'Y': -2.491315},
            'BPMA.2153.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2130.520296, 'X': 3.919980, 'Y': -2.496605},
            'BPMA.2167.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2145.020296, 'X': 4.498282, 'Y': -2.501895},
            'BPMA.2179.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2156.920296, 'X': 4.972887, 'Y': -2.506237},
            'BPMA.2184.T1': {'MAD8_name': 'BPMA.T1',    'Line': 'T1', 'S': 2161.935296, 'X': 5.172900, 'Y': -2.508066},
            'BPME.2191.T1': {'MAD8_name': 'BPME.T1',    'Line': 'T1', 'S': 2168.336421, 'X': 5.428195, 'Y': -2.510402},
            'BPME.2197.T1': {'MAD8_name': 'BPME.T1',    'Line': 'T1', 'S': 2174.467921, 'X': 5.672736, 'Y': -2.512639},  # 18

            'BPMA.2071.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2048.710889, 'X': 0.000000, 'Y': -2.466747},
            'BPMA.2086.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2063.710889, 'X': 0.000000, 'Y': -2.472224},
            'BPMA.2101.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2078.710889, 'X': 0.000000, 'Y': -2.477700},
            'BPMA.2116.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2093.710989, 'X': 0.000000, 'Y': -2.483177},
            'BPMA.2132.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2109.410989, 'X': 0.000000, 'Y': -2.488910},
            'BPMA.2145.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2122.510989, 'X': 0.000000, 'Y': -2.493693},
            'BPMA.2161.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2138.710989, 'X': 0.000000, 'Y': -2.499608},
            'BPMA.2176.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2153.710989, 'X': 0.000000, 'Y': -2.505085},
            'BPMA.2191.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2168.711039, 'X': 0.000000, 'Y': -2.510562},
            'BPMA.2206.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2183.711039, 'X': 0.000000, 'Y': -2.516038},
            'BPMA.2218.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2195.511039, 'X': 0.000000, 'Y': -2.520347},
            'BPMA.2223.T2': {'MAD8_name': 'BPMA.T2',    'Line': 'T2', 'S': 2200.372856, 'X': 0.000000, 'Y': -2.522122},
            'BPME.2229.T2': {'MAD8_name': 'BPME.T2',    'Line': 'T2', 'S': 2206.345089, 'X': 0.000000, 'Y': -2.524303},
            'BPME.2235.T2': {'MAD8_name': 'BPME.T2',    'Line': 'T2', 'S': 2212.476589, 'X': 0.000000, 'Y': -2.526541}  # 14
            }


def writeXmlFile(outputfilename="marin-daq.xml", DICT=BPM_DICT):
    first = "<DAQREQ>\n"
    TStart = "<TStart time='2023-11-22T16:00:00'/>\n"
    TStop = "<TStop time='2023-11-22T16:05:00'/>\n"
    Exp = "<Exp  name='linac'/>\n"
    CDir = "<CDir name='/daq/xfel/admtemp/'/>\n"
    Chan = "<Chan name='{}' dtype='47'/>\n"
    last = "</DAQREQ>\n"

    f = open(outputfilename, "w")
    for head in [first, TStart, TStop, Exp, CDir]:
        f.write(head)
    for bpm in list(DICT.keys()):
        f.write(Chan.format('XFEL.DIAG/BPM/{}'.format(bpm)))
    f.write(Chan.format('XFEL.SDIAG/BAM.DAQ/1932M.TL.ARRIVAL_TIME.RELATIVE'))
    f.write(Chan.format('XFEL.SDIAG/BAM.DAQ/1932S.TL.ARRIVAL_TIME.RELATIVE'))
    f.write(Chan.format('XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY.ALL'))
    f.write(last)
    f.close()


def crisp(outputfilename='crisp_bunch_{}_for_{}_trains.pk', bunch=1, nbtrain=100):
    import pydoocs
    bunch_read = pydoocs.read(bl_number_adress)['data']
    if bunch != bunch_read:
        pydoocs.write(bl_number_adress, bunch)
    print('Crisp read for bunch {} and {} trains'.format(pydoocs.read(bl_number_adress)['data'], nbtrain))
    datadict = {'Time': pydoocs.read(bl_time_adress)['data']}
    TrainList = _np.linspace(1, nbtrain, nbtrain)
    for train in TrainList:
        datadict['Train{}'.format(int(train))] = pydoocs.read(bl_current_adress)['data']
        t.sleep(0.001)

    df = _pd.DataFrame(datadict)
    df.to_pickle(outputfilename.format(bunch, nbtrain))


def findUnmatchedTrainIDinBPMs(rawdata):
    bpmdata = rawdata['XFEL.DIAG']['BPM']
    firstbpmtrains = rawdata['XFEL.DIAG']['BPM'][list(bpmdata.keys())[0]]['TrainId'][:]
    L = _np.array([])
    for bpm in bpmdata.keys():
        bpmtrains = bpmdata[bpm]['TrainId'][:]
        L = _np.append(L, _np.setdiff1d(firstbpmtrains, bpmtrains))
        L = _np.append(L, _np.setdiff1d(bpmtrains, firstbpmtrains))
    return L


def findUnmatchedTrainID(inputfilename):
    rawdata = _h5.File(inputfilename, 'r')

    bpmdata = rawdata['XFEL.DIAG']['BPM']
    energydata = rawdata['XFEL.DIAG']['BEAM_ENERGY_MEASUREMENT']
    timedata = rawdata['XFEL.SDIAG']['BAM.DAQ']

    bpmtrains = bpmdata['BPMI.1860.TL']['TrainId'][:]
    energytrains = energydata['CL']['ENERGY.ALL']['TrainId'][:]
    timetrains = timedata['1932S.TL.ARRIVAL_TIME.RELATIVE']['TrainId'][:]

    L = findUnmatchedTrainIDinBPMs(rawdata)
    L = _np.append(L, _np.setdiff1d(bpmtrains, energytrains))
    L = _np.append(L, _np.setdiff1d(energytrains, bpmtrains))
    L = _np.append(L, _np.setdiff1d(bpmtrains, timetrains))
    L = _np.append(L, _np.setdiff1d(timetrains, bpmtrains))
    L = _np.append(L, _np.setdiff1d(timetrains, energytrains))
    L = _np.append(L, _np.setdiff1d(energytrains, timetrains))
    unmatched_train_ids = [int(l) for l in list(_np.unique(L))]

    rawdata.close()

    return unmatched_train_ids


def removeUnmatchedTrainID(inputfilename, outputfilename=None):
    if outputfilename is None:
        outputfilename = 'matched_' + inputfilename

    rawdata = _h5.File(inputfilename, 'r')
    newdata = _h5.File(outputfilename, 'w')

    bpmdata = rawdata['XFEL.DIAG']['BPM']
    energydata = rawdata['XFEL.DIAG']['BEAM_ENERGY_MEASUREMENT']
    timedata = rawdata['XFEL.SDIAG']['BAM.DAQ']

    bpmtrains = bpmdata['BPMI.1860.TL']['TrainId'][:]
    energytrains = energydata['CL']['ENERGY.ALL']['TrainId'][:]
    timetrains = timedata['1932S.TL.ARRIVAL_TIME.RELATIVE']['TrainId'][:]

    nbbpm, nbtrainmax, nbbunchmax = getNbBPMTrainsBunches(rawdata)
    unmatched_train_ids = findUnmatchedTrainID(inputfilename)

    copyGroupsAndDatasets(rawdata, newdata)
    copyGroupsAndDatasets(rawdata['XFEL.DIAG'], newdata['XFEL.DIAG'])
    copyGroupsAndDatasets(rawdata['XFEL.SDIAG'], newdata['XFEL.SDIAG'])

    def copyOnlyMarchedTrainsByBranch(trainsids, rawdatabranch, newdatabranch):
        unmatched_train_indices = _np.array([])
        for train_id in unmatched_train_ids:
            train_index = _np.where(trainsids == train_id)[0]
            if len(train_index) == 1:
                unmatched_train_indices = _np.append(unmatched_train_indices, int(_np.where(trainsids == train_id)[0]))
        train_indices = _np.setdiff1d(_np.arange(len(trainsids)), unmatched_train_indices)
        copyAndSelectRecursive(rawdatabranch, newdatabranch, train_indices, _np.arange(nbbunchmax))

    copyOnlyMarchedTrainsByBranch(bpmtrains, bpmdata, newdata['XFEL.DIAG']['BPM'])
    copyOnlyMarchedTrainsByBranch(energytrains, energydata, newdata['XFEL.DIAG']['BEAM_ENERGY_MEASUREMENT'])
    copyOnlyMarchedTrainsByBranch(timetrains, timedata, newdata['XFEL.SDIAG']['BAM.DAQ'])

    rawdata.close()
    newdata.close()


def reduceH5FileByTrainBunch(inputfilename, outputfilename=None, trains=None, bunches=None):
    if outputfilename is None:
        outputfilename = 'reduced_' + inputfilename

    rawdata = _h5.File(inputfilename, 'r')
    newdata = _h5.File(outputfilename, 'w')

    nbbpmmax, nbtrainmax, nbbunchmax = getNbBPMTrainsBunches(rawdata)
    if trains is None:
        print(Warning("Train selection not provided. Using all {} trains".format(nbtrainmax)))
        trains = _np.arange(nbtrainmax)
    if bunches is None:
        print(Warning("Bunch selection not provided. Using all {} bunches".format(nbbunchmax)))
        bunches = _np.arange(nbbunchmax)

    copyAndSelectRecursive(rawdata, newdata, trains, bunches)
    rawdata.close()
    newdata.close()


def copyGroupsAndDatasets(rawdata, newdata):
    setAttributes(rawdata, newdata)
    for key in rawdata.keys():
        newdata.create_group(key)
        setAttributes(rawdata[key], newdata[key])


def copyAndSelectRecursive(rawdata, newdata, trains, bunches):
    setAttributes(rawdata, newdata)
    keys = rawdata.keys()
    print(rawdata.name)
    for key in keys:
        if type(rawdata[key]) == _h5._hl.group.Group:
            newdata.create_group(key)
            copyAndSelectRecursive(rawdata[key], newdata[key], trains, bunches)
        elif type(rawdata[key]) == _h5._hl.dataset.Dataset:
            rawarray = rawdata[key][:]
            if key in ['BUNCH_VALID.TD', 'CHARGE.TD', 'X.TD', 'Y.TD']:
                newdata.create_dataset(key, data=rawarray[trains, :][:, bunches], compression="gzip", compression_opts=9)
            elif key == 'TimeStamp':
                newdata.create_dataset(key, data=rawarray[trains, :], compression="gzip", compression_opts=9)
            elif key == 'TrainId':
                newdata.create_dataset(key, data=rawarray[trains], compression="gzip", compression_opts=9)
            elif key == 'Value':
                if rawdata.name in ['/XFEL.SDIAG/BAM.DAQ/1932S.TL.ARRIVAL_TIME.RELATIVE', '/XFEL.SDIAG/BAM.DAQ/1932M.TL.ARRIVAL_TIME.RELATIVE']:
                    newdata.create_dataset(key, data=rawarray[trains, :][:, bunches], compression="gzip", compression_opts=9)
                elif rawdata.name == '/XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY.ALL':
                    newdata.create_dataset(key, data=rawarray[trains], compression="gzip", compression_opts=9)
        setAttributes(rawdata[key], newdata[key])


def setAttributes(rawdata, newdata):
    names = list(rawdata.attrs.keys())
    values = list(rawdata.attrs.values())
    for name, value in zip(names, values):
        newdata.attrs.create(name, value)


def getNbBPMTrainsBunches(rawdata):
    bpmlist = list(rawdata['XFEL.DIAG']['BPM'].keys())
    try:
        nbtrain, nbbunch = rawdata['XFEL.DIAG']['BPM'][bpmlist[0]]['X.TD'].shape
    except ValueError:
        nbtrain = len(rawdata['XFEL.DIAG']['BPM'][bpmlist[0]]['TrainId'])
        if nbtrain == 1:
            nbbunch = len(rawdata['XFEL.DIAG']['BPM'][bpmlist[0]]['X.TD'])
        else:
            nbbunch = 1
    nbbpm = len(bpmlist)

    return nbbpm, nbtrain, nbbunch


def sortBPMListByS(bpmlist, bpmdict=BPM_DICT):
    slist = _np.array([])
    for bpm in bpmlist:
        slist = _np.append(slist, bpmdict[bpm]['S'])

    slist_sorted = [x for x, _ in sorted(zip(slist, bpmlist))]
    bpmlist_sorted = [y for _, y in sorted(zip(slist, bpmlist))]

    return slist_sorted, bpmlist_sorted


def getH5dataInDF(inputfilename, getPosition=True, getCharge=False, getEnergy=False, getTime=False):
    if len(findUnmatchedTrainID(inputfilename)) > 0:
        raise ValueError("Inconsistant Train IDs in file : {}".format(inputfilename))

    rawdata = _h5.File(inputfilename, 'r')
    bpmdata = rawdata['XFEL.DIAG']['BPM']
    energydata = rawdata['XFEL.DIAG']['BEAM_ENERGY_MEASUREMENT']['CL']['ENERGY.ALL']
    timedata = rawdata['XFEL.SDIAG']['BAM.DAQ']

    slist, bpmlist = sortBPMListByS(list(bpmdata.keys()))
    nbbpm, nbtrain, nbbunch = getNbBPMTrainsBunches(rawdata)
    data_dict = {}

    def storedata(data_dict, key, data, factor=1.0):
        try:
            data_dict[key].append(data[:]*factor)
        except:
            data_dict[key] = [data[:]*factor]

    for i, bpm in enumerate(bpmlist):
        _printProgressBar(i, nbbpm, prefix='Load {} | {} bpms, {} trains, {} bunches:'.format(inputfilename.split('/')[-1], nbbpm, nbtrain, nbbunch),
                          suffix='Complete', length=50)
        storedata(data_dict, 'Valid', bpmdata[bpm]['BUNCH_VALID.TD'])
        storedata(data_dict, 'S', _np.full((nbtrain, nbbunch), slist[i]))  # m
        if getPosition:
            storedata(data_dict, 'X', bpmdata[bpm]['X.TD'], factor=1e-3)  # mm converted in m
            storedata(data_dict, 'Y', bpmdata[bpm]['Y.TD'], factor=1e-3)  # mm converted in m
        if getCharge:
            storedata(data_dict, 'Charge', bpmdata[bpm]['CHARGE.TD'], factor=1e-9)  # nC converted to C
        if getTime:
            storedata(data_dict, 'Time', timedata['1932S.TL.ARRIVAL_TIME.RELATIVE']['Value'])  # ps
        if getEnergy:
            storedata(data_dict, 'E', _np.tile(energydata['Value'], (nbbunch, 1)).transpose(), factor=1e-3)  # MeV converted to GeV
    for key in data_dict:
        data_dict[key] = _np.asarray(data_dict[key]).flatten()
    df = _pd.DataFrame(data_dict, index=_pd.MultiIndex.from_product([range(s) for s in (nbbpm, nbtrain, nbbunch)], names=['BPM', 'TrainID', 'BunchID']))
    df.index.set_levels([bpmlist, bpmdata[bpmlist[0]]['TrainId']], level=[0, 1], inplace=True)
    rawdata.close()
    _printProgressBar(nbbpm, nbbpm, prefix='Load {} | {} bpms, {} trains, {} bunches:'.format(inputfilename.split('/')[-1], nbbpm, nbtrain, nbbunch),
                      suffix='Complete', length=50)
    return df


def reduceDFbyIndex(df, index, value):
    if type(index) == int:
        indexid = index
        indexname = df.index.names[0]
    elif type(index) == str:
        indexid = df.index.names.index(index)
        indexname = index
    else:
        raise TypeError('Unknown type {} for index value. Must be either int or str'.format(type(index)))

    try:
        value = df.index.levels[indexid][value]
    except IndexError:
        pass

    try:
        mask = df.index.get_level_values(indexname) == value
        df = df.loc[mask]
    except:
        mask = df.index.get_level_values(indexname).isin(value)
        df = df.loc[mask]
    return df


def reduceDFbyBPMTrainBunchByIndex(df, bpms=None, trains=None, bunches=None, valid=True):
    if valid:
        df = df[df['Valid'] == 1]
    if bpms is not None:
        df = reduceDFbyIndex(df, 'BPM', bpms)
    if trains is not None:
        df = reduceDFbyIndex(df, 'TrainID', trains)
    if bunches is not None:
        df = reduceDFbyIndex(df, 'BunchID', bunches)
    df.index = df.index.remove_unused_levels()
    return df


def checkTrainBunchConsistancy(df):
    # df = getH5dataInDF(inputfilename, getPosition=False)
    df_valid = reduceDFbyBPMTrainBunchByIndex(df)
    bpmids_count = df_valid.index.get_level_values(0).value_counts(sort=False).values
    nbpm = len(bpmids_count)
    # print('nbbpms = ', nbpm)
    trainids_count = df_valid.index.get_level_values(1).value_counts(sort=False).values
    ntrains = len(trainids_count)
    # print('nbtrains = ', ntrains)
    bunchids_count = df_valid.index.get_level_values(2).value_counts(sort=False).values
    nbunches = len(bunchids_count)
    # print('nbbunches = ', nbunches)

    A = bpmids_count/ntrains
    B = trainids_count
    C = bunchids_count/ntrains

    fig, ax = plotOptions(figsize=[14, 8], rows_colums=[3, 1])
    _plt.subplot(3, 1, 1)
    _plt.plot(_np.abs(A - A.round()))
    bpmnames = df_valid.index.get_level_values(0).unique().values
    ax[0].set_xticks(range(len(bpmnames)))
    ax[0].set_xticklabels(bpmnames, fontsize=8, rotation=45, ha='right')
    _plt.subplot(3, 1, 2)
    _plt.plot(B)
    ax[1].ticklabel_format(useOffset=False)
    _plt.subplot(3, 1, 3)
    _plt.plot(_np.abs(C - C.round()))


def getBunchPattern(df, refT1='BPMA.2097.T1', refT2='BPMA.2161.T2', sample=1, valid=True):
    df_reduced = reduceDFbyBPMTrainBunchByIndex(df, trains=1, valid=valid)
    df_T1 = df_reduced[df_reduced.index.get_level_values(0) == refT1]
    df_T2 = df_reduced[df_reduced.index.get_level_values(0) == refT2]
    bunchIDs_T1 = df_T1.index.get_level_values(2).unique().values
    bunchIDs_T2 = df_T2.index.get_level_values(2).unique().values
    bunchIDs_TL = df_reduced.index.get_level_values(2).unique().values
    bunchIDs_TL = _np.setdiff1d(bunchIDs_TL, bunchIDs_T1)
    bunchIDs_TL = _np.setdiff1d(bunchIDs_TL, bunchIDs_T2)

    return bunchIDs_TL[0::sample], bunchIDs_T1[0::sample], bunchIDs_T2[0::sample]


def buildPositionMatrix(df_reduced, coord):
    try:
        nb_trains = df_reduced.index.levshape[1]
        nb_bunches = df_reduced.index.levshape[2]
        M = df_reduced[coord].to_numpy().reshape((-1, nb_trains * nb_bunches)).transpose()
    except:
        nb_shots = df_reduced.index.levshape[1]
        M = df_reduced[coord].to_numpy().reshape((-1, nb_shots)).transpose()
    return M


def calcPositionMean(df_reduced, coord):
    M = buildPositionMatrix(df_reduced, coord)
    Mean = _np.array([])
    for i in range(len(M[0])):
        Mean = _np.append(Mean, _np.abs(_np.mean(M[:, i])))
    return Mean


def buildMatrixAndVectorForSVD(df, refbpmname, coord='X', meanSub=True):
    if coord in ['X', 'Y']:
        df_ref = df.loc[df.index.get_level_values('BPM') == refbpmname][['X', 'Y']]
        df_matrix = df.loc[df.index.get_level_values('BPM') != refbpmname][['X', 'Y']]

        M_X = buildPositionMatrix(df_matrix, 'X')
        M_Y = buildPositionMatrix(df_matrix, 'Y')
        Vect_ref = df_ref[coord].to_numpy()
        M = _np.concatenate((M_X, M_Y), axis=1)
    elif coord == 'Charge':
        df_ref = df.loc[df.index.get_level_values('BPM') == refbpmname][['Charge']]
        df_matrix = df.loc[df.index.get_level_values('BPM') != refbpmname][['Charge']]
        M = buildPositionMatrix(df_matrix, 'Charge')
        Vect_ref = df_ref[coord].to_numpy()
    else:
        raise ValueError('Unknown coordinate : {}'.format(coord))

    if meanSub:
        M = M - M.mean(0)

    return Vect_ref, M


def SVD(M):
    U, d, V_t = _np.linalg.svd(M, full_matrices=False)
    D = _np.diag(d)

    D_i = _np.linalg.inv(D)
    U_t = U.transpose()
    V = V_t.transpose()

    return U, D, V_t, U_t, D_i, V


def calcCoeffsWithSVD(M, ref_Vect):
    U, d, V_t = _np.linalg.svd(M, full_matrices=False)
    D = _np.diag(d)

    D_i = _np.linalg.inv(D)
    U_t = U.transpose()
    V = V_t.transpose()

    C = _np.dot(_np.dot(V, _np.dot(D_i, U_t)), ref_Vect)
    return C


def calcMeasuredPositionAndNResidual(M, ref_Vect):
    C = calcCoeffsWithSVD(M, ref_Vect)
    meas_Vect = _np.dot(M, C)
    Residual = ref_Vect - _np.dot(M, C)

    return meas_Vect, Residual


def calcJitterAndNoise(df, coord):
    Jitter = _np.array([])
    Noise = _np.array([])
    for bpm in df.index.get_level_values(0).unique():
        V, M = buildMatrixAndVectorForSVD(df, bpm, coord=coord)
        meas_Vect, Residual = calcMeasuredPositionAndNResidual(M, V)
        Jitter = _np.append(Jitter, meas_Vect.std())
        Noise = _np.append(Noise, Residual.std())

    return Jitter, Noise


def calcAvgPositionAndError(df, coord):
    AvgPos = _np.array([])
    AvgErr = _np.array([])
    for bpm in df.index.get_level_values(0).unique():
        V, M = buildMatrixAndVectorForSVD(df, bpm, coord=coord, meanSub=False)
        meas_Vect, Residual = calcMeasuredPositionAndNResidual(M, V)
        AvgPos = _np.append(AvgPos, meas_Vect.mean())
        AvgErr = _np.append(AvgErr, Residual.mean())

    return AvgPos, AvgErr


def calcResolution(df, coord):
    Resol_array = _np.array([])
    for bpm in df.index.get_level_values(0).unique():
        V, M = buildMatrixAndVectorForSVD(df, bpm, coord=coord)
        meas_Vect, Residual = calcMeasuredPositionAndNResidual(M, V)
        Resol = (meas_Vect - V).std()
        Resol_array = _np.append(Resol_array, Resol)

    return Resol_array


def matchJitterAndBeamSizeArray(S, Jitter_X, Jitter_Y, df_bpm, tolerence=0.001):
    if len(S) == len(df_bpm.S):
        return S, Jitter_X, Jitter_Y
    else:
        for i, s in enumerate(df_bpm.S):
            if _np.abs(s - S[i]) > tolerence:
                S = _np.delete(S, i)
                Jitter_X = _np.delete(Jitter_X, i)
                Jitter_Y = _np.delete(Jitter_Y, i)
        return S, Jitter_X, Jitter_Y


def calcJitterSigmaRatio(df, twiss, Jitter_X, Jitter_Y):
    S = df.S.unique()
    df_cut = twiss.data[twiss.data.S.between(min(S), max(S))]
    df_bpm = df_cut[df_cut.TYPE == 'MONI']
    S_match, Jitter_X_matched, Jitter_Y_matched = matchJitterAndBeamSizeArray(S, Jitter_X, Jitter_Y, df_bpm)
    return S_match, Jitter_X_matched/df_bpm.SIGX*100, Jitter_Y_matched/df_bpm.SIGY*100


def calcJitterAverageForOneBunch(df, twiss, Smin=None, Smax=None, SigmaRatio=False):
    S = df.S.unique()
    Jitter_X, Noise_X = calcJitterAndNoise(df, 'X')
    Jitter_Y, Noise_Y = calcJitterAndNoise(df, 'Y')

    if SigmaRatio:
        S, Jitter_X, Jitter_Y = calcJitterSigmaRatio(df, twiss, Jitter_X, Jitter_Y)

    if Smin is not None:
        index_min = (_np.abs(S - Smin)).argmin()
    else:
        index_min = 0
    if Smax is not None:
        index_max = (_np.abs(S - Smax)).argmin()
    else:
        index_max = len(S)
    return _np.mean(Jitter_X[index_min:index_max]), _np.mean(Jitter_Y[index_min:index_max])


def openAllMAD8Files(ex=3.58e-11, ey=3.58e-11, esprd=1e-6):
    twiss_tld = _m8.Output('../01_mad8/XFEL_Lattice_9/TWISS_TLD')
    twiss_tld.calcBeamSize(ex, ey, esprd)
    twiss_t1 = _m8.Output('../01_mad8/XFEL_Lattice_9/TWISS_T5D')
    twiss_t1.calcBeamSize(ex, ey, esprd)
    twiss_t2 = _m8.Output('../01_mad8/XFEL_Lattice_9/TWISS_T4D')
    twiss_t2.calcBeamSize(ex, ey, esprd)
    return twiss_tld, twiss_t1, twiss_t2


def selectRightMAD8File(bunch, bunchIDs_TL, bunchIDs_T1, bunchIDs_T2, twiss_tld, twiss_t1, twiss_t2):
    if bunch in bunchIDs_TL:
        return twiss_tld
    elif bunch in bunchIDs_T1:
        return twiss_t1
    elif bunch in bunchIDs_T2:
        return twiss_t2
    else:
        raise ValueError('Bunch ID {} not in the bunch pattern'. format(bunch))


def calcJitterAverageForMultipleBunches(df, trains=None, Smin=None, Smax=None, sample=1, ex=3.58e-11, ey=3.58e-11, esprd=1e-6, SigmaRatio=False):
    bunchIDs_TL, bunchIDs_T1, bunchIDs_T2 = getBunchPattern(df, sample=sample)
    bunchIDs = _np.append(_np.append(bunchIDs_TL, bunchIDs_T1), bunchIDs_T2)
    bunchIDs.sort()
    nbbunch = len(bunchIDs)

    twiss_tld, twiss_t1, twiss_t2 = openAllMAD8Files(ex=ex, ey=ey, esprd=esprd)

    Jitter_X_mean_list = _np.array([])
    Jitter_Y_mean_list = _np.array([])
    unused_bunches = _np.array([])
    for i, bunch in enumerate(bunchIDs):
        _printProgressBar(i, nbbunch, prefix='Caluclate average jitter for bunch {}/{}:'.format(i, nbbunch), suffix='Complete', length=50)
        df_reduced = reduceDFbyBPMTrainBunchByIndex(df, trains=trains, bunches=bunch)
        twiss = selectRightMAD8File(bunch, bunchIDs_TL, bunchIDs_T1, bunchIDs_T2, twiss_tld, twiss_t1, twiss_t2)
        try:
            Jitter_X_mean, Jitter_Y_mean = calcJitterAverageForOneBunch(df_reduced, twiss, Smin=Smin, Smax=Smax, SigmaRatio=SigmaRatio)
            Jitter_X_mean_list = _np.append(Jitter_X_mean_list, Jitter_X_mean)
            Jitter_Y_mean_list = _np.append(Jitter_Y_mean_list, Jitter_Y_mean)
        except:
            print('bunch {} not working'.format(bunch))
            unused_bunches = _np.append(unused_bunches, bunch)
    bunch_list = _np.setdiff1d(bunchIDs, unused_bunches)
    _printProgressBar(nbbunch, nbbunch, prefix='Caluclate average jitter for bunch {}/{}:'.format(nbbunch, nbbunch), suffix='Complete', length=50)
    return bunch_list, Jitter_X_mean_list, Jitter_Y_mean_list


def calcPositionMatrix(df, coord):
    for i, bpm in enumerate(df.index.get_level_values(0).unique()):
        V, M = buildMatrixAndVectorForSVD(df, bpm, coord=coord)
        meas_Vect, Residual = calcMeasuredPositionAndNResidual(M, V)
        try:
            Position_Matrix = _np.concatenate((Position_Matrix, [meas_Vect]))
        except:
            Position_Matrix = _np.array([meas_Vect])
    return Position_Matrix


def calcAngleMatrix(df, coord):
    Position_Matrix = calcPositionMatrix(df, coord)
    S = df.S.unique()
    for i in range(len(Position_Matrix)-1):
        Position_diff = Position_Matrix[i]-Position_Matrix[i+1]
        S_diff = _np.abs(S[i]-S[i+1])
        Angle_Vector = _np.tan(Position_diff/S_diff)
        try:
            Angle_Matrix = _np.concatenate((Angle_Matrix, [Angle_Vector]))
        except:
            Angle_Matrix = _np.array([Angle_Vector])
    return Angle_Matrix


def calcMaxAngle(df, coord):
    S = df.S.unique()
    Angle_Matrix = calcAngleMatrix(df, coord)
    Angle_Max = _np.array([])
    for Angle_Vector in Angle_Matrix:
        Angle_Max = _np.append(Angle_Max, max(_np.abs(Angle_Vector)))
    return S[1:], Angle_Max


def plotBunchPattern(df, sample=1, figsize=[14, 4]):
    bunchIDs_TL, bunchIDs_T1, bunchIDs_T2 = getBunchPattern(df, sample=sample)
    fig, ax = plotOptions(figsize=figsize)
    _plt.plot(bunchIDs_TL, bunchIDs_TL * 0, '+', color='C0', markersize=8, markeredgewidth=1, label='TLD')
    _plt.plot(bunchIDs_T1, bunchIDs_T1 * 0 + 1, '+', color='C2', markersize=8, markeredgewidth=1, label='T1')
    _plt.plot(bunchIDs_T2, bunchIDs_T2 * 0 + 2, '+', color='C3', markersize=8, markeredgewidth=1, label='T2')
    _plt.ylabel('Path')
    _plt.xlabel('Bunch ID')
    _plt.legend()


def plotJitterAndNoise(df, twissfile, bpms=None, trains=None, bunches=None,
                       ex=3.58e-11, ey=3.58e-11, esprd=1e-6, height_ratios=None, xlim=None, font_size=17,
                       plotJitter=True, plotAngle=False, plotSigma=False, plotBeta=False, plotDisp=False, plotNoise=False, plotMean=False, figsize=[14, 6]):
    df_reduced = reduceDFbyBPMTrainBunchByIndex(df, bpms=bpms, trains=trains, bunches=bunches)
    if xlim:
        df_reduced = df_reduced[df_reduced.S.between(xlim[0], xlim[1])]
    S = df_reduced.S.unique()
    Jitter_X, Noise_X = calcJitterAndNoise(df_reduced, 'X')
    Jitter_Y, Noise_Y = calcJitterAndNoise(df_reduced, 'Y')

    twiss = _m8.Output(twissfile)
    twiss.calcBeamSize(ex, ey, esprd)
    df_cut = twiss.data[twiss.data.S.between(min(S), max(S))]

    rows_colums = [plotJitter+plotAngle+plotSigma+plotBeta+plotDisp+plotNoise+plotMean, 1]
    fig, ax = plotOptions(figsize=figsize, rows_colums=rows_colums, sharex='all', height_ratios=height_ratios, font_size=font_size)

    spnum = 0
    if plotJitter:
        spnum += 1
        _plt.subplot(rows_colums[0], rows_colums[1], spnum)
        plot2CurvesSameAxis(S, Jitter_X, Jitter_Y, ls1='+-', ls2='+-', legend1=r'$\sigma_{J,X}$', legend2=r'$\sigma_{J,Y}$', labelX='$S$ [m]', labelY='Jitter [m]')
    if plotAngle:
        spnum += 1
        _plt.subplot(rows_colums[0], rows_colums[1], spnum)
        S_angle, Angle_Max_X = calcMaxAngle(df_reduced, 'X')
        S_angle, Angle_Max_Y = calcMaxAngle(df_reduced, 'Y')
        plot2CurvesSameAxis(S_angle, Angle_Max_X, Angle_Max_Y, ls1='+-', ls2='+-', legend1=r'$max(\theta_X)$', legend2=r'$max(\theta_Y)$', labelX='$S$ [m]', labelY='Max angle')
    if plotSigma:
        spnum += 1
        _plt.subplot(rows_colums[0], rows_colums[1], spnum)
        S_match, JitterX_SigX, JitterY_SigY = calcJitterSigmaRatio(df_reduced, twiss, Jitter_X, Jitter_Y)
        plot2CurvesSameAxis(S_match, JitterX_SigX, JitterY_SigY, ls1='+-', ls2='+-',
                            legend1=r'$\frac{\sigma_{J,X}}{\sigma_X}$', legend2=r'$\frac{\sigma_{J,Y}}{\sigma_Y}$',
                            labelX='$S$ [m]', labelY='Jitter/sigma', ticksType='plain')
        _plt.hlines([5], min(S_match), max(S_match), ls='--', colors='C3')
        ax[spnum-1].yaxis.set_major_formatter(mtick.PercentFormatter())
    if plotBeta:
        spnum += 1
        _plt.subplot(rows_colums[0], rows_colums[1], spnum)
        plot2CurvesSameAxis(df_cut.S, df_cut.BETX, df_cut.BETY, ls1='-', ls2='-', legend1=r'$\beta_X$', legend2=r'$\beta_Y$', labelX='$S$ [m]', labelY=r'$\beta$ [m]')
    if plotDisp:
        spnum += 1
        _plt.subplot(rows_colums[0], rows_colums[1], spnum)
        plot2CurvesSameAxis(df_cut.S, df_cut.DX, df_cut.DY, ls1='--', ls2='--', legend1='$D_X$', legend2='$D_Y$', labelX='$S$ [m]', labelY='Disp [m]')
    if plotNoise:
        spnum += 1
        _plt.subplot(rows_colums[0], rows_colums[1], spnum)
        plot2CurvesSameAxis(S, Noise_X, Noise_Y, ls1='+-', ls2='+-', legend1=r'$\sigma_{N,X}$', legend2=r'$\sigma_{N,Y}$', labelX='$S$ [m]', labelY='Noise [m]')
    if plotMean:
        spnum += 1
        _plt.subplot(rows_colums[0], rows_colums[1], spnum)
        X_bar = calcPositionMean(df_reduced, 'X')
        Y_bar = calcPositionMean(df_reduced, 'Y')
        plot2CurvesSameAxis(S, X_bar, Y_bar, ls1='+-', ls2='+-', legend1='$mean(|X|)$', legend2='$mean(|Y|)$', labelX='$S$ [m]', labelY='Mean [m]')

    fig.align_labels()
    _m8.Plot.AddMachineLatticeToFigure(fig, twiss)
    _plt.xlim(min([min(S), min(df_cut.S)]) - 2, max([max(S), max(df_cut.S)]) + 2)
    if xlim:
        _plt.xlim(xlim[0], xlim[1])


def plotResolution(df, twissfile, trains=None, bunches=None, ylog=False, ex=3.58e-11, ey=3.58e-11, esprd=1e-6, figsize=[14, 6]):
    df_reduced = reduceDFbyBPMTrainBunchByIndex(df, trains=trains, bunches=bunches)
    S = df_reduced.S.unique()
    Res_X = calcResolution(df_reduced, 'X')
    Res_Y = calcResolution(df_reduced, 'Y')

    twiss = _m8.Output(twissfile)
    twiss.calcBeamSize(ex, ey, esprd)
    df_cut = twiss.data[twiss.data.S.between(min(S), max(S))]

    fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 1])

    plot2CurvesSameAxis(S, Res_X, Res_Y, ls1='+-', ls2='+-', legend1=r'$R_{X}$', legend2=r'$R_{Y}$', labelX='$S$ [m]', labelY='Resolution [m]')
    fig.align_labels()
    if ylog:
        _plt.yscale('log')
    _m8.Plot.AddMachineLatticeToFigure(fig, twiss)
    _plt.xlim(min([min(S), min(df_cut.S)]) - 2, max([max(S), max(df_cut.S)]) + 2)


def plotJitterAverageForAllBunches(df, ex=3.58e-11, ey=3.58e-11, esprd=1e-6, trains=None, Smin=None, Smax=None,
                                   SigmaRatio=False, sample=1, figsize=[14, 6]):
    bunchIDs_TL, bunchIDs_T1, bunchIDs_T2 = getBunchPattern(df, sample=sample)

    bunchIDs_cut, Jitter_X_mean, Jitter_Y_mean = calcJitterAverageForMultipleBunches(df, trains=trains, Smin=Smin, Smax=Smax, sample=sample,
                                                                                     ex=ex, ey=ey, esprd=esprd, SigmaRatio=SigmaRatio)

    fig, ax = plotOptions(figsize=figsize)
    _plt.vlines(x=list(bunchIDs_T1), ymin=min(min(Jitter_X_mean), min(Jitter_Y_mean)), ymax=max(max(Jitter_X_mean), max(Jitter_Y_mean)),
                colors='C2', alpha=0.5, label='T1')
    _plt.vlines(x=list(bunchIDs_T2), ymin=min(min(Jitter_X_mean), min(Jitter_Y_mean)), ymax=max(max(Jitter_X_mean), max(Jitter_Y_mean)),
                colors='C3', alpha=0.5, label='T2')
    if SigmaRatio:
        plot2CurvesSameAxis(bunchIDs_cut, Jitter_X_mean, Jitter_Y_mean, ls1='-', ls2='-', labelX='bunchID', labelY='Average ratio Jitter Beam size',
                            legend1=r'$\frac{\sigma_{J,X}}{\sigma_X}$', legend2=r'$\frac{\sigma_{J,Y}}{\sigma_Y}$', color1='C0', color2='C1',
                            markersize=8, markeredgewidth=1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    else:
        plot2CurvesSameAxis(bunchIDs_cut, Jitter_X_mean, Jitter_Y_mean, ls1='-', ls2='-', labelX='bunchID', labelY='Average Jitter [m]',
                            legend1=r'$\sigma_{J,X}$', legend2=r'$\sigma_{J,Y}$', color1='C0', color2='C1', markersize=8, markeredgewidth=1)
    _plt.legend()


def plot1Curve(X, Y, labelX='X', labelY='Y', legend='label', ls='-', color='C0',
               markersize=15, markeredgewidth=2, ticksType='sci', printLegend=True):
    _plt.plot(X, Y, ls, color=color, markersize=markersize, markeredgewidth=markeredgewidth, label=legend)
    _plt.ylabel(labelY)
    _plt.xlabel(labelX)
    _plt.ticklabel_format(axis="y", style=ticksType, scilimits=(0, 0))
    if printLegend:
        _plt.legend()


def plot2CurvesSameAxis(X, Y1, Y2, labelX='X', labelY='Y', legend1='Y1', legend2='Y2',
                        ls1='-', ls2='-', color1='C0', color2='C1', markersize=15, markeredgewidth=2, ticksType='sci', printLegend=True):
    _plt.plot(X, Y1, ls1, color=color1, markersize=markersize, markeredgewidth=markeredgewidth, label=legend1)
    _plt.plot(X, Y2, ls2, color=color2, markersize=markersize, markeredgewidth=markeredgewidth, label=legend2)
    _plt.ylabel(labelY)
    _plt.xlabel(labelX)
    _plt.ticklabel_format(axis="y", style=ticksType, scilimits=(0, 0))
    if printLegend:
        _plt.legend()


def plotDifferentTraj(data, bpm=None, train=None, bunch=None, valid=True):
    data_reduced = reduceDFbyBPMTrainBunchByIndex(data, bpms=bpm, trains=train, bunches=bunch, valid=valid)
    for bunch in data_reduced.index.get_level_values(2).to_numpy():
        data_bunch = reduceDFbyBPMTrainBunchByIndex(data_reduced, bunches=bunch)
        S = data_bunch.S
        Y = data_bunch.Y
        _plt.plot(S, Y, '+')
    _plt.ylabel('$Y$ [m]')
    _plt.xlabel('$S$ [m]')


def plotTrajectory(data, bpm=None, train=None, bunch=None, valid=True):
    data_reduced = reduceDFbyBPMTrainBunchByIndex(data, bpms=bpm, trains=train, bunches=bunch, valid=valid)
    data_reduced = data_reduced.sort_values(by='S')
    S = data_reduced.S
    X = data_reduced.X
    Y = data_reduced.Y
    Charge = data_reduced.Charge

    plotOptions()
    ax1 = _plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(S, X, '+-', color='C0', markersize=15, markeredgewidth=2, label='$X$')
    ax1.plot(S, Y, '+-', color='C1', markersize=15, markeredgewidth=2, label='$Y$')
    ax2.plot(S, Charge, '+', color='C2', markersize=15, markeredgewidth=2, label='$Charge$')
    ax1.set_ylabel('$X/Y$ [m]')
    ax2.set_ylabel('$Charge$ [C]')
    ax1.set_xlabel('$S$ [m]')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _plt.legend(h1 + h2, l1 + l2)


def plotHistogram(data, bpm=None, train=None, bunch=None, valid=True, bins=10):
    data_reduced = reduceDFbyBPMTrainBunchByIndex(data, bpms=bpm, trains=train, bunches=bunch, valid=valid)
    X = data_reduced.X
    Y = data_reduced.Y
    Charge = data_reduced.Charge

    plotOptions(rows_colums=[1, 3])
    _plt.subplot(1, 3, 1)
    _plt.hist(X, bins=bins, histtype='step', color='C0', label='$X : std = {:1.2e} m$'.format(_np.std(X)))
    _plt.xlabel('$X$ [m]')
    _plt.legend()
    _plt.subplot(1, 3, 2)
    _plt.hist(Y, bins=bins, histtype='step', color='C1', label='$Y : std = {:1.2e} m$'.format(_np.std(Y)))
    _plt.xlabel('$Y$ [m]')
    _plt.legend()
    _plt.subplot(1, 3, 3)
    _plt.hist(Charge, bins=bins, histtype='step', color='C2', label='$Charge : std = {:1.2e} C$'.format(_np.std(Charge)))
    _plt.xlabel('$Charge$ [C]')
    _plt.legend()


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
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
