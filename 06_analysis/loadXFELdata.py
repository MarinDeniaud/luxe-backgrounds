import h5py as _h5
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd


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
        length = timemax - timemin

        if plotCumsum:
            _plt.plot(time, cumsum, '+', color='C0')
            _plt.hlines([1-percentile, percentile], xmin=min(time), xmax=max(time), colors=['C3', 'C3'], linestyles='--')
            _plt.vlines([timemin, timemax], ymin=min(cumsum), ymax=max(cumsum), colors=['C3', 'C3'], linestyles='--')
            _plt.ylabel('cumsum')
            _plt.xlabel('$t$ [fs]')
            _plt.legend([r"$\sigma_t$ = {:.2f} fs".format(length), "{} percentiles".format(percentile)])

        return length

    def calcLengthAllTrains(self, percentile=0.95):
        Lengths = _np.array([])
        trainIDs = range(len(self.df.keys())-1)
        for trainID in trainIDs:
            Lengths = _np.append(Lengths, self.calcLengthOneTrainCUMSUM(trainID, percentile=percentile))
        return trainIDs, Lengths


class XFELdata:
    def __init__(self, inputfilename, excelfilename="../01_mad8/component_list_2023.07.02.xls",
                 TLD_bpms=['BPMA.1995.TL', 'BPMA.2011.TL', 'BPMD.2022.TL', 'BPMA.2041.TL', 'BPMA.2054.TL'],
                 EmitX=3.58e-11, EmitY=3.58e-11, Esprd=6e-4, E0=16, getPosition=True, getCharge=False, getEnergy=False, getTime=False):
        print("Loaded file '{}'".format(inputfilename.split('/')[-1]))
        self.inputfilename = inputfilename
        self.bpm_adress = "XFEL.DIAG/BPM"
        self.energy_adress = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/CL/ENERGY.ALL"
        self.time_S_adress = "XFEL.SDIAG/BAM.DAQ/1932S.TL.ARRIVAL_TIME.RELATIVE"
        self.time_M_adress = "XFEL.SDIAG/BAM.DAQ/1932M.TL.ARRIVAL_TIME.RELATIVE"

        self.E0 = E0

        self.rawdata = _h5.File(inputfilename, 'r')
        self.bpmdata = self.rawdata[self.bpm_adress]
        self.energydata = self.rawdata[self.energy_adress]
        self.timeSdata = self.rawdata[self.time_S_adress]
        self.timeMdata = self.rawdata[self.time_M_adress]

        self.bpmIDs = _np.array(list(self.bpmdata.keys()))
        self.nbbpm = len(self.bpmIDs)
        self.bpmIDs_TLD = _np.array(TLD_bpms)
        self.bpmIDs_TL = _np.setdiff1d([bpmID for bpmID in self.bpmIDs if 'TL' in bpmID], self.bpmIDs_TLD)
        self.bpmIDs_T1 = _np.array([bpmID for bpmID in self.bpmIDs if 'T1' in bpmID])
        self.bpmIDs_T2 = _np.array([bpmID for bpmID in self.bpmIDs if 'T2' in bpmID])

        self.trainIDs_raw = self.bpmdata[self.bpmIDs[0]]['TrainId'][:]
        self.trainIDs_matched = _np.setdiff1d(self.trainIDs_raw, self.getUnmatchedTrainID())
        self.nbtrain = len(self.trainIDs_matched)

        self.nbbunch = self.bpmdata[self.bpmIDs[0]]['X.TD'].shape[1]
        self.bunchIDs = _np.array(range(self.nbbunch))

        df_excel_T1 = _pd.read_excel(excelfilename, sheet_name='I1toT5D')
        df_excel_T1_bpm = df_excel_T1[df_excel_T1['NAME1'].isin(self.bpmIDs)]
        df_excel_T2 = _pd.read_excel(excelfilename, sheet_name='I1toT4D')
        df_excel_T2_bpm = df_excel_T2[df_excel_T2['NAME1'].isin(self.bpmIDs)]

        self.df_excel = _pd.concat((df_excel_T1_bpm, df_excel_T2_bpm[df_excel_T2['NAME1'].isin(df_excel_T1['NAME1']) == False]))
        self.calcBeamSize(EmitX, EmitY, Esprd)

        self.s_by_section = _np.array(self.df_excel.S)
        self.bpmIDs_by_section = _np.array(self.df_excel.NAME1)

        self.s_by_s = self.s_by_section[self.s_by_section.argsort()]
        self.bpmIDs_by_s = self.bpmIDs_by_section[self.s_by_section.argsort()]

        self.df_bpm = self.getH5dataInDF(getPosition=getPosition, getCharge=getCharge, getEnergy=getEnergy, getTime=getTime)

        self.bunchIDs_XTLD = self.reduceDFbyBPMTrainBunchByIndex(self.df_bpm, bpms="BPMI.1878.TL", trains=0).index.get_level_values('BunchID')
        self.bunchIDs_XTD1 = self.reduceDFbyBPMTrainBunchByIndex(self.df_bpm, bpms="BPMA.2088.T1", trains=0).index.get_level_values('BunchID')
        self.bunchIDs_XTD2 = self.reduceDFbyBPMTrainBunchByIndex(self.df_bpm, bpms="BPMA.2132.T2", trains=0).index.get_level_values('BunchID')

    def calcBeamSize(self, EmitX, EmitY, Esprd):
        SigmaX = []
        SigmaY = []
        SigmaXP = []
        SigmaYP = []
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
            SigmaX.append(_np.sqrt(BetaX * EmitX + (DispX * Esprd / self.E0) ** 2))
            SigmaY.append(_np.sqrt(BetaY * EmitY + (DispY * Esprd / self.E0) ** 2))
            # Beam divergence calculation
            SigmaXP.append(_np.sqrt(GammaX * EmitX + (DispXP * Esprd / self.E0) ** 2))
            SigmaYP.append(_np.sqrt(GammaY * EmitY + (DispYP * Esprd / self.E0) ** 2))

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
            storedata(data_dict, 'S', _np.full((self.nbtrain, self.nbbunch), self.s_by_s[i]))  # m
            if getPosition:
                storedata(data_dict, 'X', self.bpmdata[bpm]['X.TD'][mask], factor=1e-3)  # mm converted in m
                storedata(data_dict, 'Y', self.bpmdata[bpm]['Y.TD'][mask], factor=1e-3)  # mm converted in m
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
        df.index = df.index.set_levels([self.bpmIDs_by_s, self.trainIDs_matched], level=[0, 1])
        _printProgressBar(self.nbbpm, self.nbbpm, prefix='Loading {} bpms, {} trains and {} bunches in df:'.format(self.nbbpm, self.nbtrain, self.nbbunch),
                          suffix='Complete', length=50)
        return df

    def reduceDFbyIndex(self, index, value, df=None):
        if df is None:
            df = self.df_bpm
        if type(index) is int:
            indexid = index
        elif type(index) is str:
            indexid = df.index.names.index(index)
        else:
            raise TypeError('Unknown type {} for index level. Must be either int or str'.format(type(index)))

        if type(value) is int:
            value = df.index.levels[indexid][value]

        if type(value) in [list, _np.ndarray]:
            mask = df.index.get_level_values(indexid).isin(value)
        else:
            mask = df.index.get_level_values(indexid) == value
        return df.loc[mask]

    def reduceDFbyBPMTrainBunchByIndex(self, df=None, bpms=None, trains=None, bunches=None, valid=True):
        if df is None:
            df = self.df_bpm
        if valid:
            df = df[df['Valid'] == 1]
        if bpms is not None:
            df = self.reduceDFbyIndex('BPM', bpms, df=df)
        if trains is not None:
            df = self.reduceDFbyIndex('TrainID', trains, df=df)
        if bunches is not None:
            df = self.reduceDFbyIndex('BunchID', bunches, df=df)
        df.index = df.index.remove_unused_levels()
        return df

    def reduceDFbyPath(self, path='T1', valid=True):
        match path:
            case 'TLD':
                bpms = _np.append(self.bpmIDs_TL, self.bpmIDs_TLD)
            case 'T1':
                bpms = _np.append(self.bpmIDs_TL, self.bpmIDs_T1)
            case 'T2':
                bpms = _np.append(self.bpmIDs_TL, self.bpmIDs_T2)
            case _:
                raise ValueError('Unknown value {} for path'.format(path))

        df = self.reduceDFbyBPMTrainBunchByIndex(bpms=bpms, valid=valid)
        df.index = df.index.remove_unused_levels()
        return df

    def selectBunchByPath(self, path='T1', valid=True):
        match path:
            case 'TLD':
                pass

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

    def getBunchPattern(self, refT1='BPMA.2097.T1', refT2='BPMA.2161.T2', train=1, sample=1):
        df_reduced = self.reduceDFbyBPMTrainBunchByIndex(trains=train)
        df_T1 = df_reduced[df_reduced.index.get_level_values(0) == refT1]
        df_T2 = df_reduced[df_reduced.index.get_level_values(0) == refT2]
        bunchIDs_T1 = df_T1.index.get_level_values(2).unique().values
        bunchIDs_T2 = df_T2.index.get_level_values(2).unique().values
        bunchIDs_TL = df_reduced.index.get_level_values(2).unique().values
        bunchIDs_TL = _np.setdiff1d(bunchIDs_TL, bunchIDs_T1)
        bunchIDs_TL = _np.setdiff1d(bunchIDs_TL, bunchIDs_T2)

        return bunchIDs_TL[0::sample], bunchIDs_T1[0::sample], bunchIDs_T2[0::sample]

    def getS(self, df):
        s = df.xs(df.index.get_level_values('TrainID')[0], level='TrainID').S.to_numpy()
        return s

    def calcSignal(self, df):
        bpm_names = df.index.get_level_values('BPM').unique()
        sx = []
        sy = []
        for i, bpm_name in enumerate(bpm_names):
            df_bpm = df.xs(bpm_name, level='BPM')
            sx.append(_np.std(df_bpm.X))
            sy.append(_np.std(df_bpm.Y))
        return _np.array(sx), _np.array(sy)

    def calcAverageJitterPerBunchID(self, bpms=['BPMI.1910.TL', 'BPMI.1925.TL', 'BPMI.1930.TL', 'BPMI.1939.TL']):
        df_red = self.reduceDFbyBPMTrainBunchByIndex(df=self.df_bpm, bpms=bpms)
        bunchIDs = []
        sx_mean = []
        sy_mean = []
        for bunchID in self.bunchIDs_XTLD[::10]:
            print("{}/{} bunches".format(bunchID, len(self.bunchIDs_XTLD)), end='\r')
            df = self.reduceDFbyBPMTrainBunchByIndex(df_red, bunches=bunchID)
            jx, nx = self.calcJitterAndNoise(df, 'X')
            jy, ny = self.calcJitterAndNoise(df, 'Y')
            s, sx, sy = self.calcJitterSigmaRatio(df, jx, jy)
            sx_mean.append(sx.mean())
            sy_mean.append(sy.mean())
            bunchIDs.append(bunchID)
            print("{}/{} bunches".format(len(self.bunchIDs_XTLD), len(self.bunchIDs_XTLD)), end='\r')
        return bunchIDs, sx_mean, sy_mean

    def calcJitterAndNoise(self, df, coord, meanSub=True):
        Jitter = _np.array([])
        Noise = _np.array([])
        for bpm in df.index.get_level_values(0).unique():
            V, M = self.buildMatrixAndVectorForSVD(df, bpm, coord=coord, meanSub=meanSub)
            meas_Vect, Residual = self.calcMeasuredPositionAndNResidual(M, V)
            Jitter = _np.append(Jitter, meas_Vect.std())
            Noise = _np.append(Noise, Residual.std())

        return Jitter, Noise

    def matchJitterAndBeamSizeArray(self, S, Jitter_X, Jitter_Y, df_bpm, tolerence=0.001):
        if len(S) == len(df_bpm.S):
            return S, Jitter_X, Jitter_Y
        else:
            for i, s in enumerate(df_bpm.S):
                if _np.abs(s - S[i]) > tolerence:
                    S = _np.delete(S, i)
                    Jitter_X = _np.delete(Jitter_X, i)
                    Jitter_Y = _np.delete(Jitter_Y, i)
            return S, Jitter_X, Jitter_Y

    def calcJitterSigmaRatio(self, df, Jitter_X, Jitter_Y):
        S = df.S.unique()
        df_excel = self.df_excel[self.df_excel.NAME1.isin(df.index.get_level_values('BPM').unique())]
        S_match, Jitter_X_matched, Jitter_Y_matched = self.matchJitterAndBeamSizeArray(S, Jitter_X, Jitter_Y, df_excel)
        return S_match, Jitter_X_matched / df_excel.SIGX * 100, Jitter_Y_matched / df_excel.SIGY * 100

    def calcCoeffsWithSVD(self, M, ref_Vect):
        U, d, V_t = _np.linalg.svd(M, full_matrices=False)
        D = _np.diag(d)

        D_i = _np.linalg.inv(D)
        U_t = U.transpose()
        V = V_t.transpose()

        C = _np.dot(_np.dot(V, _np.dot(D_i, U_t)), ref_Vect)
        return C

    def calcMeasuredPositionAndNResidual(self, M, ref_Vect):
        C = self.calcCoeffsWithSVD(M, ref_Vect)
        meas_Vect = _np.dot(M, C)
        Residual = ref_Vect - _np.dot(M, C)

        return meas_Vect, Residual

    def buildMatrixAndVectorForSVD(self, df, refbpmname, coord='X', meanSub=True):
        if coord in ['X', 'Y']:
            df_ref = df.loc[df.index.get_level_values('BPM') == refbpmname][['X', 'Y']]
            df_matrix = df.loc[df.index.get_level_values('BPM') != refbpmname][['X', 'Y']]

            M_X = self.buildPositionMatrix(df_matrix, 'X')
            M_Y = self.buildPositionMatrix(df_matrix, 'Y')
            Vect_ref = df_ref[coord].to_numpy()
            M = _np.concatenate((M_X, M_Y), axis=1)
        elif coord == 'Charge':
            df_ref = df.loc[df.index.get_level_values('BPM') == refbpmname][['Charge']]
            df_matrix = df.loc[df.index.get_level_values('BPM') != refbpmname][['Charge']]
            M = self.buildPositionMatrix(df_matrix, 'Charge')
            Vect_ref = df_ref[coord].to_numpy()
        else:
            raise ValueError('Unknown coordinate : {}'.format(coord))
        if meanSub:
            M = M - M.mean(0)
        return Vect_ref, M

    def buildPositionMatrix(self, df_reduced, coord):
        try:
            nb_trains = df_reduced.index.levshape[1]
            nb_bunches = df_reduced.index.levshape[2]
            M = df_reduced[coord].to_numpy().reshape((-1, nb_trains * nb_bunches)).transpose()
        except:
            nb_shots = df_reduced.index.levshape[1]
            M = df_reduced[coord].to_numpy().reshape((-1, nb_shots)).transpose()
        return M

    def plotBunchPattern(self, train=1, sample=1, figsize=[14, 4]):
        bunchIDs_TL, bunchIDs_T1, bunchIDs_T2 = self.getBunchPattern(train=train, sample=sample)
        fig, ax = plotOptions(figsize=figsize)
        _plt.plot(bunchIDs_TL, bunchIDs_TL * 0, '+', color='C0', markersize=8, markeredgewidth=1)
        _plt.plot(bunchIDs_T1, bunchIDs_T1 * 0 + 1, '+', color='C2', markersize=8, markeredgewidth=1)
        _plt.plot(bunchIDs_T2, bunchIDs_T2 * 0 + 2, '+', color='C3', markersize=8, markeredgewidth=1)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["XTLD", "XTD1", "XTD2"])
        # _plt.ylabel('Path')
        _plt.xlabel('Bunch ID')
        # _plt.legend()

    def checkBunchPath(self, bunchID,
                       refTL='BPMI.1860.TL', refTLD='BPMA.2054.TL',
                       refT1='BPMA.2097.T1', refT2='BPMA.2161.T2'):
        df_reduced = self.reduceDFbyBPMTrainBunchByIndex(trains=0, bunches=bunchID, valid=False)
        valid = df_reduced[df_reduced.index.get_level_values(0).isin([refTL, refTLD, refT1, refT2])].values
        print(valid)
        # Valid_TLD = df_reduced[df_reduced.index.get_level_values(0) == refTLD].Valid
        # Valid_T1 = df_reduced[df_reduced.index.get_level_values(0) == refT1].Valid
        # Valid_T2 = df_reduced[df_reduced.index.get_level_values(0) == refT2].Valid
        # print(Valid_TLD, Valid_T1, Valid_T2)
        # if Valid_TLD and not Valid_T1 and not Valid_T2:
        #     return 'TLD'
        # if not Valid_TLD and Valid_T1 and not Valid_T2:
        #     return 'T1'
        # if not Valid_TLD and not Valid_T1 and Valid_T2:
        #     return 'T2'
        # raise ValueError("Bunch {} is not consistant".format(bunchID))


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
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
