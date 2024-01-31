# import pydoocs
import h5py as _h5
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import time as t
import pymad8 as _m8

bunch_pattern_adress = "XFEL.DIAG/TIMER/DI1914TL/BUNCH_PATTERN"

bl_time_adress = "XFEL.SDIAG/THZ_SPECTROMETER.RECONSTRUCTION/CRD.1934.TL.NTH/OUTPUT_TIMES"
bl_current_adress = "XFEL.SDIAG/THZ_SPECTROMETER.RECONSTRUCTION/CRD.1934.TL.NTH/CURRENT_PROFILE"
bl_number_adress = "XFEL.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRD.1934.TL/NTH_BUNCH"


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
            'BPMA.1966.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 1943.710757, 'X': 0.000000, 'Y': -2.428409},  # 11
            # TLD
            'BPMD.1977.TL': {'MAD8_name': 'BPMD.TL',    'Line': 'TL', 'S': 1955.124959, 'X': 0.000000, 'Y': -2.432576},
            'BPMA.1995.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 1972.760701, 'X': 0.000000, 'Y': -2.439016},
            'BPMA.2011.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 1988.710701, 'X': 0.000000, 'Y': -2.444839},
            'BPMD.2022.TL': {'MAD8_name': 'BPMD.TL',    'Line': 'TL', 'S': 2000.125103, 'X': 0.000000, 'Y': -2.449007},
            'BPMA.2041.TL': {'MAD8_name': 'BPMA.TL',    'Line': 'TL', 'S': 2018.710845, 'X': 0.000000, 'Y': -2.455793},  # 5

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

    nbtrainmax, nbbunchmax = getNbTrainsBunches(rawdata)
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

    nbtrainmax, nbbunchmax = getNbTrainsBunches(rawdata)
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


def getNbTrainsBunches(rawdata):
    bpmlist = list(rawdata['XFEL.DIAG']['BPM'].keys())
    try:
        nbtrain, nbbunch = rawdata['XFEL.DIAG']['BPM'][bpmlist[0]]['X.TD'].shape
    except ValueError:
        nbtrain = len(rawdata['XFEL.DIAG']['BPM'][bpmlist[0]]['TrainId'])
        if nbtrain == 1:
            nbbunch = len(rawdata['XFEL.DIAG']['BPM'][bpmlist[0]]['X.TD'])
        else:
            nbbunch = 1
    return nbtrain, nbbunch


def getH5dataInDF(inputfilename, bpmdict=BPM_DICT):
    rawdata = _h5.File(inputfilename, 'r')
    bpmdata = rawdata['XFEL.DIAG']['BPM']
    energydata = rawdata['XFEL.DIAG']['BEAM_ENERGY_MEASUREMENT']['CL']['ENERGY.ALL']
    timedata = rawdata['XFEL.SDIAG']['BAM.DAQ']
    if len(findUnmatchedTrainID(inputfilename)) > 0:
        raise ValueError("Inconsistant Train IDs in file : {}".format(inputfilename))
    bpmlist = list(bpmdata.keys())
    TrainID = bpmdata[bpmlist[0]]['TrainId']
    nbtrain, nbbunch = getNbTrainsBunches(rawdata)
    keys = ['X', 'DX', 'Y', 'DY', 'Charge', 'Valid', 'S', 'E', 'DE', 'Time', 'DTime']
    keys = ['X', 'Y', 'Charge', 'Valid', 'S', 'E', 'Time']
    data = {}
    for k in keys:
        data[k] = []
    for bpm in bpmlist:
        data['X'].append(_np.array(bpmdata[bpm]['X.TD'])*1e-3)  # mm converted in m
        # data['DX'].append(_np.full((nbtrain, nbbunch), 2e-6))
        data['Y'].append(_np.array(bpmdata[bpm]['Y.TD'])*1e-3)  # mm converted in m
        # data['DY'].append(_np.full((nbtrain, nbbunch), 2e-6))
        data['Charge'].append(_np.array(bpmdata[bpm]['CHARGE.TD'])*1e-9)  # nC converted to C
        data['Valid'].append(_np.array(bpmdata[bpm]['BUNCH_VALID.TD']))
        data['S'].append(_np.full((nbtrain, nbbunch), bpmdict[bpm]['S']))  # m
        E = _np.tile(energydata['Value'], (nbbunch, 1)).transpose()*1e-3  # MeV converted to GeV
        data['E'].append(E)
        # data['DE'].append(E/100)
        T = timedata['1932S.TL.ARRIVAL_TIME.RELATIVE']['Value']  # [:, :nbbunch]  # us ??
        data['Time'].append(_np.array(T))
        # data['DTime'].append(_np.array(T)/100)
    names = ['BPM', 'TrainID', 'BunchID']
    print('All bpm done')
    for key in data:
        data[key] = _np.asarray(data[key])
    print('Asarray done')
    index = _pd.MultiIndex.from_product([range(s) for s in data['X'].shape], names=names)
    for key in data:
        data[key] = data[key].flatten()
    print('Flatten done')
    df_bpm = _pd.DataFrame(data, index=index)
    df_bpm.index.set_levels([bpmlist, TrainID], level=[0, 1], inplace=True)
    print('MI done')
    rawdata.close()

    return df_bpm


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
        df = selectValidElectronInBPMS(df)
    if bpms is not None:
        df = reduceDFbyIndex(df, 'BPM', bpms)
    if trains is not None:
        df = reduceDFbyIndex(df, 'TrainID', trains)
    if bunches is not None:
        df = reduceDFbyIndex(df, 'BunchID', bunches)
    df.index = df.index.remove_unused_levels()
    return df


def getOpticsFromXls(inputfilename='../../../../Desktop/component_list_2023.07.02.xls', sheet_name='I1toT5D', Smin=1838, Smax=2175):
    sheet_df = _pd.read_excel(inputfilename, sheet_name=sheet_name)[1:]
    reduced_sheet_df = sheet_df[sheet_df.S.between(Smin, Smax)]
    S = reduced_sheet_df.S
    BETX = reduced_sheet_df.BETX
    BETY = reduced_sheet_df.BETY
    return S, BETX, BETY


def getOpticsFromMad8(inputfilename='../01_mad8/folder_test_xfel/XFEL_Lattice_9/TWISS_T4D', Smin=1838, Smax=2175):
    tw = _m8.Output(inputfilename)
    reduced_df = tw.data[tw.data.S.between(Smin, Smax)]
    S = reduced_df.S
    BETX = reduced_df.BETX
    BETY = reduced_df.BETY
    DX = reduced_df.DX
    DY = reduced_df.DY
    return S, BETX, BETY, DX, DY


def buildMatrixAndVectorForSVD(df, refbpmname, coord='X', trains=None, bunches=None):
    df_reduced = reduceDFbyBPMTrainBunchByIndex(df, trains=trains, bunches=bunches)
    df_ref = df_reduced.loc[df_reduced.index.get_level_values('BPM') == refbpmname][['X', 'Y']]
    df_matrix = df_reduced.loc[df_reduced.index.get_level_values('BPM') != refbpmname][['X', 'Y']]

    nb_trains = df_matrix.index.levshape[1]
    nb_bunches = df_matrix.index.levshape[2]

    M_X = df_matrix['X'].to_numpy().reshape((-1, nb_trains*nb_bunches)).transpose()
    M_Y = df_matrix['Y'].to_numpy().reshape((-1, nb_trains*nb_bunches)).transpose()
    Vect_ref = df_ref[coord].to_numpy()
    M = _np.concatenate((M_X, M_Y), axis=1)

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
    """Return the correlation coefficients from a given matrix M using a Singular Value Decomposition method"""
    U, d, V_t = _np.linalg.svd(M, full_matrices=False)
    D = _np.diag(d)

    D_i = _np.linalg.inv(D)
    U_t = U.transpose()
    V = V_t.transpose()

    C = _np.dot(_np.dot(V, _np.dot(D_i, U_t)), ref_Vect)
    return C


def calcResidual(M, ref_Vect):
    C = calcCoeffsWithSVD(M, ref_Vect)
    # R = ref_Vect - _np.dot(M, C)
    R = _np.dot(M, C)
    return R


def calcBPMResolution(M, ref_Vect):
    R = calcResidual(M, ref_Vect)
    Res = _np.sqrt(sum(R**2)/len(R))
    return Res


def calcJitterAndNoise(M, ref_Vect):
    C = calcCoeffsWithSVD(M, ref_Vect)
    Jitter = _np.dot(M, C)
    Noise = ref_Vect - _np.dot(M, C)

    return Jitter, Noise


def plotBPM2D(bpmdict=BPM_DICT):
    plotOptions()

    X = [BPM_DICT[key]['X'] for key in BPM_DICT]
    S = [BPM_DICT[key]['S'] for key in BPM_DICT]
    _plt.scatter(X, S, marker='o')

    Sdiff = _np.abs(max(S)-min(S))/2
    # _plt.xlim(_np.mean(Y) - Sdiff, _np.mean(Y) + Sdiff)

    _plt.xlabel('X [m]')
    _plt.ylabel('S [m]')

    # _plt.legend()


def plotBPM3D(bpmdict=BPM_DICT):
    fig = _plt.figure()
    ax = fig.add_subplot(projection='3d')

    X = [BPM_DICT[key]['X'] for key in BPM_DICT]
    Y = [BPM_DICT[key]['Y'] for key in BPM_DICT]
    S = [BPM_DICT[key]['S'] for key in BPM_DICT]
    ax.scatter(X, S, Y, marker='o')

    Sdiff = _np.abs(max(S)-min(S))/2
    # ax.set_xlim(_np.mean(X)-Sdiff, _np.mean(X)+Sdiff)
    ax.set_zlim(_np.mean(Y) - Sdiff, _np.mean(Y) + Sdiff)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('S [m]')
    ax.set_zlabel('Y [m]')

    # _plt.legend()


def plotOptics():
    S, BETX, BETY = getOpticsFromXls()
    plotOptions()
    _plt.plot(S, BETX, '+-', color='C0', markersize=15, markeredgewidth=2, label=r'$\beta_X$')
    _plt.plot(S, BETY, '+-', color='C1', markersize=15, markeredgewidth=2, label=r'$\beta_Y$')
    _plt.ylabel(r'$\beta_X$/$\beta_Y$ [m]')
    _plt.xlabel('S [m]')
    _plt.legend()


def plotResidual(df, refbpmname, trains=None, bunches=None, bins=20):
    V_X, M_X = buildMatrixAndVectorForSVD(df, refbpmname, coord='X', trains=trains, bunches=bunches)
    V_Y, M_Y = buildMatrixAndVectorForSVD(df, refbpmname, coord='Y', trains=trains, bunches=bunches)
    R_X = calcResidual(M_X, V_X)
    R_Y = calcResidual(M_Y, V_Y)
    Res_X = calcBPMResolution(M_X, V_X)
    Res_Y = calcBPMResolution(M_Y, V_Y)

    plotOptions()
    _plt.hist(R_X, bins=bins, histtype='step', color='C0', label='${} : Res_X = {:1.2e} m$'.format(refbpmname, Res_X))
    _plt.hist(R_Y, bins=bins, histtype='step', color='C1', label='${} : Res_Y = {:1.2e} m$'.format(refbpmname, Res_Y))
    _plt.ylabel('Entries')
    _plt.xlabel('Residuals [m]')
    _plt.legend()


def plotResolutions(df, trains=None, bunches=None):
    df_reduced = reduceDFbyBPMTrainBunchByIndex(df, trains=trains, bunches=bunches)
    df_reduced = df_reduced.sort_values(by='S')
    S = df_reduced.S.unique()
    Res_X = _np.array([])
    Res_Y = _np.array([])
    for bpm in df_reduced.index.get_level_values(0).unique():
        V_X, M_X = buildMatrixAndVectorForSVD(df, bpm, coord='X', trains=trains, bunches=bunches)
        V_Y, M_Y = buildMatrixAndVectorForSVD(df, bpm, coord='Y', trains=trains, bunches=bunches)
        Res_X = _np.append(Res_X, calcBPMResolution(M_X, V_X))
        Res_Y = _np.append(Res_Y, calcBPMResolution(M_Y, V_Y))

    plotOptions()
    _plt.plot(S, Res_X, '+-', color='C0', markersize=15, markeredgewidth=2, label='$Res_X$')
    _plt.plot(S, Res_Y, '+-', color='C1', markersize=15, markeredgewidth=2, label='$Res_Y$')
    _plt.ylabel('Resolution [m]')
    _plt.xlabel('$S$ [m]')
    _plt.legend()


def plotJitterAndNoise(df, trains=None, bunches=None):
    df_reduced = reduceDFbyBPMTrainBunchByIndex(df, trains=trains, bunches=bunches)
    df_reduced = df_reduced.sort_values(by='S')
    S = df_reduced.S.unique()
    Jitter_X = _np.array([])
    Jitter_Y = _np.array([])
    Noise_X = _np.array([])
    Noise_Y = _np.array([])
    for bpm in df_reduced.index.get_level_values(0).unique():
        V_X, M_X = buildMatrixAndVectorForSVD(df, bpm, coord='X', trains=trains, bunches=bunches)
        V_Y, M_Y = buildMatrixAndVectorForSVD(df, bpm, coord='Y', trains=trains, bunches=bunches)
        J_X, N_X = calcJitterAndNoise(M_X, V_X)
        J_Y, N_Y = calcJitterAndNoise(M_Y, V_Y)
        Jitter_X = _np.append(Jitter_X, J_X.std())
        Jitter_Y = _np.append(Jitter_Y, J_Y.std())
        Noise_X = _np.append(Noise_X, N_X.std())
        Noise_Y = _np.append(Noise_Y, N_Y.std())

    plotOptions(rows_colums=[3, 1])
    _plt.subplot(3, 1, 1)
    _plt.plot(S, Jitter_X, '+-', color='C0', markersize=15, markeredgewidth=2, label='$Jitter_X$')
    _plt.plot(S, Jitter_Y, '+-', color='C1', markersize=15, markeredgewidth=2, label='$Jitter_Y$')
    _plt.ylabel('$X/Y$ [m]')
    _plt.xlabel('$S$ [m]')
    _plt.legend()

    S_optics, BETX, BETY, DX, DY = getOpticsFromMad8(Smin=min(S), Smax=max(S))

    _plt.subplot(3, 1, 2)
    _plt.plot(S_optics, BETX, '-', color='C0', markersize=15, markeredgewidth=2, label=r'$\beta_X$')
    _plt.plot(S_optics, BETY, '-', color='C1', markersize=15, markeredgewidth=2, label=r'$\beta_Y$')
    _plt.ylabel(r'$\beta_X$/$\beta_Y$ [m]')
    _plt.xlabel('S [m]')
    _plt.legend()

    _plt.subplot(3, 1, 3)
    _plt.plot(S_optics, DX, '--', color='C0', markersize=15, markeredgewidth=2, label=r'$D_X$')
    _plt.plot(S_optics, DY, '--', color='C1', markersize=15, markeredgewidth=2, label=r'$D_Y$')
    _plt.ylabel(r'$D_X$/$D_Y$ [m]')
    _plt.xlabel('S [m]')
    _plt.legend()

    plotOptions()
    _plt.plot(S, Noise_X, '+-', color='C0', markersize=15, markeredgewidth=2, label='$Noise_X$')
    _plt.plot(S, Noise_Y, '+-', color='C1', markersize=15, markeredgewidth=2, label='$Noise_Y$')
    _plt.ylabel('$X/Y$ [m]')
    _plt.xlabel('$S$ [m]')
    _plt.legend()


def plotSurvey(inputfilename):
    df_bpm, df_e, df_at = getH5dataInDF(inputfilename)

    plotTrajectory(df_bpm, train=0, bunch=0)
    plotHistogram(df_bpm, bpm=0, train=0)
    plotHistogram(df_bpm, bpm=0, bunch=0)


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


def convertBPMNameToPosition(bpmnames, bpmdict=BPM_DICT):
    df = _pd.DataFrame(bpmdict).transpose()
    try:
        return df.loc[bpmnames]['S'].to_numpy()
    except AttributeError:
        return df.loc[bpmnames]['S'].to_numpy()


def getFirstBPMName(bpmnames, bpmdict=BPM_DICT):
    df = _pd.DataFrame(bpmdict).transpose()
    try:
        df = df[df.index.isin(bpmnames)]
    except TypeError:
        df = df[df.index == bpmnames]
    return df[df['S'] == df['S'].min()].index.to_numpy()[0]


def removeAllColumnsWithZeros(df):
    df_nonzero = df.loc[:, (df != 0).any(axis=0)]
    return df_nonzero


def selectValidElectronInBPMS(df_bpm):
    df = df_bpm[df_bpm['Valid'] == 1]
    return df


def removeUnusedIndex(df):
    df.index = df.index.remove_unused_levels()


def plotOptions(figsize=[9, 6], rows_colums=[1, 1], font_size=17):
    _plt.rcParams['font.size'] = font_size
    fig, ax = _plt.subplots(rows_colums[0], rows_colums[1], figsize=(figsize[0], figsize[1]))
    fig.tight_layout()
