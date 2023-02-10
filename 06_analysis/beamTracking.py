import numpy as _np
import pylab as _pl
import matplotlib.pyplot as _plt
import pymad8 as _m8
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


def setTrackCollection(nb_part, energy, paramdict, filemad8, filebdsim):
    T_C = _m8.Sim.Track_Collection(energy)

    T_C.AddTrack(0, 0, 0, 0, 0, 0)
    # T_C.AddTrack(0, 0, 0, 0, 0, 0.01)
    T_C.GenerateNtracks(nb_part-1, paramdict)

    T_C.WriteMad8Track(filemad8)
    T_C.WriteBdsimTrack(filebdsim)

    return T_C


def setSamplersAndTrack(twissfile, rmatfile, nb_sampler=100):
    twiss = _m8.Output(twissfile)
    rmat = _m8.Output(rmatfile, 'rmat')

    T = _m8.Sim.Tracking(twiss, rmat)
    # T.GenerateSamplers(nb_sampler)
    # T.AddSamplers('IP.LUXE.T20', select='name')
    # T.AddSamplers(['QUAD', 'RBEN', 'SBEN', 'SEXT', 'OCTU', 'MARK', 'DRIF', '    '], select='type')
    T.AddAllElementsAsSamplers()

    return T


def getChi2(Track, coord, ref_index, ref_coord):
    SLOPE = []
    ERR = []
    COV_ERR = []
    for index in range(878):
        try:
            slope, err, cov_err = Track.PlotCorrelation(index, coord, ref_index, ref_coord, linFit=True, noPlots=True)
            SLOPE.append(slope)
            ERR.append(err)
            COV_ERR.append(cov_err)
        except:
            pass

    return ERR


def getTheoryPos(Track, coord, ref_index):
    ref_mux = Track.twiss.getRowsByIndex(ref_index)['MU{}'.format(coord)]

    def calcTheory(mu):
        return _np.sin(2 * _np.pi * (ref_mux - mu))

    mu_vect = Track.twiss.data['MU{}'.format(coord)].to_numpy()
    return calcTheory(mu_vect)


def getTheoryAng(Track, coord, ref_index):
    ref_mux = Track.twiss.getRowsByIndex(ref_index)['MU{}'.format(coord)]
    ref_alpha = Track.twiss.getRowsByIndex(ref_index)['ALPH{}'.format(coord)]

    def calcTheory(mu):
        return _np.sin(2 * _np.pi * (ref_mux - mu) + _np.arctan(-1 / ref_alpha))

    mu_vect = Track.twiss.data['MU{}'.format(coord)].to_numpy()
    return calcTheory(mu_vect)


def TheoryFitAndChi2(Track, ref_name, initial_fit):
    ref_index = Track.twiss.getIndexByNames(ref_name)
    S_vect = Track.twiss.data['S'].to_numpy()

    THEORY_X = getTheoryPos(Track, 'X', ref_index)
    THEORY_PX = getTheoryAng(Track, 'X', ref_index)
    THEORY_Y = getTheoryPos(Track, 'Y', ref_index)
    THEORY_PY = getTheoryAng(Track, 'Y', ref_index)

    THEORY_X_CURVE = interp1d(S_vect, THEORY_X, fill_value="extrapolate")
    THEORY_PX_CURVE = interp1d(S_vect, THEORY_PX, fill_value="extrapolate")
    THEORY_Y_CURVE = interp1d(S_vect, THEORY_Y, fill_value="extrapolate")
    THEORY_PY_CURVE = interp1d(S_vect, THEORY_PY, fill_value="extrapolate")

    THEORY_CURVE = {'X': THEORY_X_CURVE, 'PX': THEORY_PX_CURVE, 'Y': THEORY_Y_CURVE, 'PY': THEORY_PY_CURVE}

    CHI2_X = getChi2(Track, 'X', ref_index, 'X')
    CHI2_PX = getChi2(Track, 'X', ref_index, 'PX')
    CHI2_Y = getChi2(Track, 'Y', ref_index, 'Y')
    CHI2_PY = getChi2(Track, 'Y', ref_index, 'PY')

    CHI2_X_CURVE = interp1d(S_vect, CHI2_X, fill_value="extrapolate")
    CHI2_PX_CURVE = interp1d(S_vect, CHI2_PX, fill_value="extrapolate")
    CHI2_Y_CURVE = interp1d(S_vect, CHI2_Y, fill_value="extrapolate")
    CHI2_PY_CURVE = interp1d(S_vect, CHI2_PY, fill_value="extrapolate")

    CHI2_CURVE = {'X': CHI2_X_CURVE, 'PX': CHI2_PX_CURVE, 'Y': CHI2_Y_CURVE, 'PY': CHI2_PY_CURVE}

    S_list_X = fsolve(THEORY_X_CURVE, initial_fit[0])
    S_list_PX = fsolve(THEORY_PX_CURVE, initial_fit[1])
    S_list_Y = fsolve(THEORY_Y_CURVE, initial_fit[2])
    S_list_PY = fsolve(THEORY_PY_CURVE, initial_fit[3])

    S_list = {'X': S_list_X, 'PX': S_list_PX, 'Y': S_list_Y, 'PY': S_list_PY}

    return THEORY_CURVE, CHI2_CURVE, S_list


def PlotTheoryFitAndChi2(Track, THEORY, CHI2, S_list):
    S = range(0, 290, 1)
    coords = ['X', 'Y', 'PX', 'PY']
    ref_S = Track.twiss.getRowsByNames('IP.LUXE.T20')['S'].tolist()[0]

    fig, ax = _plt.subplots(4, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [1, 1, 1, 1]}, sharex='all')
    for i, coord in enumerate(coords):
        offset = _np.std(THEORY[coord](S)) / _np.std(CHI2[coord](S))

        _plt.subplot(len(coords), 1, i + 1)
        _plt.plot(S, _np.abs(THEORY[coord](S)), label='Theory for {}'.format(coord))
        _plt.plot(S, CHI2[coord](S) * offset, ls='--', label='Measured Chi2 for {}'.format(coord))
        _plt.axvline(ref_S, ls='--', color='C2', label='LUXE IP')

        label = 'Fitted S positions'
        for s in S_list[coord]:
            _plt.axvline(s, ls=':', color='C3', label=label)
            label = None

        #_plt.ylabel("$|sin(\mu_{IP}-\mu)|$")
        _plt.xlabel('S [m]')
        _plt.yscale('log')
        _plt.legend()


def PlotMuTheory(Track, coord, ref_name='IP.LUXE.T20'):
    if coord in ['X', 'Y']:
        plane_coord = coord
        offset = 0
    elif coord in ['PX', 'PY']:
        plane_coord = getPlaneCoord(coord)
        ref_alpha = Track.twiss.getRowsByNames(ref_name)['ALPH{}'.format(plane_coord)].to_numpy()
        offset = _np.arctan(-1 / ref_alpha)
    else:
        raise ValueError('Input coord should be X, Y, PX or PY')

    MU_IP = Track.twiss.getRowsByNames(ref_name)['MU{}'.format(plane_coord)].values[0]

    if plane_coord == 'X':
        color = 'C0'
    elif plane_coord == 'Y':
        color = 'C1'

    label = 'Predicted MU{} values for {}'.format(plane_coord, coord)
    for i in range(0, 8):
        _plt.axhline(y=MU_IP - i / 2 + offset / (2 * _np.pi), color=color, ls='--', label=label)
        label = None


def PlotMuSimulation(Track, coord, S_list):
    label = 'S values from simulation for {}'.format(coord)
    for s in S_list[coord]:
        _plt.axvline(s, color='C3', ls=':', label=label)
        label = None


def PlotPhaseAdvanceCheck(Track, coord, S_list, ref_name='IP.LUXE.T20'):
    ref_S = Track.twiss.getRowsByNames(ref_name)['S'].tolist()[0]
    _plt.axvline(ref_S, ls='-', color='C2', label=ref_name)

    if coord in ['X', 'PX']:
        Track.twiss.plotXY('S', 'MUX', color='C0')
    elif coord in ['Y', 'PY']:
        Track.twiss.plotXY('S', 'MUY', color='C1')

    PlotMuTheory(Track, coord, ref_name=ref_name)
    PlotMuSimulation(Track, coord, S_list)

    _plt.xlabel('S [m]')
    _plt.ylabel('Mu [2$\pi$ rad]')
    _plt.legend()


def buildBPMmatrix(Track, ref_index, ref_coord, BPM_list=None, BPM_list_type='pos', s_range=None, noise=None, mean_sub=False):
    ref_vect = Track.pymad8_df[Track.pymad8_df.index.get_level_values(0) == ref_index][ref_coord].to_numpy()

    if BPM_list is not None:
        if BPM_list_type == 'pos':
            s_list = []
            for s in BPM_list:
                s_list.append(Track.twiss.getIndexByNearestS(s))
            reduced_df = Track.pymad8_df[Track.pymad8_df.index.get_level_values(0).isin(s_list)]
        if BPM_list_type == 'type':
            reduced_df = Track.pymad8_df.loc[Track.pymad8_df['TYPE'].isin(BPM_list)]
    else:
        reduced_df = Track.pymad8_df
    if s_range is not None:
        reduced_df = reduced_df.loc[reduced_df['S'] <= s_range[1]]
        reduced_df = reduced_df.loc[reduced_df['S'] >= s_range[0]]
    if ref_index in reduced_df.index.get_level_values(0).unique():
        reduced_df = reduced_df.drop(index=ref_index, level=0)

    M_X = reduced_df['X'].to_numpy().reshape((-1, len(Track.initial_dict))).transpose()
    M_Y = reduced_df['Y'].to_numpy().reshape((-1, len(Track.initial_dict))).transpose()
    S_vect = reduced_df[reduced_df.index.get_level_values(1) == 0]['S'].to_numpy()
    M = _np.concatenate((M_X, M_Y), axis=1)

    if noise is not None:
        M_noise = _np.random.normal(0, noise, M.shape)
        M = M + M_noise
    if mean_sub:
        V_mean = M.mean(0)
        M = M - V_mean

    return M, ref_vect, S_vect


def SVD(M, ref_vect):
    U, d, V_t = _np.linalg.svd(M, full_matrices=False)
    D = _np.diag(d)

    D_i = _np.linalg.inv(D)
    U_t = U.transpose()
    V = V_t.transpose()

    C = _np.dot(_np.dot(V, _np.dot(D_i, U_t)), ref_vect)
    return C


def PlotSVD(S_vect, C, predicted_S_list=None, ref_S=None, printLabels=False, logScale=False):
    C1, C2 = _np.split(C, 2)
    _plt.plot(S_vect, _np.abs(C1), '+-', color='C0', label='X')
    _plt.plot(S_vect, _np.abs(C2), '+-', color='C1', label='Y')

    if predicted_S_list is not None:
        label = 'Previously found positions'
        for s in predicted_S_list:
            _plt.axvline(s, ls=':', color='C3', label=label)
            label = None

    if ref_S is not None:
        _plt.axvline(ref_S, ls='--', color='C2', label='Reference Sampler')

    if logScale:
        _plt.yscale("log")
    _plt.ylabel('Correlation factors (ABSOLUTE)')
    _plt.xlabel('S [m]')
    if printLabels:
        _plt.legend()


def PlotBPMSVD(Track, ref_name='IP.LUXE.T20', ref_coord='X', BPM_list=None, BPM_list_type='pos',
               s_range=None, noise=10e-6, mean_sub=False,
               predicted_S_list=None, printLabels=False, logScale=False):
    ref_index = Track.twiss.getIndexByNames(ref_name)
    ref_S = Track.twiss.getElementByIndex(ref_index, 'S')
    M, ref_vect, S_vect = buildBPMmatrix(Track, ref_index, ref_coord,
                                         BPM_list=BPM_list, BPM_list_type=BPM_list_type,
                                         s_range=s_range, noise=noise, mean_sub=mean_sub)
    C_vect = SVD(M, ref_vect)

    PlotSVD(S_vect, C_vect, predicted_S_list=predicted_S_list, ref_S=ref_S, printLabels=printLabels, logScale=logScale)


def CalcResolution(Track, ref_coord, ref_name, BPM_list, noise=10e-6):
    ref_index = Track.twiss.getIndexByNames(ref_name)
    M, Real_vect, S_vect = buildBPMmatrix(Track, ref_index, ref_coord, BPM_list=BPM_list, noise=noise)
    C_vect = SVD(M, Real_vect)

    Meas_vect = _np.dot(M, C_vect)
    Res_array = Meas_vect - Real_vect
    return Res_array


def SortCoeff(Track, ref_coord, ref_name, BPM_list, noise=10e-6):
    ref_index = Track.twiss.getIndexByNames(ref_name)
    M, Real_vect, S_vect = buildBPMmatrix(Track, ref_index, ref_coord, BPM_list=BPM_list, noise=noise)
    C_vect = SVD(M, Real_vect)
    C1, C2 = _np.split(C_vect, 2)
    if ref_coord in ['X', 'PX']:
        return BPM_list[_np.argsort(C1)]
    elif ref_coord in ['Y', 'PY']:
        return BPM_list[_np.argsort(C2)]
    else:
        raise ValueError('Input ref_coord should be X, Y, PX or PY')


def PlotResolution(Track, ref_coord, ref_name, S_list, noise=10e-6, bins=50):
    unit = _m8.Sim.CheckUnits(ref_coord)
    Res = CalcResolution(Track, ref_coord, ref_name, S_list[ref_coord], noise=noise)
    _plt.hist(Res, bins=bins, histtype='step', label='{} : std = %1.3f u{}'.format(ref_coord, unit) % (_np.std(Res) * 1e6))
    _plt.xlabel('IP_meas-IP_real [{}]'.format(unit))
    _plt.ylabel('Number of Events')
    _plt.legend()


def PlotResWrtBPMnoise(Track, ref_coord, ref_name, S_list, noise_range):
    unit = _m8.Sim.CheckUnits(ref_coord)
    std_list = []
    for noise in noise_range:
        Res = CalcResolution(Track, ref_coord, ref_name, S_list[ref_coord], noise=noise)
        std_list.append(_np.std(Res))
    _plt.plot(noise_range, std_list, '+-', label=ref_coord)
    _plt.xlabel('Noise at BPMs [m]')
    _plt.ylabel('Resolution at the IP [{}]'.format(unit))
    _plt.legend()


def PlotResWrtNumberOfBPM(Track, ref_coord, ref_name, S_list, noise=10e-6, sortcoeff=False):
    unit = _m8.Sim.CheckUnits(ref_coord)
    std_list = []
    BPM_nb = range(1, len(S_list[ref_coord]) + 1)

    if sortcoeff:
        BPM_sorted_list = SortCoeff(Track, ref_coord, ref_name, S_list[ref_coord], noise=noise)
    else:
        BPM_sorted_list = S_list[ref_coord]

    for i in BPM_nb:
        Res = CalcResolution(Track, ref_coord, ref_name, BPM_sorted_list[:i], noise=noise)
        std_list.append(_np.std(Res))
    _plt.plot(BPM_nb, std_list, '+-', label=ref_coord)
    _plt.xlabel('Number of BPM used')
    _plt.ylabel('Resolution at the IP [{}]'.format(unit))
    _plt.legend()


def getOppositeCoord(coord):
    if coord == 'X':
        return 'Y'
    elif coord == 'Y':
        return 'X'
    elif coord == 'PX':
        return 'PY'
    elif coord == 'PY':
        return 'PX'
    else:
        raise ValueError('Input coord should be X, Y, PX or PY')


def getPlaneCoord(coord):
    if coord in ['X', 'Y']:
        return coord
    elif coord == 'PX':
        return 'X'
    elif coord == 'PY':
        return 'Y'
    else:
        raise ValueError('Input coord should be X, Y, PX or PY')
