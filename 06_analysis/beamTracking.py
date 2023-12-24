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


def PlotTheoryFitAndChi2(Track, coord, ref_S, ref_coord, initial_fit, bdsimCompare=False):
    ax1 = _plt.gca()
    ax2 = ax1.twinx()
    S = _np.unique(Track.twiss.getColumnsByKeys('S'))

    Curve_Theory, S_list = Track.getTheoryAndFit(coord, ref_S, ref_coord, initial_fit)
    ax1.plot(S, _np.abs(Curve_Theory(S)), ls='-', color='C0', label='Theory for ${{{c}}}$'.format(c=ref_coord))
    ax1.set_ylabel("$|sin(\mu_{\\rm IP}-\mu (s))|$")
    ax1.set_yscale('log')

    Curve_Chi2_pymad8 = Track.pymad8.getChi2(coord, ref_S, ref_coord)
    ax2.plot(S, Curve_Chi2_pymad8(S), ls='--', color='C1', label='$\chi^2$ for ${{{c}}}$ from Mad8'.format(c=ref_coord))
    ax2.set_ylabel("$\chi^2$")
    ax2.set_yscale('log')

    if bdsimCompare:
        Curve_Chi2_bdsim = Track.bdsim.getChi2(coord, ref_S, ref_coord)
        ax2.plot(S, Curve_Chi2_bdsim(S), ls='', color='C3', marker='+', label='$\chi^2$ for ${{{c}}}$ from BDSIM'.format(c=ref_coord))

    label = 'Fitted $s$ positions'
    for s in S_list:
        ax1.axvline(s, ls=':', color='C4', label=label)
        label = None

    ax1.axvline(ref_S, ls='--', color='C2', label='Reference Sampler')
    ax1.set_xlabel('$s$ [m]')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _plt.legend(h1 + h2, l1 + l2)


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
    for s in S_list:
        _plt.axvline(s, color='C3', ls=':', label=label)
        label = None


def PlotPhaseAdvanceCheck(Track, coord, S_list, ref_name='IP.LUXE.T20'):
    ref_S = Track.twiss.getElementByNames(ref_name, 'S')
    _plt.axvline(ref_S, ls='-', color='C2', label=ref_name)

    if coord in ['X', 'PX']:
        Track.twiss.plotXY('S', 'MUX', color='C0')
    elif coord in ['Y', 'PY']:
        Track.twiss.plotXY('S', 'MUY', color='C1')

    PlotMuTheory(Track, coord, ref_name=ref_name)
    PlotMuSimulation(Track, coord, S_list)

    _plt.xlabel('$s$ [m]')
    _plt.ylabel('$\mu$ [2$\pi$ rad]')
    _plt.legend()


def PlotSVDCoeff(Track, ref_S, ref_coord='X', BPM_list=None, BPM_list_type='pos', s_range=[-_np.inf, _np.inf],
                 noise=10e-6, mean_sub=False, predicted_S_list=None, printLabels=True, bdsimCompare=False):
    M, ref_vect, S_vect = Track.pymad8.buildBPMmatrix(ref_S, ref_coord, BPM_list=BPM_list, BPM_list_type=BPM_list_type,
                                                      s_range=s_range, noise=noise, mean_sub=mean_sub)
    C_vect = Track.pymad8.SVD(M, ref_vect)
    C_X, C_Y = _np.split(C_vect, 2)
    _plt.plot(S_vect, _np.abs(C_X), '+-', color='C0', markersize=15, markeredgewidth=2, label='$|c_x|$ from Mad8')
    _plt.plot(S_vect, _np.abs(C_Y), '+-', color='C1', markersize=15, markeredgewidth=2, label='$|c_y|$ from Mad8')

    if bdsimCompare:
        M_bdsim, ref_vect_bdsim, S_vect_bdsim = Track.bdsim.buildBPMmatrix(ref_S, ref_coord, BPM_list=BPM_list, BPM_list_type=BPM_list_type,
                                                                           s_range=s_range, noise=noise, mean_sub=mean_sub)
        C_vect_bdsim = Track.bdsim.SVD(M_bdsim, ref_vect_bdsim)
        C_X_bdsim, C_Y_bdsim = _np.split(C_vect_bdsim, 2)
        _plt.plot(S_vect_bdsim, _np.abs(C_X_bdsim), 'o--', color='C0',
                  markersize=15, markeredgewidth=2, markerfacecolor='None', label='$|c_x|$ from BDSIM')
        _plt.plot(S_vect_bdsim, _np.abs(C_Y_bdsim), 'o--', color='C1',
                  markersize=15, markeredgewidth=2, markerfacecolor='None', label='$|c_y|$ from BDSIM')

    if predicted_S_list is not None:
        label = 'Previously found positions'
        for s in predicted_S_list:
            _plt.axvline(s, ls=':', color='C3', label=label)
            label = None

    _plt.axvline(ref_S, ls='--', color='C2', label='Reference Sampler')
    _plt.ylabel('$|c_x|$ / $|c_y|$')
    _plt.xlabel('$s$ [m]')
    if printLabels:
        _plt.legend()


def PlotResolution(Track, ref_coord, ref_S, S_dict, noise=10e-6, bins=50, bdsimCompare=False, color=None):
    unit = _m8.Sim.CheckUnits(ref_coord)
    if color is None:
        color = getColor(ref_coord)

    Res = Track.pymad8.CalcResolution(ref_coord, ref_S, S_dict[ref_coord], noise=noise)
    _plt.hist(Res, bins=bins, histtype='step', color=color,
              label='$R_{{{c}}}$ = {:1.3f} $\\mu${} from Mad8'.format(_np.std(Res)*1e6, unit, c=getLabelCoord(ref_coord)))

    if bdsimCompare:
        Res_bdsim = Track.bdsim.CalcResolution(ref_coord, ref_S, S_dict[ref_coord], noise=noise)
        _plt.hist(Res_bdsim, bins=bins, histtype='step', ls='--', color=color,
                  label='$R_{{{c}}}$ = {:1.3f} $\\mu${} from BDSIM'.format(_np.std(Res_bdsim)*1e6, unit, c=getLabelCoord(ref_coord)))

    _plt.xlabel('${}_{{{m}}}-{}_{{{t}}}$ [{}]'.format(getLabelCoord(ref_coord), getLabelCoord(ref_coord), unit, m='IP,meas', t='IP,track'))
    _plt.ylabel('Number of Events')
    _plt.legend()


def PlotResWrtBPMnoise(Track, ref_coord, ref_S, S_dict, noise_range, bdsimCompare=False):
    unit = _m8.Sim.CheckUnits(ref_coord)

    std_list = []
    for noise in noise_range:
        Res = Track.pymad8.CalcResolution(ref_coord, ref_S, S_dict[ref_coord], noise=noise)
        std_list.append(_np.std(Res))
    _plt.plot(noise_range, std_list, '+-', color=getColor(ref_coord), markersize=15, markeredgewidth=2,
              label='$R_{{{c}}}$ ({} BPMs / Mad8)'.format(len(S_dict[ref_coord]), c=getLabelCoord(ref_coord)))

    if bdsimCompare:
        std_list_bdsim = []
        for noise in noise_range:
            Res_bdsim = Track.bdsim.CalcResolution(ref_coord, ref_S, S_dict[ref_coord], noise=noise)
            std_list_bdsim.append(_np.std(Res_bdsim))
        _plt.plot(noise_range, std_list_bdsim, 'o--', color=getColor(ref_coord), markersize=15, markeredgewidth=2, markerfacecolor='None',
                  label='$R_{{{c}}}$ ({} BPMs / BDSIM)'.format(len(S_dict[ref_coord]), c=getLabelCoord(ref_coord)))

    _plt.xlabel('Noise at BPMs [m]')
    _plt.ylabel('$R_{{{c}}}$ [{}]'.format(unit, c=getLabelCoord(ref_coord)))
    _plt.legend()


def PlotResWrtNumberOfBPM(Track, ref_coord, ref_S, S_dict, noise=10e-6, sortcoeff=False, bdsimCompare=False):
    unit = _m8.Sim.CheckUnits(ref_coord)
    BPM_nb = range(1, len(S_dict[ref_coord]) + 1)

    if sortcoeff:
        BPM_sorted_list = Track.pymad8.SortCoeff(ref_coord, ref_S, S_dict[ref_coord], noise=noise)
    else:
        BPM_sorted_list = S_dict[ref_coord]

    std_list = []
    for i in BPM_nb:
        Res = Track.pymad8.CalcResolution(ref_coord, ref_S, BPM_sorted_list[:i], noise=noise)
        std_list.append(_np.std(Res))
    _plt.plot(BPM_nb, std_list, '+-', color=getColor(ref_coord), markersize=15, markeredgewidth=2,
              label='$R_{{{c}}}$ ({} m of BPM noise / Mad8)'.format(noise, c=getLabelCoord(ref_coord)))

    if bdsimCompare:
        std_list_bdsim = []
        for i in BPM_nb:
            Res_bdsim = Track.pymad8.CalcResolution(ref_coord, ref_S, BPM_sorted_list[:i], noise=noise)
            std_list_bdsim.append(_np.std(Res_bdsim))
        _plt.plot(BPM_nb, std_list_bdsim, 'o--', color=getColor(ref_coord), markersize=15, markeredgewidth=2, markerfacecolor='None',
                  label='$R_{{{c}}}$ ({} m of BPM noise / BDSIM)'.format(noise, c=getLabelCoord(ref_coord)))

    _plt.xlabel('Number of BPM used')
    _plt.ylabel('$R_{{{c}}}$ [{}]'.format(unit, c=getLabelCoord(ref_coord)))
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


def getColor(coord):
    if coord in ['X', 'PX']:
        return 'C0'
    elif coord in ['Y', 'PY']:
        return 'C1'
    else:
        raise ValueError('Input coord should be X, Y, PX or PY')


def getLabelCoord(coord):
    if coord == 'X':
        return "x"
    elif coord == 'Y':
        return "y"
    elif coord == 'PX':
        return "x'"
    elif coord == 'PY':
        return "y'"
    else:
        raise ValueError('Input coord should be X, Y, PX or PY')
