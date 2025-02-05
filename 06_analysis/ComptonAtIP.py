import numpy as _np
import matplotlib.pyplot as _plt
import pybdsim as _bd
import pandas as _pd
import pymad8 as _m8
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.optimize import fsolve


def Load(inputfilename, outputfilename=None, write=False):
    data = _bd.Data.Load(inputfilename)
    e = data.GetEvent()
    et = data.GetEventTree()
    npart = et.GetEntries()
    sampler_names = e.GetSamplerNames()
    sampler_data = e.GetSampler(sampler_names[0])

    particle_list = []
    for particle in range(npart):
        et.GetEntry(particle)
        particle_dict = {'X': 0, 'PX': 0, 'Y': 0, 'PY': 0, 'Z': 0, 'E': 0}
        particle_dict['X'] = sampler_data.x[0]
        particle_dict['PX'] = sampler_data.xp[0]
        particle_dict['Y'] = sampler_data.y[0]
        particle_dict['PY'] = sampler_data.yp[0]
        particle_dict['Z'] = sampler_data.z
        particle_dict['E'] = sampler_data.energy[0]
        particle_list.append(particle_dict)

    if write:
        Write(outputfilename, particle_list)


def Write(outputfilename, particle_list):
    file = open(outputfilename, 'w')
    for part in particle_list:
        for key in part:
            file.write(str(part[key]))
            file.write(' ')
        file.write("\n")
    file.close()


def getBDSIMdataInDF(inputfilename):
    data = _bd.Data.Load(inputfilename)
    e = data.GetEvent()
    et = data.GetEventTree()
    nbentires = et.GetEntries()
    sampler_names = e.GetSamplerNames()
    nbbpm = len(sampler_names)

    bpmlist = []
    data_dict = {'partID': [], 'X': [], 'PX': [], 'Y': [], 'PY': [], 'S': [], 'E': []}
    for i, bpm_name in enumerate(sampler_names):
        _printProgressBar(i, nbbpm, prefix='Load {} | {} bpms, {} particles:'.format(inputfilename.split('/')[-1], nbbpm, nbentires),
                          suffix='Complete', length=50)
        bpm = e.GetSampler(bpm_name)
        bpmlist.append(str(bpm_name).strip("."))
        for evt in et:
            for i, partID in enumerate(bpm.partID):
                if partID == 11:
                    data_dict['partID'].append(partID)
                    data_dict['X'].append(bpm.x[i])
                    data_dict['PX'].append(bpm.xp[i])
                    data_dict['Y'].append(bpm.y[i])
                    data_dict['PY'].append(bpm.yp[i])
                    data_dict['S'].append(bpm.S)
                    data_dict['E'].append(bpm.energy[i])
    for key in data_dict:
        data_dict[key] = _np.asarray(data_dict[key]).flatten()
    df = _pd.DataFrame(data_dict, index=_pd.MultiIndex.from_product([range(s) for s in (nbbpm, len(data_dict['X']))], names=['BPM', 'Particle']))
    df.index.set_levels([bpmlist], level=[0], inplace=True)
    _printProgressBar(nbbpm, nbbpm, prefix='Load {} | {} bpms, {} particles:'.format(inputfilename.split('/')[-1], nbbpm, nbentires),
                      suffix='Complete', length=50)
    return df


def CalcCorrelChi2(df, sampler_name, ref_sampler_name, coord, ref_coord):
    V = df.loc[df.index.get_level_values('BPM') == sampler_name][[coord]].to_numpy().flatten()
    ref_V = df.loc[df.index.get_level_values('BPM') == ref_sampler_name][[ref_coord]].to_numpy().flatten()

    def linear(x, a, b):
        return a * x + b

    popt, pcov = curve_fit(linear, V, ref_V)
    slope, cst = popt
    err = sum((ref_V - slope * V - cst) ** 2)

    return err, slope, cst


def getChi2(df, coord, ref_bpm, ref_coord):
    S = _np.array([])
    Vect_Chi2 = _np.array([])
    nbbpm = len(df.index.get_level_values('BPM').unique())

    for i, bpm in enumerate(df.index.get_level_values('BPM').unique()):
        _printProgressBar(i, nbbpm, prefix='Calulate chi2 for {} bpms :'.format(nbbpm), suffix='Complete', length=50)
        S = _np.append(S, df.loc[df.index.get_level_values('BPM') == bpm][['S']].to_numpy().flatten()[0])
        Vect_Chi2 = _np.append(Vect_Chi2, CalcCorrelChi2(df, bpm, ref_bpm, coord, ref_coord)[0])
    _printProgressBar(nbbpm, nbbpm, prefix='Calulate chi2 for {} bpms :'.format(nbbpm), suffix='Complete', length=50)

    Curve_Chi2 = interp1d(S, Vect_Chi2, fill_value="extrapolate")
    return S, Curve_Chi2


def PlotCorrelation(df, bpm, coord, ref_bpm, ref_coord, figsize=[9, 9]):
    fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 1], sharex=False, height_ratios=None, font_size=12)

    V = df.loc[df.index.get_level_values('BPM') == bpm][[coord]].to_numpy().flatten()
    ref_V = df.loc[df.index.get_level_values('BPM') == ref_bpm][[ref_coord]].to_numpy().flatten()

    _plt.plot(V, ref_V, ls='', marker='+', color='C0', label='BDSIM data')

    err, slope, cst = CalcCorrelChi2(df, bpm, ref_bpm, coord, ref_coord)
    _plt.plot(V, slope * V + cst, color='C2', label='Fit: $\chi^2$ = {}'.format(err))

    _plt.ticklabel_format(axis="x", style="sci", scilimits=(-6, 0))
    _plt.xlabel("{} [m]".format(coord))
    _plt.ylabel("{}_ref [m]".format(ref_coord))
    _plt.legend()


def PlotAllCorrelation(df, bpm, ref_bpm, figsize=[14, 8]):
    fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 2], sharex=False, height_ratios=None, font_size=12)

    X = df.loc[df.index.get_level_values('BPM') == bpm][['X']].to_numpy().flatten()
    ref_X = df.loc[df.index.get_level_values('BPM') == ref_bpm][['X']].to_numpy().flatten()
    Y = df.loc[df.index.get_level_values('BPM') == bpm][['Y']].to_numpy().flatten()
    ref_Y = df.loc[df.index.get_level_values('BPM') == ref_bpm][['Y']].to_numpy().flatten()

    ax[0].plot(X, ref_X, ls='', marker='+', color='C0', label='BDSIM data')
    errX, slopeX, cstX = CalcCorrelChi2(df, bpm, ref_bpm, 'X', 'X')
    ax[0].plot(X, slopeX * X + cstX, color='C2', label='Fit: $\chi^2$ = {}'.format(errX))

    ax[1].plot(Y, ref_Y, ls='', marker='+', color='C0', label='BDSIM data')
    errY, slopeY, cstY = CalcCorrelChi2(df, bpm, ref_bpm, 'Y', 'Y')
    ax[1].plot(Y, slopeY * Y + cstY, color='C2', label='Fit: $\chi^2$ = {}'.format(errY))

    ax[0].ticklabel_format(axis="x", style="sci", scilimits=(-6, 0))
    ax[0].set_xlabel("{} [m]".format('X'))
    ax[0].set_ylabel("{}_ref [m]".format('X'))
    ax[0].legend()

    ax[1].ticklabel_format(axis="x", style="sci", scilimits=(-6, 0))
    ax[1].set_xlabel("{} [m]".format('Y'))
    ax[1].set_ylabel("{}_ref [m]".format('Y'))
    ax[1].legend()


def PlotTheoryFitAndChi2(df, coord, ref_bpm, ref_coord, figsize=[14, 6]):
    fig, ax = plotOptions(figsize=figsize, rows_colums=[1, 1], sharex=False, height_ratios=None)

    S, Curve_Chi2_bdsim = getChi2(df, coord, ref_bpm, ref_coord)
    _plt.plot(S, Curve_Chi2_bdsim(S), ls='', color='C3', marker='+', label='$\chi^2$ for ${{{c}}}$ from BDSIM'.format(c=ref_coord))

    ax.axvline(df.loc[df.index.get_level_values('BPM') == ref_bpm][['S']].to_numpy().flatten()[0], ls='--', color='C2', label='Reference Sampler')

    _plt.ylabel('$\chi^2$')
    _plt.xlabel('$s$ [m]')
    _plt.legend()


def buildPositionMatrix(df_reduced, coord):
    nb_particles = df_reduced.index.levshape[1]
    M = df_reduced[coord].to_numpy().reshape((-1, nb_particles)).transpose()
    return M


def buildMatrixAndVectorForSVD(df, refbpmname, coord='X'):
    df_ref = df.loc[df.index.get_level_values('BPM') == refbpmname][['X', 'Y']]
    df_matrix = df.loc[df.index.get_level_values('BPM') != refbpmname][['X', 'Y']]

    M_X = buildPositionMatrix(df_matrix, 'X')
    M_Y = buildPositionMatrix(df_matrix, 'Y')
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


def plot2CurvesSameAxis(X, Y1, Y2, labelX='X', labelY='Y', legend1='Y1', legend2='Y2',
                        ls1='-', ls2='-', color1='C0', color2='C1', markersize=15, markeredgewidth=2, ticksType='sci', printLegend=True):
    _plt.plot(X, Y1, ls1, color=color1, markersize=markersize, markeredgewidth=markeredgewidth, label=legend1)
    _plt.plot(X, Y2, ls2, color=color2, markersize=markersize, markeredgewidth=markeredgewidth, label=legend2)
    _plt.ylabel(labelY)
    _plt.xlabel(labelX)
    _plt.ticklabel_format(axis="y", style=ticksType, scilimits=(0, 0))
    if printLegend:
        _plt.legend()


def plotJitterAndNoise(df, ex=3.58e-11, ey=3.58e-11, esprd=1e-6, height_ratios=None,
                       plotAngle=False, plotSigma=False, plotBeta=False, plotDisp=False, plotNoise=False, plotMean=False, figsize=[14, 6]):
    S = df.S.unique()
    Jitter_X, Noise_X = calcJitterAndNoise(df, 'X')
    Jitter_Y, Noise_Y = calcJitterAndNoise(df, 'Y')

    rows_colums = [1+plotAngle+plotSigma+plotBeta+plotDisp+plotNoise+plotMean, 1]
    fig, ax = plotOptions(figsize=figsize, rows_colums=rows_colums, sharex='all', height_ratios=height_ratios)

    spnum = 1
    _plt.subplot(rows_colums[0], rows_colums[1], spnum)
    plot2CurvesSameAxis(S, Jitter_X, Jitter_Y, ls1='+-', ls2='+-', legend1=r'$\sigma_{J,X}$', legend2=r'$\sigma_{J,Y}$', labelX='$S$ [m]', labelY='Jitter [m]')
    if plotNoise:
        spnum += 1
        _plt.subplot(rows_colums[0], rows_colums[1], spnum)
        plot2CurvesSameAxis(S, Noise_X, Noise_Y, ls1='+-', ls2='+-', legend1=r'$\sigma_{N,X}$', legend2=r'$\sigma_{N,X}$', labelX='$S$ [m]', labelY='Noise [m]')

    fig.align_labels()


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

# =============================================


def Gamma(E, m=0.511e6):
    return E / m


def Beta(E, m=0.511e6):
    gamma = Gamma(E, m)
    return _np.sqrt(1 - (1 / (gamma * gamma)))


def convertEnergyLab2Rest(E0, Elab, alpha, m=0.511e6):
    gamma = Gamma(E0, m)
    beta = Beta(E0, m)
    return Elab*gamma*(1.0 - beta * _np.cos(alpha))


# to get the laser photon energy to the rest frame,
# alpha = laser lab frame angle to e beam
def doppler(Elab, gam, beta, alpha):
    return Elab*gam*(1.0 - beta * _np.cos(alpha))


# rest frame photon energy in
# scattering angle from MC accept reject
def scatteredEnergy(E0, theta):
    m = 0.511e6
    return E0 / (1 + (E0 / m) * (1 - _np.cos(theta)))


# Compton differential cross-section
def differentialCrossSec(E0, theta):
    Ep = scatteredEnergy(E0, theta)
    r0 = 2.817940545232519E-12
    multiplier = ((r0 * r0) / 2) * (Ep / E0) * (Ep / E0)
    terms = (E0 / Ep + Ep / E0 - _np.sin(theta) * _np.sin(theta))
    return multiplier*terms


# accept reject for a given rest frame photon energy
def acceptReject(E0):
    theta = _np.arccos(1 - 2 * _np.random.rand())
    knMax = differentialCrossSec(E0, 0)
    knTheta = differentialCrossSec(E0, theta)
    knRand = _np.random.rand() * knMax
    if knRand < knTheta:
        result = theta
    else:
        result = acceptReject(E0)
    return result
